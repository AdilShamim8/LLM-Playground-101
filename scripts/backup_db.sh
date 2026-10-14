#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  backup_db.sh — PostgreSQL database backup with rotation
# ═══════════════════════════════════════════════════════════════════════
#
#  WHY THIS EXISTS:
#  Without backups, a database corruption or accidental deletion is
#  irreversible. This script:
#  - Creates a timestamped backup of the PostgreSQL database
#  - Compresses it with gzip to save space
#  - Keeps the last 30 days of backups (configurable)
#  - Uploads to S3 if configured (optional)
#
#  Usage:
#    ./scripts/backup_db.sh
#
#  Schedule with cron (daily at 2 AM):
#    0 2 * * * /path/to/scripts/backup_db.sh >> /var/log/db-backup.log 2>&1
#

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
DB_NAME="${POSTGRES_DB:-llm_playground}"
DB_USER="${POSTGRES_USER:-llm_user}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"

# ── Color output ────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BACKUP]${NC} $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Create backup directory ────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"

# ── Generate backup filename ───────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"

log "Starting backup of database: $DB_NAME"

# ── Check if pg_dump is available ──────────────────────────────────────
if ! command -v pg_dump &> /dev/null; then
    fail "pg_dump not found. Install PostgreSQL client tools."
fi

# ── Create backup ──────────────────────────────────────────────────────
log "Creating backup: $BACKUP_FILE"

if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_FILE"; then
    FILESIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    log "Backup completed: $BACKUP_FILE ($FILESIZE)"
else
    fail "Backup failed!"
fi

# ── Clean up old backups ───────────────────────────────────────────────
log "Cleaning up backups older than $RETENTION_DAYS days..."

DELETED=$(find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +"$RETENTION_DAYS" -delete -print | wc -l)

if [ "$DELETED" -gt 0 ]; then
    log "Deleted $DELETED old backup(s)"
else
    log "No old backups to clean up"
fi

# ── List remaining backups ─────────────────────────────────────────────
log "Current backups:"
ls -lh "$BACKUP_DIR"/${DB_NAME}_*.sql.gz 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

# ── Optional: Upload to S3 ──────────────────────────────────────────────
if [ -n "${S3_BACKUP_BUCKET:-}" ]; then
    log "Uploading to S3: s3://$S3_BACKUP_BUCKET/"
    if aws s3 cp "$BACKUP_FILE" "s3://$S3_BACKUP_BUCKET/$(basename "$BACKUP_FILE")"; then
        log "S3 upload completed"
    else
        fail "S3 upload failed!"
    fi
fi

log "Backup process complete!"
