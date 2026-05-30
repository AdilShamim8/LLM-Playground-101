"""
Utility functions for web crawling.
URL normalization, validation, domain extraction,
result merging, and deduplication helpers.
"""

import hashlib
import re
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    urlencode,
    parse_qsl,
)
from typing import Optional
from loguru import logger


# ── URL Utilities ─────────────────────────────────────────────────

EXCLUDED_EXTENSIONS = frozenset([
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
    ".webp", ".ico", ".mp3", ".mp4", ".avi", ".mov", ".wmv",
    ".flv", ".zip", ".tar", ".gz", ".rar", ".7z", ".exe",
    ".dmg", ".pkg", ".deb", ".rpm", ".iso", ".bin", ".dat",
    ".css", ".js", ".xml", ".rss", ".atom", ".json",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
])

JUNK_URL_PATTERNS = [
    re.compile(p) for p in [
        r"/wp-login\.php",
        r"/wp-admin/",
        r"/cart/?$",
        r"/checkout/?$",
        r"\?add-to-cart=",
        r"/search\?",
        r"/tag/",
        r"/author/",
        r"#",
        r"javascript:",
        r"mailto:",
        r"tel:",
    ]
]


def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication.
    - Lowercases scheme and host
    - Removes default ports (80, 443)
    - Sorts query parameters
    - Removes fragment
    - Strips trailing slash from path
    """
    try:
        parsed = urlparse(url.strip())

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Clean path
        path = parsed.path.rstrip("/") or "/"

        # Sort query parameters for canonical form
        query_params = sorted(parse_qsl(parsed.query))
        query = urlencode(query_params)

        normalized = urlunparse((
            scheme, netloc, path, parsed.params, query, ""
        ))
        return normalized

    except Exception:
        return url


def is_valid_url(
    url: str,
    allowed_domains: Optional[list[str]] = None,
    excluded_extensions: Optional[frozenset] = None,
) -> bool:
    """
    Check if a URL is worth crawling.
    Filters out media files, admin pages, and junk patterns.
    """
    if not url or not isinstance(url, str):
        return False

    if excluded_extensions is None:
        excluded_extensions = EXCLUDED_EXTENSIONS

    try:
        parsed = urlparse(url)

        # Must have http/https scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Must have a netloc
        if not parsed.netloc:
            return False

        # Check file extension
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in excluded_extensions):
            return False

        # Check junk patterns
        if any(p.search(url) for p in JUNK_URL_PATTERNS):
            return False

        # Domain whitelist
        if allowed_domains:
            domain = extract_domain(url)
            if domain not in allowed_domains:
                return False

        return True

    except Exception:
        return False


def extract_domain(url: str) -> str:
    """Extract the domain (netloc) from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def extract_base_domain(url: str) -> str:
    """
    Extract base domain (e.g., 'example.com' from
    'sub.example.com').
    """
    domain = extract_domain(url)
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def url_to_hash(url: str) -> str:
    """Stable hash of a normalized URL."""
    normalized = normalize_url(url)
    return hashlib.sha256(normalized.encode()).hexdigest()


def make_absolute_url(base_url: str, relative_url: str) -> str:
    """Resolve a relative URL against a base URL."""
    try:
        return urljoin(base_url, relative_url)
    except Exception:
        return relative_url


# ── Result Utilities ──────────────────────────────────────────────

def deduplicate_urls(urls: list[str]) -> list[str]:
    """
    Deduplicate a list of URLs using normalized form.
    Preserves order (first occurrence kept).
    """
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            result.append(url)
    return result


def merge_crawl_results(
    results_list: list[list[dict]],
) -> list[dict]:
    """
    Merge multiple crawl result lists, deduplicating
    by content hash.
    """
    seen_hashes: set[str] = set()
    merged: list[dict] = []

    for results in results_list:
        for page in results:
            h = page.get("content_hash", "")
            if not h:
                h = hashlib.sha256(
                    page.get("text", "").encode()
                ).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(page)

    logger.info(
        f"Merged {sum(len(r) for r in results_list)} pages "
        f"-> {len(merged)} unique"
    )
    return merged


def filter_urls_by_domain(
    urls: list[str],
    allowed_domains: list[str],
) -> list[str]:
    """Keep only URLs from allowed domains."""
    return [
        url for url in urls
        if extract_domain(url) in allowed_domains
    ]


def estimate_crawl_size(
    seed_urls: list[str],
    max_depth: int,
    avg_links_per_page: int = 50,
) -> int:
    """Estimate total pages that could be crawled."""
    total = 0
    for depth in range(max_depth + 1):
        total += len(seed_urls) * (avg_links_per_page ** depth)
    return total


def chunk_urls(
    urls: list[str], chunk_size: int
) -> list[list[str]]:
    """Split URL list into chunks for parallel crawling."""
    return [
        urls[i:i + chunk_size]
        for i in range(0, len(urls), chunk_size)
    ]


def build_robots_url(base_url: str) -> str:
    """Build robots.txt URL from any page URL."""
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def extract_links_from_text(text: str) -> list[str]:
    """Extract all URLs mentioned in plain text."""
    url_pattern = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+'
    )
    return url_pattern.findall(text)