#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run_pipeline.sh — End-to-end LLM training pipeline
# ═══════════════════════════════════════════════════════════════════════
#
#  This script runs the complete LLM pipeline in order:
#    1. Verify installation (run_verification.py)
#    2. Collect data (web crawl + Common Crawl)
#    3. Clean data (RefinedWeb / Dolma / FineWeb)
#    4. Train BPE tokenizer
#    5. Tokenize corpus to binary shards
#    6. Pre-train the model
#    7. Supervised fine-tune (SFT)
#    8. Train reward model
#    9. RLHF (PPO)
#   10. Evaluate
#
#  Usage:
#    ./scripts/run_pipeline.sh           # Run all steps
#    ./scripts/run_pipeline.sh --step 6  # Run only step 6 (pre-train)
#    ./scripts/run_pipeline.sh --verify   # Run verification only
#
#  Before running: copy .env.example to .env and fill in values.
#

set -euo pipefail
# set -e: exit on error
# set -u: error on undefined variable
# set -o pipefail: fail if any command in a pipe fails

# ── Color codes for readable output ─────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"; echo -e "${BLUE}  STEP $1: $2${NC}"; echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"; }

# ── Check prerequisites ────────────────────────────────────────────────
check_prereqs() {
    log_info "Checking prerequisites..."

    if [ ! -f .env ]; then
        log_error ".env file not found. Copy .env.example to .env and fill in values."
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Install Python 3.10+."
        exit 1
    fi

    log_info "Python: $(python3 --version)"
    log_info "Working directory: $(pwd)"
}

# ── Step 1: Verification ────────────────────────────────────────────────
step_verify() {
    log_step 1 "Runtime Verification"
    python3 run_verification.py
}

# ── Step 2: Data Collection ─────────────────────────────────────────────
step_collect_data() {
    log_step 2 "Data Collection (Web Crawl + Common Crawl)"
    log_warn "This step downloads from the internet. It can take hours and use 100s of GB."
    python3 -c "
from data.crawling.common_crawl import CommonCrawlProcessor, CommonCrawlConfig
config = CommonCrawlConfig(
    crawl_id='CC-MAIN-2023-50',
    num_segments=5,
    max_pages_per_segment=10000,
    languages=['en'],
    output_dir='./data/raw/cc',
)
processor = CommonCrawlProcessor(config)
total = processor.process_and_save('./data/raw/cc/output.jsonl')
print(f'Collected: {total:,} documents')
"
}

# ── Step 3: Data Cleaning ──────────────────────────────────────────────
step_clean_data() {
    log_step 3 "Data Cleaning (RefinedWeb pipeline)"
    python3 -c "
from data.cleaning.refined_web import RefinedWebCleaner, RefinedWebConfig
rw = RefinedWebCleaner(RefinedWebConfig())
stats = rw.process_file('./data/raw/cc/output.jsonl', './data/processed/clean.jsonl')
print(f'Passed: {stats[\"passed\"]}/{stats[\"total\"]}')
"
}

# ── Step 4: Train Tokenizer ─────────────────────────────────────────────
step_train_tokenizer() {
    log_step 4 "Train BPE Tokenizer"
    python3 -c "
import json
from data.tokenization.bpe_tokenizer import BPEConfig, ByteLevelBPETokenizer

config = BPEConfig(vocab_size=50257, min_frequency=2)
tokenizer = ByteLevelBPETokenizer(config)

texts = []
with open('./data/processed/clean.jsonl') as f:
    for line in f:
        texts.append(json.loads(line)['text'])

tokenizer.train(texts)
tokenizer.save('./tokenizers/bpe_50k')
print('Tokenizer saved to ./tokenizers/bpe_50k')
"
}

# ── Step 5: Tokenize Corpus ─────────────────────────────────────────────
step_tokenize_corpus() {
    log_step 5 "Tokenize Corpus to Binary Shards"
    python3 -c "
from data.tokenization.bpe_tokenizer import ByteLevelBPETokenizer, BPEConfig
from data.tokenization.utils import tokenize_corpus_to_bin, split_dataset

tokenizer = ByteLevelBPETokenizer(BPEConfig(vocab_size=50257))
tokenizer.load('./tokenizers/bpe_50k')

stats = tokenize_corpus_to_bin(
    input_path='./data/processed/',
    output_dir='./data/tokenized/',
    tokenizer=tokenizer,
    max_tokens_per_file=100_000_000,
)
print(f'Shards: {stats[\"num_shards\"]}, Tokens: {stats[\"total_tokens\"]:,}')

splits = split_dataset('./data/tokenized/', './data/split/')
print(f'Train: {splits.train_tokens:,} tokens')
"
}

# ── Step 6: Pre-train ───────────────────────────────────────────────────
step_pretrain() {
    log_step 6 "Pre-training"
    log_warn "This step requires a GPU. It can take days for large models."
    python3 -c "
from models import GPTModel, ModelConfig
from training import PretrainTrainer, PretrainConfig, PretrainDataset

model = GPTModel(ModelConfig(
    vocab_size=50257, hidden_size=768, num_layers=12,
    num_heads=12, intermediate_size=3072,
    use_rope=True, use_rmsnorm=True, use_swiglu=True,
))

config = PretrainConfig(
    data_path='./data/split',
    output_dir='./checkpoints/pretrain',
    max_steps=100000,
)
trainer = PretrainTrainer(model, config)
train_data = PretrainDataset(config.data_path, config.max_seq_length, 'train')
val_data = PretrainDataset(config.data_path, config.max_seq_length, 'val')
trainer.train(train_data, val_data)
"
}

# ── Step 7: SFT ────────────────────────────────────────────────────────
step_sft() {
    log_step 7 "Supervised Fine-Tuning (SFT)"
    python3 -c "
from models import GPTModel
from training import SFTTrainer, SFTConfig, SFTDataset

model = GPTModel.from_pretrained('./checkpoints/pretrain/best')
config = SFTConfig(data_path='./data/sft', output_dir='./checkpoints/sft')
dataset = SFTDataset('./data/sft', model.tokenizer, config)
trainer = SFTTrainer(model, model.tokenizer, config)
trainer.train(dataset)
"
}

# ── Step 8: Reward Model ────────────────────────────────────────────────
step_reward_model() {
    log_step 8 "Reward Model Training"
    python3 -c "
from models import GPTModel
from training.rlhf import RewardModel, RewardModelConfig, RewardModelTrainer, PreferenceDataset

backbone = GPTModel.from_pretrained('./checkpoints/sft/final')
rm = RewardModel(backbone, RewardModelConfig())
dataset = PreferenceDataset('./data/preference', model.tokenizer)
trainer = RewardModelTrainer(rm, RewardModelConfig())
trainer.train(dataset)
"
}

# ── Step 9: RLHF (PPO) ──────────────────────────────────────────────────
step_rlhf() {
    log_step 9 "RLHF (PPO)"
    python3 -c "
from models import GPTModel
from training.rlhf import PPOTrainer, PPOConfig, RewardModel

actor = GPTModel.from_pretrained('./checkpoints/sft/final')
ref = GPTModel.from_pretrained('./checkpoints/sft/final')
critic = GPTModel.from_pretrained('./checkpoints/sft/final')

config = PPOConfig(total_steps=1000, output_dir='./checkpoints/ppo')
trainer = PPOTrainer(actor, critic, ref, rm, model.tokenizer, config)
with open('./data/ppo_prompts.txt') as f:
    prompts = f.readlines()
trainer.train(prompts)
"
}

# ── Step 10: Evaluate ───────────────────────────────────────────────────
step_evaluate() {
    log_step 10 "Evaluation"
    python3 -c "
from evaluation.metrics import compute_perplexity
from evaluation.benchmarks import run_benchmarks
print('Running evaluation suite...')
# Add specific evaluation calls here
"
}

# ── Main ────────────────────────────────────────────────────────────────
main() {
    local step=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --step) step="$2"; shift 2 ;;
            --verify) step="1"; shift ;;
            -h|--help)
                echo "Usage: $0 [--step N] [--verify]"
                echo "  --step N   Run only step N (1-10)"
                echo "  --verify   Run verification only (step 1)"
                echo ""
                echo "Steps:"
                echo "  1: Verify installation"
                echo "  2: Collect data"
                echo "  3: Clean data"
                echo "  4: Train tokenizer"
                echo "  5: Tokenize corpus"
                echo "  6: Pre-train model"
                echo "  7: SFT"
                echo "  8: Reward model"
                echo "  9: RLHF (PPO)"
                echo " 10: Evaluate"
                exit 0
                ;;
            *) log_error "Unknown option: $1"; exit 1 ;;
        esac
    done

    check_prereqs

    if [ -n "$step" ]; then
        case $step in
            1) step_verify ;;
            2) step_collect_data ;;
            3) step_clean_data ;;
            4) step_train_tokenizer ;;
            5) step_tokenize_corpus ;;
            6) step_pretrain ;;
            7) step_sft ;;
            8) step_reward_model ;;
            9) step_rlhf ;;
            10) step_evaluate ;;
            *) log_error "Invalid step: $step (must be 1-10)"; exit 1 ;;
        esac
    else
        # Run all steps
        step_verify
        step_collect_data
        step_clean_data
        step_train_tokenizer
        step_tokenize_corpus
        step_pretrain
        step_sft
        step_reward_model
        step_rlhf
        step_evaluate
    fi

    echo ""
    log_info "Pipeline complete!"
}

main "$@"
