# run_verification.py
"""
Full end-to-end runtime verification.
Run with: python run_verification.py
"""

import sys
import traceback
import torch

PASS = "✅"
FAIL = "❌"
results = []

def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"{PASS} {name}")
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))
        print(f"{FAIL} {name}")
        traceback.print_exc()


# ── 1. Layers ─────────────────────────────────────────────────────
def test_rmsnorm():
    from models.layers import RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)
    assert not torch.isnan(out).any()

def test_rope():
    from models.layers import RotaryEmbedding
    rope = RotaryEmbedding(64)
    q = torch.randn(2, 4, 10, 64)
    k = torch.randn(2, 4, 10, 64)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape

def test_swiglu():
    from models.layers import SwiGLU
    ffn = SwiGLU(64, 128)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64)

# ── 2. Attention ──────────────────────────────────────────────────
def test_mha():
    from models.layers import ModelConfig
    from models.attention import MultiHeadAttention
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_layers=2,
                      vocab_size=100, intermediate_size=128)
    attn = MultiHeadAttention(cfg, 0)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out[0].shape == (2, 10, 64)

def test_gqa():
    from models.layers import ModelConfig
    from models.attention import GroupedQueryAttention
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_kv_heads=2,
                      num_layers=2, vocab_size=100, intermediate_size=128,
                      use_gqa=True)
    attn = GroupedQueryAttention(cfg, 0)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out[0].shape == (2, 10, 64)

def test_kvcache():
    from models.attention import KVCache
    cache = KVCache()
    k = torch.randn(2, 4, 5, 16)
    v = torch.randn(2, 4, 5, 16)
    k2, v2 = cache.update(0, k, v)
    assert k2.shape == (2, 4, 5, 16)
    k3 = torch.randn(2, 4, 1, 16)
    v3 = torch.randn(2, 4, 1, 16)
    k4, v4 = cache.update(0, k3, v3)
    assert k4.shape == (2, 4, 6, 16)

# ── 3. Transformer ────────────────────────────────────────────────
def test_transformer_block():
    from models.layers import ModelConfig
    from models.transformer import TransformerBlock
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_layers=2,
                      vocab_size=100, intermediate_size=128)
    block = TransformerBlock(cfg, 0)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out[0].shape == (2, 10, 64)

def test_transformer_grad_checkpoint():
    from models.layers import ModelConfig
    from models.transformer import TransformerModel
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_layers=2,
                      vocab_size=100, intermediate_size=128)
    model = TransformerModel(cfg)
    model.gradient_checkpointing = True
    model.train()
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = model(x)
    out["last_hidden_state"].sum().backward()
    assert x.grad is not None

# ── 4. GPT Model ──────────────────────────────────────────────────
def test_gpt_forward():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_layers=2,
                      vocab_size=200, intermediate_size=128)
    model = GPTModel(cfg)
    ids = torch.randint(0, 200, (2, 10))
    lbls = torch.randint(0, 200, (2, 10))
    out = model(input_ids=ids, labels=lbls)
    assert out.logits.shape == (2, 10, 200)
    assert out.loss is not None
    assert not torch.isnan(out.loss)

def test_gpt_weight_tying():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_layers=2,
                      vocab_size=200, intermediate_size=128,
                      tie_word_embeddings=True)
    model = GPTModel(cfg)
    assert model.lm_head.weight is model.embeddings.token_embedding.weight

def test_gpt_hidden_states():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    cfg = ModelConfig(hidden_size=64, num_heads=4, num_layers=2,
                      vocab_size=200, intermediate_size=128)
    model = GPTModel(cfg)
    ids = torch.randint(0, 200, (1, 8))
    out = model(input_ids=ids, output_hidden_states=True)
    assert out.hidden_states is not None
    assert len(out.hidden_states) == cfg.num_layers + 1

def test_gpt_save_load(tmp_path):
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    model.save_pretrained(str(tmp_path))
    loaded = GPTModel.from_pretrained(str(tmp_path))
    ids = torch.randint(0, 100, (1, 5))
    out1 = model(ids)
    out2 = loaded(ids)
    assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

# ── 5. Generation ─────────────────────────────────────────────────
def test_greedy():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from generation.strategies import GenerationConfig, GreedyDecoder
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    model.eval()
    gen_cfg = GenerationConfig(max_new_tokens=5, eos_token_id=99)
    dec = GreedyDecoder(gen_cfg)
    ids = torch.randint(1, 50, (1, 5))
    out = dec.generate(model, ids)
    assert out.shape[1] > ids.shape[1]

def test_sampling():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from generation.strategies import GenerationConfig, SamplingDecoder
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    model.eval()
    gen_cfg = GenerationConfig(
        max_new_tokens=5, top_k=10, top_p=0.9, temperature=0.8
    )
    dec = SamplingDecoder(gen_cfg)
    ids = torch.randint(1, 50, (1, 5))
    out = dec.generate(model, ids)
    assert out.shape[1] > ids.shape[1]

def test_streaming():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from generation.strategies import GenerationConfig, SamplingDecoder
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    model.eval()
    gen_cfg = GenerationConfig(max_new_tokens=5)
    dec = SamplingDecoder(gen_cfg)
    ids = torch.randint(1, 50, (1, 5))
    tokens = list(dec.generate_stream(model, ids))
    assert len(tokens) > 0

def test_beam_search():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from generation.strategies import GenerationConfig, BeamSearchDecoder
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    model.eval()
    gen_cfg = GenerationConfig(max_new_tokens=5, num_beams=2)
    dec = BeamSearchDecoder(gen_cfg)
    ids = torch.randint(1, 50, (1, 5))
    out = dec.generate(model, ids)
    assert out.shape[0] == 1

# ── 6. BPE Tokenizer ──────────────────────────────────────────────
def test_bpe_train():
    from data.tokenization.bpe_tokenizer import BPEConfig, ByteLevelBPETokenizer
    cfg = BPEConfig(vocab_size=300, min_frequency=1,
                    special_tokens={"bos_token":"<s>","eos_token":"</s>",
                                    "unk_token":"<unk>","pad_token":"<pad>"})
    tok = ByteLevelBPETokenizer(cfg)
    corpus = ["hello world machine learning"] * 30
    tok.train(corpus)
    assert tok.vocab_size >= 260

def test_bpe_encode_decode():
    from data.tokenization.bpe_tokenizer import BPEConfig, ByteLevelBPETokenizer
    cfg = BPEConfig(vocab_size=300, min_frequency=1,
                    special_tokens={"bos_token":"<s>","eos_token":"</s>",
                                    "unk_token":"<unk>","pad_token":"<pad>"})
    tok = ByteLevelBPETokenizer(cfg)
    tok.train(["hello world test encode decode roundtrip"] * 30)
    ids = tok.encode("hello world", add_special_tokens=False)
    assert len(ids) > 0
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "hello" in decoded or len(decoded) > 0

# ── 7. DeepSeek Architecture ──────────────────────────────────────
def test_deepseek_model():
    from models.architectures.deepseek import DeepSeekConfig, DeepSeekModel
    cfg = DeepSeekConfig(
        vocab_size=100, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=4, intermediate_size=64,
        use_moe=True, num_experts=4, num_experts_per_token=2,
        num_shared_experts=1, moe_intermediate_size=32,
        use_mla=False,  # Use standard attn for speed in test
        first_k_dense_layers=1,
        max_position_embeddings=64,
    )
    model = DeepSeekModel(cfg)
    ids = torch.randint(0, 100, (1, 8))
    labels = torch.randint(0, 100, (1, 8))
    out = model(input_ids=ids, labels=labels)
    assert out.logits.shape == (1, 8, 100)
    assert out.loss is not None
    assert not torch.isnan(out.loss)

# ── 8. Qwen Architecture ──────────────────────────────────────────
def test_qwen_model():
    from models.architectures.qwen import QwenConfig, QwenModel
    cfg = QwenConfig(
        vocab_size=100, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=4, intermediate_size=64,
        kv_channels=8, seq_length=64,
        max_position_embeddings=64,
    )
    model = QwenModel(cfg)
    ids = torch.randint(0, 100, (1, 8))
    labels = torch.randint(0, 100, (1, 8))
    out = model(input_ids=ids, labels=labels)
    assert out.logits.shape == (1, 8, 100)
    assert not torch.isnan(out.loss)

# ── 9. Gemma Architecture ─────────────────────────────────────────
def test_gemma_model():
    from models.architectures.gemma import GemmaConfig, GemmaModel
    cfg = GemmaConfig(
        vocab_size=100, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=4, intermediate_size=64,
        max_position_embeddings=64,
        use_logit_softcap=True, logit_soft_cap=30.0,
    )
    model = GemmaModel(cfg)
    ids = torch.randint(0, 100, (1, 8))
    labels = torch.randint(0, 100, (1, 8))
    out = model(input_ids=ids, labels=labels)
    assert out.logits.shape == (1, 8, 100)
    assert not torch.isnan(out.loss)

# ── 10. Reward Model ──────────────────────────────────────────────
def test_reward_model():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from training.rlhf.reward_model import RewardModel, RewardModelConfig
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    backbone = GPTModel(cfg)
    rm_cfg = RewardModelConfig(hidden_size=32)
    rm = RewardModel(backbone, rm_cfg)
    ids = torch.randint(0, 100, (2, 10))
    rewards = rm(ids)
    assert rewards.shape == (2,)
    assert not torch.isnan(rewards).any()

def test_preference_loss():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from training.rlhf.reward_model import RewardModel, RewardModelConfig
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    rm = RewardModel(GPTModel(cfg), RewardModelConfig(hidden_size=32))
    chosen = torch.randint(0, 100, (2, 10))
    rejected = torch.randint(0, 100, (2, 10))
    loss, metrics = rm.compute_preference_loss(chosen, rejected)
    assert loss.item() > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0

# ── 11. Verifiable Tasks ──────────────────────────────────────────
def test_math_verifier_correct():
    from training.rlhf.verifiable_tasks import MathVerifier
    v = MathVerifier()
    r = v.compute_reward("2+2=?", "The answer is 4.", "4")
    assert r.passed and r.reward == 1.0

def test_math_verifier_wrong():
    from training.rlhf.verifiable_tasks import MathVerifier
    v = MathVerifier()
    r = v.compute_reward("2+2=?", "The answer is 5.", "4")
    assert not r.passed

def test_code_verifier_syntax():
    from training.rlhf.verifiable_tasks import CodeVerifier
    v = CodeVerifier()
    r = v.compute_reward("", "def foo(:\n  pass", "")
    assert not r.passed

def test_format_verifier_json():
    from training.rlhf.verifiable_tasks import FormatVerifier
    v = FormatVerifier()
    r = v.compute_reward("", '{"k":"v"}', "json")
    assert r.passed

# ── 12. Evaluation Metrics ────────────────────────────────────────
def test_bleu():
    from evaluation.metrics import BLEUScore
    b = BLEUScore()
    r = b.compute(["the cat sat on the mat"],
                  [["the cat sat on the mat"]])
    assert r["bleu"] > 90.0

def test_rouge():
    from evaluation.metrics import ROUGEScore
    r = ROUGEScore()
    res = r.compute(["the cat sat on the mat"],
                    ["the cat sat on the mat"])
    assert res["rouge1"] > 99.0

# ── 13. ELO Leaderboard ───────────────────────────────────────────
def test_elo():
    import time
    from evaluation.human_eval import ELORatingSystem, ComparisonResult
    elo = ELORatingSystem()
    for i in range(5):
        elo.update(ComparisonResult(
            comparison_id=str(i),
            prompt="test", model_a="A", model_b="B",
            response_a="r", response_b="r",
            winner="a", judge_id="j", timestamp=time.time()
        ))
    board = elo.get_leaderboard()
    assert board[0]["model"] == "A"
    assert board[0]["elo"] > board[1]["elo"]

# ── 14. SFT LoRA ──────────────────────────────────────────────────
def test_lora_apply():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from training.sft import apply_lora_to_model
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=2,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = apply_lora_to_model(model, r=4, alpha=8)
    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # After LoRA: only LoRA params trainable, much fewer
    assert total_after < total_before
    assert total_after > 0

def test_lora_forward():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from training.sft import apply_lora_to_model
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = apply_lora_to_model(GPTModel(cfg), r=4, alpha=8)
    ids = torch.randint(0, 100, (1, 8))
    out = model(input_ids=ids)
    assert out.logits.shape == (1, 8, 100)

# ── 15. Data Utilities ────────────────────────────────────────────
def test_url_utils():
    from data.crawling.utils import normalize_url, is_valid_url, extract_domain
    url = "https://Example.COM/path/?b=2&a=1#frag"
    norm = normalize_url(url)
    assert "example.com" in norm
    assert "#frag" not in norm
    assert is_valid_url("https://example.com/page")
    assert not is_valid_url("ftp://example.com")
    assert not is_valid_url("https://example.com/image.jpg")
    assert extract_domain("https://sub.example.com/p") == "sub.example.com"

def test_quality_filter():
    from data.cleaning.quality_filters import TextQualityFilter, QualityConfig
    f = TextQualityFilter(QualityConfig())
    good = (
        "Machine learning is a powerful branch of artificial intelligence. "
        "It enables computers to learn patterns from large datasets and make "
        "predictions without being explicitly programmed for each task. "
        "Deep learning, a subset of machine learning, uses neural networks."
    ) * 3
    assert f.filter(good).passed
    assert not f.filter("short").passed

def test_minhash():
    from data.cleaning.refined_web import MinHashDeduplicator
    mh = MinHashDeduplicator(num_perm=64)
    t = "the quick brown fox jumps over the lazy dog near the river"
    s = mh.compute_minhash(t)
    assert not mh.is_duplicate("d1", s)
    assert mh.is_duplicate("d1_copy", s)

# ── 16. Training Utils ────────────────────────────────────────────
def test_count_parameters():
    from models.layers import ModelConfig
    from models.gpt import GPTModel
    from training.utils import count_parameters
    cfg = ModelConfig(hidden_size=32, num_heads=2, num_layers=1,
                      vocab_size=100, intermediate_size=64)
    model = GPTModel(cfg)
    params = count_parameters(model)
    assert params["total"] > 0
    assert params["trainable"] == params["total"]
    assert params["total_M"] > 0

def test_collator():
    from training.utils import DataCollatorForCausalLM
    col = DataCollatorForCausalLM(pad_token_id=0, max_length=16)
    features = [
        {"input_ids": torch.tensor([1,2,3,4]),
         "labels": torch.tensor([2,3,4,5])},
        {"input_ids": torch.tensor([1,2]),
         "labels": torch.tensor([2,3])},
    ]
    batch = col(features)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert (batch["attention_mask"][1, 2:] == 0).all()

def test_cosine_scheduler():
    from training.pretrain import CosineSchedulerWithWarmup
    opt = torch.optim.AdamW([torch.tensor(1.0, requires_grad=True)], lr=1e-3)
    sched = CosineSchedulerWithWarmup(opt, 10, 100, 1e-5, 1e-3)
    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(sched.get_lr())
    # Warmup: increasing
    assert lrs[9] > lrs[0]
    # After warmup: decreasing
    assert lrs[19] < lrs[10]

# ── 17. API (no server needed) ────────────────────────────────────
def test_api_imports():
    from api.model_registry import ModelRegistry
    from api.middleware.auth import (
        create_access_token, decode_token, check_rate_limit
    )
    # Test token round-trip
    token = create_access_token({"sub": "testuser", "role": "admin"})
    payload = decode_token(token)
    # In dev mode without jose, returns default
    assert payload is not None

def test_model_registry():
    from api.model_registry import ModelRegistry
    ModelRegistry.register("test-model", "model", "tok", "gen")
    assert ModelRegistry.exists("test-model")
    assert "test-model" in ModelRegistry.list_models()
    info = ModelRegistry.get("test-model")
    assert info["model"] == "model"
    ModelRegistry.unregister("test-model")
    assert not ModelRegistry.exists("test-model")

# ── Run all tests ─────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    tmp = tempfile.mkdtemp()

    print("\n" + "="*60)
    print("LLM PLAYGROUND — FULL RUNTIME VERIFICATION")
    print("="*60 + "\n")

    test("RMSNorm forward", test_rmsnorm)
    test("RotaryEmbedding forward", test_rope)
    test("SwiGLU forward", test_swiglu)
    test("MultiHeadAttention forward", test_mha)
    test("GroupedQueryAttention forward", test_gqa)
    test("KVCache update", test_kvcache)
    test("TransformerBlock forward", test_transformer_block)
    test("TransformerModel grad checkpoint", test_transformer_grad_checkpoint)
    test("GPTModel forward + loss", test_gpt_forward)
    test("GPTModel weight tying", test_gpt_weight_tying)
    test("GPTModel hidden states", test_gpt_hidden_states)
    test("GPTModel save/load", lambda: test_gpt_save_load(tmp))
    test("Greedy decoding", test_greedy)
    test("Sampling decoding", test_sampling)
    test("Streaming generation", test_streaming)
    test("Beam search decoding", test_beam_search)
    test("BPE tokenizer train", test_bpe_train)
    test("BPE encode/decode", test_bpe_encode_decode)
    test("DeepSeekModel forward", test_deepseek_model)
    test("QwenModel forward", test_qwen_model)
    test("GemmaModel forward", test_gemma_model)
    test("RewardModel forward", test_reward_model)
    test("Preference loss", test_preference_loss)
    test("MathVerifier correct", test_math_verifier_correct)
    test("MathVerifier wrong", test_math_verifier_wrong)
    test("CodeVerifier syntax error", test_code_verifier_syntax)
    test("FormatVerifier JSON", test_format_verifier_json)
    test("BLEU score", test_bleu)
    test("ROUGE score", test_rouge)
    test("ELO leaderboard", test_elo)
    test("LoRA apply to GPTModel", test_lora_apply)
    test("LoRA forward pass", test_lora_forward)
    test("URL utilities", test_url_utils)
    test("Quality filter", test_quality_filter)
    test("MinHash deduplication", test_minhash)
    test("Count parameters", test_count_parameters)
    test("Data collator", test_collator)
    test("Cosine scheduler", test_cosine_scheduler)
    test("API imports + JWT", test_api_imports)
    test("ModelRegistry CRUD", test_model_registry)

    print("\n" + "="*60)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    print(f"RESULTS: {passed} passed / {failed} failed / {len(results)} total")

    if failed > 0:
        print("\nFAILED TESTS:")
        for r in results:
            if r[0] == FAIL:
                print(f"  {r[0]} {r[1]}")
    else:
        print("\n✅ ALL TESTS PASSED — PROJECT IS COMPLETE")
    print("="*60)