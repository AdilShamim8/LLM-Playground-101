"""
Gradio UI for LLM Playground.
Full-featured chatbot interface with:
- Model selection
- Generation parameter controls
- Streaming chat
- Evaluation tab
- Training monitoring
"""

import inspect
import json
import os
import time
from typing import Iterator

import gradio as gr
import requests
from loguru import logger

DEFAULT_API_BASE = "http://localhost:8000/api/v1"
API_BASE = os.getenv("API_BASE", DEFAULT_API_BASE)
CHAT_STREAM = os.getenv("CHAT_STREAM", "false").lower() in (
    "1", "true", "yes"
)

def _build_theme():
    try:
        if hasattr(gr.themes, "GoogleFont"):
            return gr.themes.Base(
                primary_hue="teal",
                secondary_hue="orange",
                neutral_hue="gray",
                font=[
                    gr.themes.GoogleFont("IBM Plex Sans"),
                    "Segoe UI",
                    "sans-serif",
                ],
                font_mono=[
                    gr.themes.GoogleFont("IBM Plex Mono"),
                    "Consolas",
                    "monospace",
                ],
            )
        return gr.themes.Base(
            primary_hue="teal",
            secondary_hue="orange",
            neutral_hue="gray",
        )
    except Exception:
        return None


UI_THEME = _build_theme()
UI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-0: #f5efe6;
    --bg-1: #efe5d6;
    --ink: #14181b;
    --ink-muted: #516066;
    --accent: #0e6b68;
    --accent-2: #d5753d;
    --accent-3: #0b4947;
    --card: rgba(255, 255, 255, 0.9);
    --card-strong: #ffffff;
    --card-dark: #111e24;
    --card-dark-2: #0c161b;
    --border: rgba(20, 24, 27, 0.12);
    --shadow: 0 24px 50px rgba(14, 30, 37, 0.16);
    --radius-lg: 22px;
    --radius-md: 14px;
}

.gradio-container {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    color: var(--ink);
    background:
        radial-gradient(900px 420px at 10% -10%, rgba(14, 107, 104, 0.18), transparent 60%),
        radial-gradient(700px 380px at 90% 0%, rgba(213, 117, 61, 0.18), transparent 60%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
    min-height: 100vh;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(1400px 520px at 50% -20%, rgba(255, 255, 255, 0.55), transparent 55%),
        radial-gradient(320px 200px at 14% 18%, rgba(14, 107, 104, 0.12), transparent 70%),
        radial-gradient(380px 240px at 88% 16%, rgba(213, 117, 61, 0.16), transparent 75%);
    pointer-events: none;
    z-index: 0;
}

.gradio-container > * {
    position: relative;
    z-index: 1;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container .title {
    font-family: "Fraunces", "Times New Roman", serif;
    letter-spacing: 0.2px;
}

.hero {
    margin: 14px 0 8px;
    text-align: center;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--ink);
}

.hero-sub {
    color: var(--ink-muted);
    font-size: 1rem;
    margin-top: 6px;
}

.hero-meta {
    margin-top: 12px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(20, 24, 27, 0.08);
    color: var(--ink-muted);
    font-size: 0.85rem;
}

.chat-panel {
    position: relative;
    padding: 16px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: var(--shadow);
    overflow: hidden;
    animation: float-in 0.7s ease both;
}

.chat-panel::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(240px 160px at 8% 8%, rgba(14, 107, 104, 0.09), transparent 60%),
        radial-gradient(220px 160px at 92% 0%, rgba(213, 117, 61, 0.09), transparent 60%);
    pointer-events: none;
}

.chat-panel > * {
    position: relative;
}

.chat-container { max-height: 520px; overflow-y: auto; }

.chat-container .message {
    border-radius: 14px;
    padding: 10px 12px;
    font-size: 0.95rem;
    line-height: 1.45;
}

.chat-container .message.user {
    background: rgba(14, 107, 104, 0.12);
    border: 1px solid rgba(14, 107, 104, 0.2);
}

.chat-container .message.bot {
    background: rgba(213, 117, 61, 0.12);
    border: 1px solid rgba(213, 117, 61, 0.2);
}

.composer textarea {
    background: #f7f5f1;
    border-radius: 12px;
    border: 1px solid rgba(20, 24, 27, 0.14);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}

.send-btn button {
    background: var(--accent);
    color: #fff;
    border: none;
    box-shadow: 0 14px 28px rgba(14, 107, 104, 0.25);
}

.send-btn button:hover {
    background: var(--accent-3);
}

.param-panel {
    padding: 16px;
    border-radius: var(--radius-lg);
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: linear-gradient(180deg, var(--card-dark), var(--card-dark-2));
    color: #f4f4f2;
    box-shadow: var(--shadow);
    animation: float-in 0.7s ease both;
}

.param-panel h3,
.param-panel label,
.param-panel .label {
    color: #e6efe9;
}

.param-panel input,
.param-panel textarea,
.param-panel select {
    background: rgba(255, 255, 255, 0.08);
    color: #f4f4f2;
    border: 1px solid rgba(255, 255, 255, 0.14);
}

.param-panel button,
.param-panel .gr-button {
    background: var(--accent);
    border: none;
}

.param-panel button:hover,
.param-panel .gr-button:hover {
    filter: brightness(1.05);
}

@keyframes float-in {
    from { transform: translateY(16px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

@media (max-width: 900px) {
    .chat-panel, .param-panel { border-radius: var(--radius-md); }
    .hero-title { font-size: 2.1rem; }
}
"""


def _supports_param(callable_obj, param_name: str) -> bool:
    try:
        return param_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


CHATBOT_SUPPORTS_TYPE = _supports_param(gr.Chatbot, "type")
def _chatbot_expects_messages() -> bool:
    try:
        desc = getattr(gr.Chatbot(), "_value_description", "")
        return "message" in str(desc).lower()
    except Exception:
        return True


CHATBOT_EXPECTS_MESSAGES = _chatbot_expects_messages()
LAUNCH_SUPPORTS_THEME = _supports_param(gr.Blocks.launch, "theme")
LAUNCH_SUPPORTS_CSS = _supports_param(gr.Blocks.launch, "css")
BLOCKS_SUPPORTS_THEME = _supports_param(gr.Blocks, "theme")
BLOCKS_SUPPORTS_CSS = _supports_param(gr.Blocks, "css")


def _messages_to_pairs(messages: list[dict]) -> list[list[str]]:
    pairs: list[list[str]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pairs.append([content, ""])
        elif role == "assistant":
            if not pairs:
                pairs.append(["", content])
            else:
                pairs[-1][1] = content
    return pairs


def _format_chatbot_output(messages: list[dict]):
    if CHATBOT_SUPPORTS_TYPE or CHATBOT_EXPECTS_MESSAGES:
        return messages
    return _messages_to_pairs(messages)


# ─── API Helpers ──────────────────────────────────────────────────

def stream_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    decoding_strategy: str,
) -> Iterator[str]:
    """Stream chat from API."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "decoding_strategy": decoding_strategy,
        "stream": True,
    }

    try:
        with requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.exception("stream_chat failed")
        yield f"\n[Error: {e}]"


def run_evaluation(
    task: str,
    texts_input: str,
    hypotheses_input: str,
    references_input: str,
    model_name: str
) -> str:
    """Run evaluation job via API."""
    payload = {
        "model_name": model_name,
        "task": task,
        "texts": [t.strip() for t in texts_input.split("\n") if t.strip()],
        "hypotheses": [
            h.strip()
            for h in hypotheses_input.split("\n") if h.strip()
        ],
        "references": [
            r.strip()
            for r in references_input.split("\n") if r.strip()
        ],
    }

    try:
        resp = requests.post(
            f"{API_BASE}/evaluate", json=payload
        )
        job = resp.json()
        job_id = job.get("job_id")
        if not job_id:
            return json.dumps({"error": "Missing job_id", "raw": job})

        # Poll for result
        for _ in range(20):
            time.sleep(1)
            poll = requests.get(
                f"{API_BASE}/evaluate/{job_id}"
            )
            result = poll.json()
            if result["status"] in ("completed", "failed"):
                return json.dumps(result, indent=2)

        return "Timeout waiting for results."
    except Exception as e:
        logger.exception("run_evaluation failed")
        return f"Error: {e}"


# ─── Gradio Interface ─────────────────────────────────────────────

def build_ui():
    blocks_kwargs = {
        "title": "LLM Playground",
    }
    if UI_THEME is not None and BLOCKS_SUPPORTS_THEME and not LAUNCH_SUPPORTS_THEME:
        blocks_kwargs["theme"] = UI_THEME
    if BLOCKS_SUPPORTS_CSS and not LAUNCH_SUPPORTS_CSS:
        blocks_kwargs["css"] = UI_CSS

    with gr.Blocks(**blocks_kwargs) as demo:

        gr.HTML(
            f"""
            <div class="hero">
                <div class="hero-title">LLM Playground</div>
                <div class="hero-sub">Build, evaluate, and steer models end-to-end.</div>
                <div class="hero-meta">
                    <span class="pill">API: {API_BASE}</span>
                    <span class="pill">Mode: UI</span>
                </div>
            </div>
            """
        )

        with gr.Tabs():

            # ── Chat Tab ─────────────────────────────────────────
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3, elem_classes=["chat-panel"]):
                        chatbot_kwargs = {
                            "label": "Conversation",
                            "height": 480,
                            "elem_classes": ["chat-container"],
                        }
                        if CHATBOT_SUPPORTS_TYPE:
                            chatbot_kwargs["type"] = "messages"
                        chatbot = gr.Chatbot(**chatbot_kwargs)
                        with gr.Row():
                            user_input = gr.Textbox(
                                label="Your message",
                                placeholder="Ask anything...",
                                lines=3,
                                scale=4,
                                elem_classes=["composer"],
                            )
                            send_btn = gr.Button(
                                "Send 🚀",
                                variant="primary",
                                scale=1,
                                elem_classes=["send-btn"],
                            )
                        with gr.Row():
                            clear_btn = gr.Button(
                                "🗑️ Clear", scale=1
                            )
                            system_prompt = gr.Textbox(
                                value=(
                                    "You are a helpful, harmless, "
                                    "and honest AI assistant."
                                ),
                                label="System Prompt",
                                lines=2,
                                scale=3
                            )

                    with gr.Column(scale=1, elem_classes=["param-panel"]):
                        gr.Markdown("### ⚙️ Parameters")
                        model_choice = gr.Dropdown(
                            choices=["demo"],
                            value="demo",
                            label="Model"
                        )
                        refresh_models_btn = gr.Button(
                            "Refresh models",
                            variant="secondary"
                        )
                        temperature = gr.Slider(
                            0.0, 2.0, value=0.7, step=0.05,
                            label="🌡️ Temperature"
                        )
                        top_p = gr.Slider(
                            0.0, 1.0, value=0.9, step=0.05,
                            label="🎯 Top-p"
                        )
                        top_k = gr.Slider(
                            0, 100, value=50, step=1,
                            label="🔝 Top-k"
                        )
                        max_tokens = gr.Slider(
                            32, 2048, value=512, step=32,
                            label="📝 Max tokens"
                        )
                        rep_penalty = gr.Slider(
                            1.0, 2.0, value=1.1, step=0.05,
                            label="🔄 Repetition Penalty"
                        )
                        decode_strategy = gr.Radio(
                            ["sampling", "greedy", "beam_search"],
                            value="sampling",
                            label="🎲 Decoding Strategy"
                        )

                # State
                chat_history = gr.State([])

                def refresh_models():
                    try:
                        resp = requests.get(
                            f"{API_BASE}/models",
                            timeout=10
                        )
                        data = resp.json()
                        models = data.get("models", [])
                        if "demo" not in models:
                            models = ["demo"] + models
                        value = models[0] if models else "demo"
                        return gr.update(choices=models, value=value)
                    except Exception:
                        return gr.update()

                def user_message(
                    user_msg: str,
                    history: list,
                    sys_prompt: str
                ):
                    if not user_msg.strip():
                        return history, "", _format_chatbot_output(history)
                    history.append({
                        "role": "user",
                        "content": user_msg
                    })
                    return history, "", _format_chatbot_output(history)

                def bot_response_once(
                    history: list,
                    sys_prompt: str,
                    model: str,
                    temp: float,
                    tp: float,
                    tk: int,
                    mt: int,
                    rp: float,
                    strategy: str,
                ):
                    if not history:
                        return history, _format_chatbot_output(history)

                    messages = []
                    if sys_prompt:
                        messages.append({
                            "role": "system",
                            "content": sys_prompt
                        })
                    messages.extend(history)

                    try:
                        payload = {
                            "model": model,
                            "messages": messages,
                            "temperature": temp,
                            "top_p": tp,
                            "top_k": tk,
                            "max_tokens": mt,
                            "repetition_penalty": rp,
                            "decoding_strategy": strategy,
                            "stream": False,
                        }
                        resp = requests.post(
                            f"{API_BASE}/chat/completions",
                            json=payload,
                            timeout=60
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                    except Exception as e:
                        logger.exception("non-stream chat failed")
                        content = f"[Error: {e}]"

                    history.append({
                        "role": "assistant",
                        "content": content or "*(No response generated)*"
                    })
                    return history, _format_chatbot_output(history)

                def bot_response_stream(
                    history: list,
                    sys_prompt: str,
                    model: str,
                    temp: float,
                    tp: float,
                    tk: int,
                    mt: int,
                    rp: float,
                    strategy: str,
                ):
                    if not history:
                        yield history, _format_chatbot_output(history)
                        return

                    messages = []
                    if sys_prompt:
                        messages.append({
                            "role": "system",
                            "content": sys_prompt
                        })
                    messages.extend(history)

                    # Add placeholder for streaming
                    history.append({
                        "role": "assistant",
                        "content": ""
                    })

                    full_response = ""
                    for token in stream_chat(
                        messages, model, temp, tp, tk, mt, rp, strategy
                    ):
                        full_response += token
                        history[-1]["content"] = full_response
                        yield history, _format_chatbot_output(history)

                    if not full_response:
                        history[-1]["content"] = (
                            "*(No response generated)*"
                        )
                    yield history, _format_chatbot_output(history)

                bot_handler = (
                    bot_response_stream if CHAT_STREAM else bot_response_once
                )

                # Wire up events
                send_btn.click(
                    user_message,
                    [user_input, chat_history, system_prompt],
                    [chat_history, user_input, chatbot],
                    queue=False,
                ).then(
                    bot_handler,
                    [
                        chat_history, system_prompt,
                        model_choice, temperature, top_p,
                        top_k, max_tokens, rep_penalty, decode_strategy
                    ],
                    [chat_history, chatbot],
                    queue=False,
                )

                user_input.submit(
                    user_message,
                    [user_input, chat_history, system_prompt],
                    [chat_history, user_input, chatbot],
                    queue=False,
                ).then(
                    bot_handler,
                    [
                        chat_history, system_prompt,
                        model_choice, temperature, top_p,
                        top_k, max_tokens, rep_penalty, decode_strategy
                    ],
                    [chat_history, chatbot],
                    queue=False,
                )

                clear_btn.click(
                    lambda: ([], _format_chatbot_output([]), ""),
                    outputs=[chat_history, chatbot, user_input],
                    queue=False,
                )

                refresh_models_btn.click(
                    refresh_models,
                    outputs=model_choice,
                    queue=False,
                )
                demo.load(refresh_models, outputs=model_choice, queue=False)

            # ── Generation Lab Tab ────────────────────────────────
            with gr.TabItem("🧪 Generation Lab"):
                gr.Markdown(
                    "### Compare generation strategies side-by-side"
                )
                with gr.Row():
                    lab_prompt = gr.Textbox(
                        label="Prompt",
                        lines=4,
                        placeholder="Enter prompt..."
                    )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎯 Greedy")
                        greedy_out = gr.Textbox(
                            label="Output", lines=6
                        )
                        greedy_btn = gr.Button("Generate")
                    with gr.Column():
                        gr.Markdown("#### 🎲 Sampling (top-p)")
                        sampling_out = gr.Textbox(
                            label="Output", lines=6
                        )
                        sampling_btn = gr.Button("Generate")
                    with gr.Column():
                        gr.Markdown("#### 🔭 Beam Search")
                        beam_out = gr.Textbox(
                            label="Output", lines=6
                        )
                        beam_btn = gr.Button("Generate")

                def generate_with_strategy(
                    prompt: str, strategy: str
                ) -> str:
                    try:
                        if not prompt.strip():
                            return "Please enter a prompt."
                        payload = {
                            "model": "demo",
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "stream": False,
                            "decoding_strategy": strategy,
                            "temperature": 0.7 if strategy == "sampling" else 1.0,
                            "max_tokens": 200,
                        }
                        resp = requests.post(
                            f"{API_BASE}/chat/completions",
                            json=payload,
                            timeout=60
                        )
                        data = resp.json()
                        return (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "Error")
                        )
                    except Exception as e:
                        logger.exception("generate_with_strategy failed")
                        return f"Error: {e}"

                greedy_btn.click(
                    lambda p: generate_with_strategy(p, "greedy"),
                    lab_prompt, greedy_out,
                    queue=False,
                )
                sampling_btn.click(
                    lambda p: generate_with_strategy(p, "sampling"),
                    lab_prompt, sampling_out,
                    queue=False,
                )
                beam_btn.click(
                    lambda p: generate_with_strategy(p, "beam"),
                    lab_prompt, beam_out,
                    queue=False,
                )

            # ── Evaluation Tab ────────────────────────────────────
            with gr.TabItem("📊 Evaluation"):
                gr.Markdown("### Run Evaluation Benchmarks")
                with gr.Row():
                    with gr.Column():
                        eval_task = gr.Dropdown(
                            choices=[
                                "bleu", "rouge",
                                "perplexity", "mmlu", "gsm8k"
                            ],
                            value="bleu",
                            label="Evaluation Task"
                        )
                        eval_model = gr.Textbox(
                            value="demo",
                            label="Model Name"
                        )
                        hyp_input = gr.Textbox(
                            label="Hypotheses (one per line)",
                            lines=5,
                            placeholder="Generated text 1\nGenerated text 2"
                        )
                        ref_input = gr.Textbox(
                            label="References (one per line)",
                            lines=5,
                            placeholder="Reference text 1\nReference text 2"
                        )
                        texts_input = gr.Textbox(
                            label="Texts for Perplexity",
                            lines=3,
                            placeholder="Texts to evaluate..."
                        )
                        run_eval_btn = gr.Button(
                            "▶️ Run Evaluation",
                            variant="primary"
                        )

                    with gr.Column():
                        eval_results = gr.JSON(
                            label="Results",
                        )
                        eval_plot = gr.Plot(
                            label="Metrics Visualization"
                        )

                def display_eval_results(
                    task, texts, hyps, refs, model
                ):
                    result_str = run_evaluation(
                        task, texts, hyps, refs, model
                    )
                    try:
                        result_dict = json.loads(result_str)
                        return result_dict, None
                    except Exception:
                        return {"raw": result_str}, None

                run_eval_btn.click(
                    display_eval_results,
                    [
                        eval_task, texts_input,
                        hyp_input, ref_input, eval_model
                    ],
                    [eval_results, eval_plot],
                    queue=False,
                )

            # ── Model Info Tab ────────────────────────────────────
            with gr.TabItem("ℹ️ Model Info"):
                gr.Markdown("### Model Architecture & Training Info")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
**Architecture Supported:**
- GPT-2 (learned absolute positions, LayerNorm, GeLU)
- LLaMA-style (RoPE, RMSNorm, SwiGLU, GQA)
- DeepSeek-style (GQA, RoPE, SwiGLU, MoE ready)

**Training Pipeline:**
1. **Pre-training**: Causal LM on web data
2. **SFT**: Instruction following with loss masking
3. **Reward Model**: Bradley-Terry on preference pairs
4. **PPO**: KL-constrained policy optimization

**Generation Strategies:**
- Greedy: argmax at each step
- Top-k: restrict to k most likely tokens
- Top-p (Nucleus): dynamic vocabulary by CDF
- Beam Search: explore multiple hypotheses
                        """)
                    with gr.Column():
                        gr.Markdown("""
**Data Pipeline:**
- Web crawler (async, polite)
- Common Crawl (WARC/WET)
- RefinedWeb cleaning (MinHash dedup)
- FineWeb educational scoring
- BPE tokenization (GPT-2 style)

**Evaluation:**
- Perplexity (language modeling)
- BLEU (translation, generation)
- ROUGE (summarization)
- MMLU (knowledge, 57 subjects)
- GSM8K (mathematical reasoning)

**API:**
- OpenAI-compatible REST API
- Streaming SSE support
- JWT authentication
- Background eval jobs
                        """)

        gr.Markdown(
            "---\n"
            "Built with ❤️ | "
            "LLM Playground v1.0 | "
            "Production-Level Implementation"
        )

    return demo


def launch():
    demo = build_ui()
    if CHAT_STREAM and hasattr(demo, "queue"):
        demo.queue()
    ui_host = os.getenv("UI_HOST", "0.0.0.0")
    ui_port = int(os.getenv("UI_PORT", "7860"))
    launch_kwargs = {
        "server_name": ui_host,
        "server_port": ui_port,
        "share": False,
        "debug": False,
        "show_error": True,
        "favicon_path": None,
    }
    if UI_THEME is not None and LAUNCH_SUPPORTS_THEME:
        launch_kwargs["theme"] = UI_THEME
    if LAUNCH_SUPPORTS_CSS:
        launch_kwargs["css"] = UI_CSS
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch()