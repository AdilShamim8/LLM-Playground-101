"""Model information and management panel."""

import gradio as gr
import requests


def build_model_panel(api_base: str) -> None:
    """Build the model info and management panel."""

    gr.Markdown("### 🤖 Model Management")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Load Model")
            load_model_name = gr.Textbox(
                label="Model Name",
                placeholder="my-gpt-7b",
            )
            load_model_path = gr.Textbox(
                label="Model Path",
                placeholder="./checkpoints/sft/final",
            )
            load_tokenizer_path = gr.Textbox(
                label="Tokenizer Path (optional)",
                placeholder="./tokenizers/bpe",
            )
            load_device = gr.Dropdown(
                choices=["auto", "cuda", "cpu"],
                value="auto",
                label="Device",
            )
            load_dtype = gr.Dropdown(
                choices=["bfloat16", "float16", "float32"],
                value="bfloat16",
                label="Dtype",
            )
            load_btn = gr.Button("⬆️ Load Model", variant="primary")
            load_status = gr.Textbox(
                label="Load Status", interactive=False
            )

        with gr.Column():
            gr.Markdown("#### Loaded Models")
            refresh_btn = gr.Button("🔄 Refresh")
            models_list = gr.JSON(label="Models")

            gr.Markdown("#### Architecture Info")
            gr.Markdown("""
**Supported Architectures:**
| Family | Key Features |
|--------|-------------|
| GPT-2 | Abs. pos, LayerNorm, GeLU |
| LLaMA | RoPE, RMSNorm, SwiGLU, GQA |
| DeepSeek | MLA, MoE, GQA, RoPE |
| Qwen | Dynamic NTK, LogN-attn |
| Gemma | Softcap, Emb-scale, GeGLU |
            """)

    def load_model_api(name, path, tok_path, device, dtype):
        payload = {
            "model_name": name,
            "model_path": path,
            "tokenizer_path": tok_path or None,
            "device": device,
            "dtype": dtype,
        }
        try:
            resp = requests.post(
                f"{api_base}/models/load",
                json=payload,
                headers={"X-API-Key": "dev-key-123"},
                timeout=120,
            )
            data = resp.json()
            if resp.status_code == 200:
                return f"✅ Loaded: {name} ({data.get('num_parameters', '?')} params)"
            return f"❌ Error: {data.get('detail', 'Unknown')}"
        except Exception as e:
            return f"❌ Connection error: {e}"

    def list_models_api():
        try:
            resp = requests.get(
                f"{api_base}/models/list",
                headers={"X-API-Key": "dev-key-123"},
                timeout=10,
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    load_btn.click(
        load_model_api,
        [load_model_name, load_model_path, load_tokenizer_path,
         load_device, load_dtype],
        load_status,
    )

    refresh_btn.click(list_models_api, outputs=models_list)