"""Evaluation panel UI component."""

import json

import gradio as gr
import requests


def build_eval_panel(api_base: str) -> None:
    """Build the evaluation tab panel."""

    gr.Markdown("### 📊 Evaluation Benchmarks")

    with gr.Row():
        with gr.Column():
            eval_task = gr.Dropdown(
                choices=[
                    "bleu",
                    "rouge",
                    "perplexity",
                    "mmlu",
                    "gsm8k",
                ],
                value="bleu",
                label="Evaluation Task",
            )
            eval_model = gr.Textbox(
                value="demo",
                label="Model Name",
            )

            with gr.Accordion("Input Data", open=True):
                hyp_input = gr.Textbox(
                    label="Hypotheses (one per line)",
                    lines=5,
                    placeholder="Generated text 1\nGenerated text 2",
                )
                ref_input = gr.Textbox(
                    label="References (one per line)",
                    lines=5,
                    placeholder="Reference 1\nReference 2",
                )
                texts_input = gr.Textbox(
                    label="Texts for Perplexity",
                    lines=3,
                    placeholder="Text to compute perplexity...",
                )

            run_eval_btn = gr.Button(
                "▶️ Run Evaluation", variant="primary"
            )

        with gr.Column():
            eval_results = gr.JSON(label="Results")
            eval_status = gr.Textbox(
                label="Status", interactive=False
            )

    def run_eval(task, texts, hyps, refs, model_name):
        payload = {
            "model_name": model_name,
            "task": task,
            "texts": [t.strip() for t in texts.split("\n") if t.strip()],
            "hypotheses": [h.strip() for h in hyps.split("\n") if h.strip()],
            "references": [r.strip() for r in refs.split("\n") if r.strip()],
        }

        try:
            resp = requests.post(
                f"{api_base}/evaluate", json=payload, timeout=10
            )
            job = resp.json()
            job_id = job.get("job_id", "")

            import time
            for _ in range(30):
                time.sleep(1)
                poll = requests.get(
                    f"{api_base}/evaluate/{job_id}", timeout=10
                )
                data = poll.json()
                if data["status"] == "completed":
                    return data.get("results", {}), "✅ Complete"
                elif data["status"] == "failed":
                    return {}, f"❌ Failed: {data.get('error')}"

            return {}, "⏱️ Timeout"
        except Exception as e:
            return {}, f"❌ Error: {e}"

    run_eval_btn.click(
        run_eval,
        [eval_task, texts_input, hyp_input, ref_input, eval_model],
        [eval_results, eval_status],
    )