"""Generation parameter controls panel."""

import gradio as gr


def build_param_panel() -> dict:
    """
    Build the parameter control panel.

    Returns:
        dict of Gradio components keyed by parameter name
    """
    with gr.Column(scale=1, elem_classes=["param-panel"]):
        gr.Markdown("### ⚙️ Generation Parameters")

        model_choice = gr.Dropdown(
            choices=["demo", "llm-playground-7b"],
            value="demo",
            label="🤖 Model",
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.7,
            step=0.05,
            label="🌡️ Temperature",
            info="Higher = more random",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="🎯 Top-p (nucleus)",
            info="Nucleus sampling threshold",
        )
        top_k = gr.Slider(
            minimum=0,
            maximum=200,
            value=50,
            step=1,
            label="🔝 Top-k",
            info="Restrict to top-k tokens",
        )
        max_tokens = gr.Slider(
            minimum=32,
            maximum=4096,
            value=512,
            step=32,
            label="📝 Max New Tokens",
        )
        rep_penalty = gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.05,
            label="🔄 Repetition Penalty",
            info=">1 reduces repetition",
        )
        decode_strategy = gr.Radio(
            choices=["sampling", "greedy", "beam_search"],
            value="sampling",
            label="🎲 Decoding Strategy",
        )
        num_beams = gr.Slider(
            minimum=1,
            maximum=8,
            value=4,
            step=1,
            label="🔭 Beam Width",
            visible=False,
        )

        # Show beam width only when beam_search selected
        decode_strategy.change(
            fn=lambda s: gr.update(visible=(s == "beam_search")),
            inputs=decode_strategy,
            outputs=num_beams,
        )

    return {
        "model_choice": model_choice,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "rep_penalty": rep_penalty,
        "decode_strategy": decode_strategy,
        "num_beams": num_beams,
    }