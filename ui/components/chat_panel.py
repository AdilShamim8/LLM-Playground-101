"""
Chat panel UI component.
Extracted from ui/app.py for modularity.
"""

import gradio as gr


def build_chat_panel(
    stream_fn,
    user_message_fn,
    clear_fn,
):
    """
    Build the main chat panel with message history,
    input box, and controls.

    Args:
        stream_fn: Async generator for bot responses
        user_message_fn: Function to handle user message
        clear_fn: Function to clear history

    Returns:
        (chatbot, user_input, chat_history, send_btn, clear_btn)
    """
    with gr.Column(scale=3):
        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
            show_copy_button=True,
            avatar_images=("👤", "🤖"),
            type="messages",
        )
        with gr.Row():
            user_input = gr.Textbox(
                label="Your message",
                placeholder="Ask anything... (Shift+Enter for newline)",
                lines=3,
                scale=4,
                show_label=False,
                container=False,
            )
            send_btn = gr.Button(
                "Send 🚀",
                variant="primary",
                scale=1,
                min_width=80,
            )
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
            system_prompt_box = gr.Textbox(
                value=(
                    "You are a helpful, harmless, "
                    "and honest AI assistant."
                ),
                label="System Prompt",
                lines=2,
                scale=3,
            )

    chat_history = gr.State([])

    send_btn.click(
        user_message_fn,
        [user_input, chat_history, system_prompt_box],
        [chat_history, user_input],
    ).then(
        stream_fn,
        chat_history,
        chat_history,
    )

    user_input.submit(
        user_message_fn,
        [user_input, chat_history, system_prompt_box],
        [chat_history, user_input],
    ).then(
        stream_fn,
        chat_history,
        chat_history,
    )

    clear_btn.click(
        clear_fn,
        outputs=[chat_history, chatbot],
    )

    return chatbot, user_input, chat_history, send_btn, clear_btn