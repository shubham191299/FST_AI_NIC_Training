import gradio as gr
import ollama

MODEL = "llama3.2:3b"
SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer clearly."
MAX_HISTORY = 10  # last N messages kept, keeps replies fast


def respond(message, history):
    # Keep only role/content and trim old turns
    past = [{"role": m["role"], "content": m["content"]} for m in history]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + past[-MAX_HISTORY:]
        + [{"role": "user", "content": message}]
    )

    reply = ""
    for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
        reply += chunk["message"]["content"]
        yield reply


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="My Local Chatbot",
    description="Runs fully offline with Ollama.",
)

if __name__ == "__main__":
    demo.launch()