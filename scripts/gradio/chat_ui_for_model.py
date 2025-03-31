# chat_ui_for_model.py
import argparse
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch a chat interface for an AI model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    args = parser.parse_args()

    # Load model and tokenizer
    checkpoint = args.model
    device = "cuda"
    print(f"Loading model {checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    print("Model loaded successfully!")

    @spaces.GPU(duration=120)
    def predict(message, history, temperature, top_p):
        history.append({"role": "user", "content": message})
        input_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs, max_new_tokens=1024, temperature=float(temperature), top_p=float(top_p), do_sample=True
        )
        decoded = tokenizer.decode(outputs[0])
        response = decoded.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
        return response

    # Create the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown(f"# Chat with {checkpoint}")
        chatbot = gr.ChatInterface(
            predict,
            additional_inputs=[
                gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
                gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P"),
            ],
            type="messages",
        )

    demo.launch()


if __name__ == "__main__":
    main()
