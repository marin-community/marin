"""
Usage:
python3 -m marin.processing.fasttext.demo --model-path /nlp/scr/cychou/fasttext_model.bin --share
"""

import argparse
import gradio as gr
import fasttext


# Define the prediction function
def predict(text):
    text = text.replace("\n", " ")

    try:
        prediction = model.predict(text)[0][0]  # Get the prediction label
    except Exception as e:
        prediction = f"Model errored out: {str(e)}"

    return prediction


def build_demo():
    # Create the Gradio interface
    demo = gr.Interface(
        fn=predict,  # Function to be called
        inputs=gr.Textbox(),  # User input
        outputs=gr.Textbox(),  # Model prediction output
    )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="The path to the FastText model")
    parser.add_argument("--share", action="store_true", help="Whether to share the link publicly")

    args = parser.parse_args()

    # Load the pre-trained FastText model
    model = fasttext.load_model(args.model_path)

    demo = build_demo()
    # Launch the Gradio app
    demo.launch(share=args.share)
