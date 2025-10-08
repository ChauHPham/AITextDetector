import gradio as gr
import torch
import os
from ai_text_detector.models import DetectorModel
from ai_text_detector.datasets import DatasetLoader

# Initialize model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the trained model if it exists, otherwise use a base model for demo"""
    global model, tokenizer
    
    model_path = "models/ai_detector"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model = DetectorModel.load(model_path)
        tokenizer = model.tokenizer
    else:
        print("No trained model found. Using base RoBERTa model for demo.")
        # Use a base model for demonstration
        model = DetectorModel("roberta-base")
        tokenizer = model.tokenizer

def detect_text(text):
    """Detect if text is AI-generated or human-written"""
    if not text.strip():
        return "Please enter some text to analyze."
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            human_prob = probabilities[0][0].item()
            ai_prob = probabilities[0][1].item()
        
        # Determine prediction
        if ai_prob > human_prob:
            label = "🤖 AI-generated"
            confidence = ai_prob
        else:
            label = "🧑 Human-written"
            confidence = human_prob
        
        return f"{label} (confidence: {confidence:.1%})"
        
    except Exception as e:
        return f"Error processing text: {str(e)}"

# Load model on startup
load_model()

# Create Gradio interface
with gr.Blocks(title="AI Text Detector", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔍 AI Text Detector")
    gr.Markdown("Paste any text below to detect if it was written by AI or a human.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to analyze",
                placeholder="Enter text here...",
                lines=5,
                max_lines=10
            )
            detect_btn = gr.Button("🔍 Detect", variant="primary")
        
        with gr.Column():
            result_output = gr.Textbox(
                label="Prediction",
                interactive=False,
                lines=3
            )
    
    # Connect the button to the function
    detect_btn.click(
        fn=detect_text,
        inputs=text_input,
        outputs=result_output
    )
    
    # Also detect on Enter key
    text_input.submit(
        fn=detect_text,
        inputs=text_input,
        outputs=result_output
    )
    
    # Add some example texts
    gr.Markdown("### 💡 Try these examples:")
    
    examples = [
        "The sunset painted the sky in hues of crimson and gold, casting long shadows across the meadow.",
        "The quantum tensor optimization algorithm significantly reduced inference latency by 23.7%.",
        "I went to the store yesterday and bought some milk and bread.",
        "The implementation leverages advanced neural architecture search techniques to optimize model performance."
    ]
    
    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=result_output,
        fn=detect_text,
        cache_examples=False
    )

if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
