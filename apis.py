from flask import Flask, jsonify, request
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import logging
from data_preprocess import clean_text

# Step 1: Initialize Flask App
app = Flask(__name__)


# Step 2: Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 3: Loading the model and tokenizer

MODEL_PATH = "model_output/fine_tuned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    logger.info("Loaded the model and tokenizer successfully")
except Exception as e:
    logger.error(f"Error loading the model and tokenizer: {e}")
    raise RuntimeError(f"Failed to load model and tokenizer: {e}")

# step 4 Prediction class mapping for human readability

class_mapping =  {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}

# Step 5: Prepare input same way you did during model training

def prepare_input(text, max_length=512):
    """
    Prepares the text input for model inference.
    """
    cleaned_text = clean_text(text) 
    tokenized_input = tokenizer(
        cleaned_text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return tokenized_input

# Step 6: Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to perform model inference.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "REQUEST must be JSON"}), 400
        
        data = request.get_json()
        text = data.get("text", None)

        if not text or not isinstance(text, str):
            return jsonify({"error": "Invalid input. 'text' field is required and must be a string."}), 400
        
        input_text = prepare_input(text)
        input_text = {key: value.to(DEVICE) for key, value in input_text.items()}

        # Perform Inference
        with torch.no_grad():
            outputs = model(**input_text)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_class].item()

        # Map the predicted class to a human-readable label
        predicted_label = class_mapping.get(predicted_class, "Unknown")

        # Return Prediction
        return jsonify({
            "text": text,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence
        }), 200

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An internal error occurred during prediction. Please try again."}), 500
    

# Step &: Run the Flask App

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)