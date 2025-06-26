import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model
import json
import requests
import gdown

# Load environment variables from a .env file
load_dotenv()

# Configure the GenerativeAI API key using the loaded environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Initialize the GenerativeModel with the specified model name, configuration, and safety settings
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
# Function to read image data from a file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Replace Gemini model with a generic model interface

def is_leaf_label(label):
    # Expanded keywords for better leaf/plant detection
    leaf_keywords = [
        'leaf', 'leaves', 'foliage', 'plant', 'tree', 'herb', 'shrub', 'flora',
        'vegetation', 'branch', 'stem', 'vine', 'sapling', 'seedling',
        # Add common plant/leaf names from ImageNet
        'maple', 'oak', 'ash', 'willow', 'poplar', 'birch', 'elm', 'sycamore',
        'beech', 'chestnut', 'linden', 'mulberry', 'fig', 'cypress', 'acacia',
        'eucalyptus', 'bamboo', 'banana', 'cabbage', 'lettuce', 'spinach',
        'kale', 'mint', 'basil', 'parsley', 'cilantro', 'thyme', 'rosemary',
        'sage', 'dill', 'oregano', 'lavender', 'ivy', 'clover', 'fern', 'moss'
    ]
    return any(keyword in label.lower() for keyword in leaf_keywords)

def generate_model_response(prompt, image_path):
    # Only use fine-tuned models for predictions
    return predict_with_finetuned_model(image_path)

# Initial input prompt for the plant pathologist
input_prompt = """
As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

**Analysis Guidelines:**

1. **Disease Identification:** Examine the provided information or samples to identify and characterize plant diseases accurately.

2. **Detailed Findings:** Provide in-depth findings on the nature and extent of the identified plant diseases, including affected plant parts, symptoms, and potential causes.

3. **Next Steps:** Outline the recommended course of action for managing and controlling the identified plant diseases. This may involve treatment options, preventive measures, or further investigations.

4. **Recommendations:** Offer informed recommendations for maintaining plant health, preventing disease spread, and optimizing overall plant well-being.

5. **Important Note:** As a plant pathologist, your insights are vital for informed decision-making in agriculture and plant management. Your response should be thorough, concise, and focused on plant health.

**Disclaimer:**
*"Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Consult with qualified agricultural experts before implementing any strategies or treatments."*

Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, adhering to the structured 
"""

# Load class indices from file (generated during training)
indices_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_indices.json')
with open(indices_path, 'r') as f:
    class_indices = json.load(f)
# Ensure class_names is ordered by index
class_names = [None] * len(class_indices)
for k, v in class_indices.items():
    class_names[v] = k

# Only keep InceptionV3 for fine-tuned predictions
MODEL_PATHS = {
    'InceptionV3': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'InceptionV3_plant_disease_model.h5'),
}

MODEL_URLS = {
    'InceptionV3': 'https://drive.google.com/uc?id=1o0jDryVGXmbVxl2WOdPrS90Fm3LeOp9o'
}
for name, url in MODEL_URLS.items():
    path = MODEL_PATHS[name]
    if not os.path.exists(path):
        print(f"Downloading {name} model with gdown...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        print(f"{name} model downloaded.")

missing_models = [name for name, path in MODEL_PATHS.items() if not os.path.exists(path)]
if missing_models:
    print(f"Warning: The following fine-tuned model files are missing and will not be used: {', '.join(missing_models)}")

fine_tuned_models = {name: load_model(path) for name, path in MODEL_PATHS.items() if os.path.exists(path)}

# Use InceptionV3 model for PlantVillage prediction
inception_model = fine_tuned_models['InceptionV3'] if 'InceptionV3' in fine_tuned_models else None

def predict_plant_disease(image_path):
    if inception_model is None:
        return "InceptionV3 model not loaded."
    img = Image.open(image_path).convert('RGB').resize((299, 299))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = inception_model.predict(x)
    class_id = np.argmax(preds)
    confidence = np.max(preds) * 100
    class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
    return f"PlantVillage Prediction: {class_name}\nConfidence: {confidence:.2f}%"

def predict_with_finetuned_model(image_path):
    results = {}
    for name in MODEL_PATHS.keys():
        if name in fine_tuned_models:
            model = fine_tuned_models[name]
            try:
                img = Image.open(image_path).convert('RGB').resize((299, 299))
                x = np.array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                preds = model.predict(x)
                class_id = np.argmax(preds)
                confidence = np.max(preds) * 100
                class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
                results[name] = f"Prediction: {class_name}\nConfidence: {confidence:.2f}%"
            except Exception as e:
                results[name] = f"Error during prediction: {str(e)}"
        else:
            results[name] = "Model file missing."
    summary = '\n\n'.join([f"{model}:\n{result}" for model, result in results.items()])
    return summary

# Update process_uploaded_files to use fine-tuned models for all predictions

def process_uploaded_files(files):
    file_path = files[0].name if files else None
    plant_response = predict_plant_disease(file_path) if file_path else None
    general_response = predict_with_finetuned_model(file_path) if file_path else None
    # Extract confidence from plant_response
    confidence = None
    if plant_response and 'Confidence:' in plant_response:
        try:
            confidence_str = plant_response.split('Confidence:')[1].split('%')[0].strip()
            confidence = float(confidence_str)
        except Exception:
            confidence = None
    if confidence is not None and confidence < 60:
        combined_response = "invalid"
    else:
        combined_response = f"PlantVillage Model Prediction:\n{plant_response}\n\n---\n\nOther ML Model Predictions (Fine-tuned):\n{general_response}" if plant_response and general_response else plant_response or general_response
    return file_path, combined_response

# Gradio interface setup
with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    # Upload button for user to provide images
    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types=["image"],
        file_count="multiple",
    )
     # Set up the upload button to trigger the processing function
    upload_button.upload(process_uploaded_files, upload_button, combined_output)

# Launch the Gradio interface with debug mode enabled and public sharing
# You will get a public URL when running locally or on cloud

demo.launch(debug=True, share=True)
