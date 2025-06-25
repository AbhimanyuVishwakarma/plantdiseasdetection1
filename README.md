# Plant Disease Detection App

A web application for detecting plant diseases from leaf images using deep learning models. Built with Gradio for the interface and TensorFlow/Keras for model training and inference.

## Features
- Upload plant leaf images to detect diseases.
- Uses custom-trained models (MobileNetV2, ResNet50, InceptionV3, EfficientNetB0) on the PlantVillage dataset.
- Compares predictions from your plant disease model and standard ImageNet models.
- Gradio interface with public sharing enabled.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/plant-disease-detection-app.git
cd plant-disease-detection-app/PlantdiseaseDetectionApp
```

### 2. Install Requirements
It is recommended to use a virtual environment (e.g., conda or venv).
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
- Place `PlantVillage.zip` in the `PlantdiseaseDetectionApp` directory.
- The script will automatically extract it if not already extracted.

### 4. Train the Models
```bash
python train_plantvillage_model.py
```
This will train and save multiple models for plant disease detection.

### 5. Run the App
```bash
python app_new1.py
```
- The app will launch locally and provide a public URL (using Gradio's `share=True`).
- Open the provided link in your browser to use the app.

## Deployment
- You can deploy this app on platforms like Hugging Face Spaces, Google Colab, or any cloud VM that supports Python and Gradio.
- For GitHub deployment, push your code and models to your repository.

## Project Structure
```
PlantdiseaseDetectionApp/
  app_new1.py
  train_plantvillage_model.py
  requirements.txt
  plant_disease_model.h5
  PlantVillage.zip
  PlantVillage/
  ...
```

## Credits
- PlantVillage dataset
- TensorFlow, Keras, Gradio

## License
MIT License
