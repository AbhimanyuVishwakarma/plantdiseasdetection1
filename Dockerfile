# Use the official TensorFlow image as a base
FROM tensorflow/tensorflow:2.10.0

# Set the working directory
WORKDIR /app

# Copy your application code and model files
COPY . /app

# Install additional Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir gradio google-generativeai python-dotenv gunicorn \
    && pip install --force-reinstall "protobuf>=3.9.2,<3.20"

# Expose the port your app runs on (change if needed)
EXPOSE 8080

# Start the app using gunicorn
CMD ["gunicorn", "PlantdiseaseDetectionApp.app_new1:app", "--bind", "0.0.0.0:8080"]
