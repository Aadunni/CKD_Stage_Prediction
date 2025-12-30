# Use a base image compatible with TensorFlow and Streamlit
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app
COPY . .

# Cloud Run expects port 8080
EXPOSE 8080

# Start Streamlit and make it listen on Cloud Run's port
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
