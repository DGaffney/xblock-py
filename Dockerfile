# Use the official Python 3.9 slim image
FROM python:3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME_LARGE="xblock-large-patch3-224"
ENV MODEL_NAME=$MODEL_NAME_LARGE
ENV MODEL_PATH="/app/models"

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    requests \
    pillow \
    runpod \
    aiohttp

# Copy your Python script into the container
COPY server.py .

# Download the model during the build
# RUN mkdir -p $MODEL_PATH && \
#     python -c "\
# from transformers import AutoModelForImageClassification, AutoFeatureExtractor;\
# model = AutoModelForImageClassification.from_pretrained('howdyaendra/$MODEL_NAME', cache_dir='$MODEL_PATH');\
# feature_extractor = AutoFeatureExtractor.from_pretrained('howdyaendra/$MODEL_NAME', cache_dir='$MODEL_PATH');"

# Set the cache directory to the model path using HF_HOME
ENV HF_HOME=$MODEL_PATH
RUN apt-get update && apt-get install -y nano tmux rsync cron
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN pip install --no-cache-dir runpod
ENTRYPOINT ["/entrypoint.sh"]
CMD ["./entrypoint.sh"]
