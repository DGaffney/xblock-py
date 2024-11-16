#!/usr/bin/env python
import os
import asyncio
from asyncio import Queue
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import runpod
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import json
from timm import create_model
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Asynchronous HTTP client
import aiohttp

# Set the number of threads to 1
torch.set_num_threads(1)

# Use environment variables
NUM_WORKERS = 100
MODEL_NAME_LARGE = os.getenv('MODEL_NAME_LARGE', 'swin_s3_base_224-xblockm-timm')
MODEL_NAME = MODEL_NAME_LARGE
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models')

# Check if CUDA (GPU) is available; if not, default to CPU
device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define model details
model_id = f"howdyaendra/{MODEL_NAME}"
cache_dir = "./models"
# Download model files
model_weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", cache_dir=cache_dir)
config_path = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)
# Load configuration
with open(config_path) as f:
    config = json.load(f)

num_classes = config.get("num_classes", 13)
# Create the model and load weights
model_name = "swin_s3_base_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def create_model_instance():
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.to(device)
    # Load weights
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def create_text_model_instance():
    return {
        "tokenizer": AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation"),
        "model": AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation").to("cuda")
    }



model_pool = Queue()
for _ in range(int(NUM_WORKERS/5)):
    model_pool.put_nowait(create_model_instance())

embedder_pool = Queue()
for _ in range(int(NUM_WORKERS/5)):
    embedder_pool.put_nowait(create_text_model_instance())

# Image transformations
img_size = (224, 224)
transform = T.Compose([
    T.Resize(img_size),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

LABEL_MAP = {
    "S": "sexual",
    "H": "hate",
    "V": "violence",
    "HR": "harassment",
    "SH": "self-harm",
    "S3": "sexual/minors",
    "H2": "hate/threatening",
    "V2": "violence/graphic",
    "OK":	"OK",
}


def get_text_labels(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits
    # Apply softmax to get probabilities (scores)
    probabilities = logits.softmax(dim=-1).squeeze().tolist()
    # Retrieve the labels
    id2label = model.config.id2label
    labels = [id2label[idx] for idx in range(len(probabilities))]
    # Combine labels and probabilities, then sort
    return dict(zip([LABEL_MAP[e] for e in labels], probabilities))

async def process_single_image(image_url, model, transform, device, config, top_k):
    """
    Process a single image URL to get top-k predictions.
    """
    start_time = time.time()
    try:
        # Download image asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    return {'error': f'Failed to download image. HTTP Status: {response.status}', 'url': image_url}
                content = await response.read()
        
        # Open and process the image
        image = Image.open(BytesIO(content)).convert('RGB')
        cuda_image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(cuda_image)
        probabilities = [float(e) for e in logits.sigmoid().cpu().numpy()[0]]
        label_prob_pairs = list(zip(config["label_names"], probabilities))
        label_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_k_predictions = label_prob_pairs[:top_k]
        return {"image_url": image_url, "labels": {label: prob for label, prob in top_k_predictions}, "time": time.time()-start_time}
    except Exception as e:
        return {'error': str(e), 'url': image_url}

async def process_request(job):
    """
    Asynchronous handler function to process incoming requests.
    Accepts either a single dictionary or a list of dictionaries as input.
    """
    try:
        start_time = time.time()
        input_data = job.get('input', {}).get("jobs")
        # If input_data is a single dict, convert it to a list for unified processing
        if isinstance(input_data, dict):
            input_data = [input_data]
        results = []
        # Acquire models for the entire batch
        embedder = await embedder_pool.get()
        model = await model_pool.get()
        try:
            # Process each item in the input list
            for data in input_data:
                cid = data.get('cid')
                image_urls = data.get('image_urls')
                post_text = data.get('text')
                top_k = data.get('top_k', 13)
                image_results = {}
                text_results = {}
                # Process text if present
                if post_text:
                    text_results = get_text_labels(post_text, embedder["model"], embedder["tokenizer"])
                # Process images if present
                if image_urls:
                    tasks = [
                        process_single_image(url, model, transform, device, config, top_k)
                        for url in image_urls
                    ]
                    image_results = await asyncio.gather(*tasks)
                # Collect the results for this input
                results.append({
                    'cid': cid,
                    'text': post_text,
                    'image_results': image_results,
                    'text_results': text_results,
                    'timing': {
                        'total_time': time.time() - start_time
                    }
                })
        finally:
            # Return models to the pool after the entire batch is processed
            await embedder_pool.put(embedder)
            await model_pool.put(model)
        # Return results list if multiple inputs, else a single result dictionary
        return results if len(results) > 1 else results[0]
    except Exception as e:
        return {'error': str(e)}

def adjust_concurrency(current_concurrency):
    """
    Adjusts the concurrency level based on the current request rate.
    For this example, we'll keep the concurrency level fixed.
    """
    return NUM_WORKERS

# Start the serverless function with the handler and concurrency modifier
runpod.serverless.start(
    {"handler": process_request, "concurrency_modifier": adjust_concurrency}
)
