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


async def get_text_labels_batch(items, model, tokenizer):
    """
    Process a batch of text items to get labels and probabilities.

    Args:
        items (list): List of dictionaries containing text metadata and content.
        model: Pre-trained model for text classification.
        tokenizer: Tokenizer for the model.

    Returns:
        list: List of dictionaries with original metadata and label predictions.
    """
    texts = [item["text"] for item in items]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=-1).cpu().numpy()
    id2label = model.config.id2label
    
    results = []
    for item, probs in zip(items, probabilities):
        labels = [id2label[idx] for idx in range(len(probs))]
        label_probs = dict(zip([LABEL_MAP[e] for e in labels], probs))
        results.append({**item, "labels": label_probs})
    return results

async def process_image_batch(items, model, transform, device, config, top_k):
    """
    Process a batch of image items to get top-k predictions.

    Args:
        items (list): List of dictionaries containing image metadata and URLs.
        model: Pre-trained model for inference.
        transform: Image transformation pipeline.
        device: Device to run the model on ('cpu' or 'cuda').
        config: Configuration containing label names.
        top_k (int): Number of top predictions to return.
    """
    async def fetch_image(session, item):
        url = item["url"]
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                return item, content
            return item, None
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, item) for item in items]
        fetched_images = await asyncio.gather(*tasks)
    
    results = []
    for item, content in fetched_images:
        if content is None:
            results.append({**item, 'error': f'Failed to download image.'})
            continue
        try:
            image = Image.open(BytesIO(content)).convert('RGB')
            cuda_image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(cuda_image)
            probabilities = [float(e) for e in logits.sigmoid().cpu().numpy()[0]]
            label_prob_pairs = list(zip(config["label_names"], probabilities))
            label_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            top_k_predictions = label_prob_pairs[:top_k]
            results.append({
                **item,
                "labels": {label: prob for label, prob in top_k_predictions}
            })
        except Exception as e:
            results.append({**item, 'error': str(e)})
    return results

async def process_request(job):
    """
    Asynchronous handler function to process incoming requests.
    Accepts either a single dictionary or a list of dictionaries as input.
    """
    try:
        start_time = time.time()
        input_dict = job.get('input', {})
        job_type = input_dict.get('job_type')
        batch_type = input_dict.get('batch_type')
        input_queue = input_dict.get('params')
        top_k = input_dict.get('top_k', 13)
        if batch_type == 'images':
            for j in input_queue:
                did = j['did']
                id = j['cid']
                j["url"] = f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{id}@jpeg"
        if batch_type == "images":
            try:
                model = await model_pool.get()
                results = await process_image_batch(items, model, transform, device, config, top_k)
            finally:
                await model_pool.put(model)
        else:
            try:
                embedder = await embedder_pool.get()
                results = await process_image_batch(items, model, transform, device, config, top_k)
            finally:
                await embedder_pool.put(embedder)
        print(results)
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
