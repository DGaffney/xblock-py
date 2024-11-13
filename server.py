#!/usr/bin/env python
import os
import asyncio
import time
import torch
from transformers import pipeline
import runpod
from PIL import Image
from io import BytesIO

# Asynchronous HTTP client
import aiohttp

# Set the number of threads to 1
torch.set_num_threads(1)

# Use environment variables
MODEL_NAME_LARGE = os.getenv('MODEL_NAME_LARGE', 'howdyaendra/swin_s3_base_224-xblockm-timm') #WAS: xblock-large-patch3-224
MODEL_NAME = MODEL_NAME_LARGE
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models')

# Check if CUDA (GPU) is available; if not, default to CPU
device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the classifier pipeline, using GPU if available
classifier = pipeline(
    "image-classification",
    model=f"howdyaendra/{MODEL_NAME}",
    device=device,
    torch_dtype=torch_dtype
)


async def process_request(job):
    """
    Asynchronous handler function to process incoming requests.
    """
    try:
        input_data = job.get('input', {})
        image_url = input_data.get('image_url')  # Expecting 'image_url' in the input
        top_k = input_data.get('top_k', 4)       # Default top_k to 4 if not provided

        if not image_url:
            return {'error': 'No image_url provided in the input.'}

        # Measure download time
        download_start = time.time()

        # Download the image asynchronously if it's a URL
        if image_url.startswith('http://') or image_url.startswith('https://'):
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        return {'error': f'Failed to download image. HTTP Status: {response.status}'}
                    content = await response.read()
                    image = Image.open(BytesIO(content)).convert('RGB')
        else:
            # Load the image from a local path
            image = Image.open(image_url).convert('RGB')

        download_end = time.time()
        download_time = download_end - download_start

        # Measure classification time
        classification_start = time.time()
        
        # Perform the classification
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, classifier, image)

        classification_end = time.time()
        classification_time = classification_end - classification_start

        # Return the results along with timing information
        return {
            'results': results,
            'timing': {
                'download_time': download_time,
                'classification_time': classification_time
            }
        }

    except Exception as e:
        # Handle exceptions and return an error message
        return {'error': str(e)}

def adjust_concurrency(current_concurrency):
    """
    Adjusts the concurrency level based on the current request rate.
    For this example, we'll keep the concurrency level fixed.
    """
    return 100

# Start the serverless function with the handler and concurrency modifier
runpod.serverless.start(
    {"handler": process_request, "concurrency_modifier": adjust_concurrency}
)

       # with torch.no_grad():
       #      logits = model(inputs)
       #
       #  # apply sigmoid activation to convert logits to probabilities
       #  # getting labels with confidence threshold of 0.5
       #  predictions = logits.sigmoid() > 0.5
       #
       #  # converting one-hot encoded predictions back to list of labels
       #  predictions = predictions.float().numpy().flatten() # convert boolean predictions to float
