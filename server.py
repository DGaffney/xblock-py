import os
import asyncio
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
MODEL_NAME_LARGE = os.getenv('MODEL_NAME_LARGE', 'xblock-large-patch3-224')
MODEL_NAME = MODEL_NAME_LARGE
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models')

# Initialize the image classification pipeline
classifier = pipeline(
    "image-classification",
    model=f"howdyaendra/{MODEL_NAME}",
    cache_dir=MODEL_PATH,
    local_files_only=True  # Ensure it uses the local model
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

        # Perform the classification
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, classifier, image, top_k)

        # Return the results
        return {'results': results}

    except Exception as e:
        # Handle exceptions and return an error message
        return {'error': str(e)}

def adjust_concurrency(current_concurrency):
    """
    Adjusts the concurrency level based on the current request rate.
    For this example, we'll keep the concurrency level fixed.
    """
    max_concurrency = 5
    min_concurrency = 1

    # For now, return a fixed concurrency level
    return max(min_concurrency, min(max_concurrency, current_concurrency))

# Start the serverless function with the handler and concurrency modifier
runpod.serverless.start(
    {"handler": process_request, "concurrency_modifier": adjust_concurrency}
)

# if __name__ == "__main__":
#     # Simulate a job
#     job = {
#         "input": {
#             "image_url": "https://example.com/path/to/your/image.jpg",
#             "top_k": 4
#         }
#     }
#
#     # Run the async function
#     import asyncio
#     result = asyncio.run(process_request(job))
#     print(result)