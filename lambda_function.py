import os
import io
import json
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

# ----------------------
# CONFIG
# ----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "model.onnx")

CLASS_MAP = {
    0: "cbsd",
    1: "cbb",
    2: "cmd",
    3: "cgm",
    4: "healthy"
}

IMAGE_SIZE = (224, 224)

# ----------------------
# LOAD MODEL (COLD START)
# ----------------------
session = ort.InferenceSession(
    MODEL_NAME,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ----------------------
# IMAGE DOWNLOAD
# ----------------------
def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LambdaImageFetcher/1.0)"
    }
    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response:
        return response.read()

# ----------------------
# PREPROCESS
# ----------------------
def preprocess(url):
    image_data = download_image(url)

    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img = np.array(img).astype(np.float32) / 255.0  # HWC
    img = img.transpose(2, 0, 1)                    # CHW
    img = np.expand_dims(img, axis=0)               # NCHW

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    img = (img - mean) / std

    return img.astype(np.float32)

# ----------------------
# INFERENCE
# ----------------------
def infer(x):
    outputs = session.run(
        [output_name],
        {input_name: x}
    )
    return outputs[0]

# ----------------------
# POSTPROCESS
# ----------------------
def postprocess(outputs):
    class_idx = outputs.argmax(axis=-1)[0]
    # print("Predicted class index:", class_idx)
    return f"Predicted Class: {CLASS_MAP[class_idx]}"

# ----------------------
# FULL PIPELINE
# ----------------------
def predict(url):
    x = preprocess(url)
    y = infer(x)
    print("Raw model output:", y)
    return postprocess(y)

# ----------------------
# AWS LAMBDA HANDLER
# ----------------------
def lambda_handler(event, context):
    """
    Expected input:
    {
        "url": "https://example.com/image.jpg"
    }
    """
    try:
        # Parse the body if it's a Function URL request
        if "body" in event:
            body = json.loads(event["body"])
        else:
            # Direct invocation from console
            body = event
        
        url = body["url"]
        result = predict(url)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(result)
        }
    except KeyError as e:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": f"Missing required field: {str(e)}"})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": str(e)})
        }