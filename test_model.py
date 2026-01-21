import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from imageio.v3 import imread


# ### Basic Inference Test
def test_basic_inference():
    sess = ort.InferenceSession("./artefacts/models/onnx/model.onnx")

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x})

    assert y[0].shape == (1, 5), "Output shape mismatch"


# ### Inference on Real Cassava Image
# ---- CONFIG ----
MODEL_PATH = "./artefacts/models/model.onnx"
IMAGE_PATH = ".test_img.jpg"  # change to your test image path
IMG_SIZE = 224                     # change if needed
MEAN = [0.485, 0.456, 0.406]       # ImageNet mean
STD = [0.229, 0.224, 0.225]        # ImageNet std
# ----------------


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img).astype(np.float32) / 255.0
    img = (img - MEAN) / STD           # normalize
    img = np.transpose(img, (2, 0, 1)) # HWC → CHW
    img = np.expand_dims(img, axis=0)  # batch dim

    img = img.astype(np.float32)

    return img

def main():
    # Create ONNX Runtime session
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    # Inspect model inputs / outputs
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    # Preprocess input
    x = preprocess_image(IMAGE_PATH)

    # Run inference
    outputs = session.run(
        [output_name],
        {input_name: x}
    )

    logits = outputs[0]

    print("Output shape:", logits.shape)
    print("Raw output:", logits)


# ### Cassava ONNX Classifier
class CassavaONNXClassifier:
    def __init__(self, model_path="./artefacts/models/model.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
 
        self.class_map = {
            0: "cbsd",
            1: "cbb",
            2: "cmd",
            3: "cgm",
            4: "healthy"
        }

    # ----------------------
    # PREPROCESS
    # ----------------------
    def preprocess(self, data):
        data = data[0]
        image_bytes = bytes(data["body"])

        # Read image -> HWC, uint8
        img = imread(image_bytes)

        # Convert to float tensor
        img = torch.from_numpy(img).float() / 255.0  # HWC

        # Convert to CHW
        img = img.permute(2, 0, 1)  # CHW

        # Resize + normalize
        img = self.transform(img)

        # Add batch dimension
        img = img.unsqueeze(0)  # [1, 3, 224, 224]

        # Convert to NumPy and float32 for ONNX
        return img.numpy().astype(np.float32)

    # ----------------------
    # INFERENCE
    # ----------------------
    def infer(self, input_tensor):
        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )
        return outputs[0]

    # ----------------------
    # POSTPROCESS
    # ----------------------
    def postprocess(self, outputs):
        class_idx = outputs.argmax(axis=-1)[0]
        print("Predicted class index:", class_idx)
        return f"Predicted Class: {self.class_map[class_idx]}"

    # ----------------------
    # FULL PIPELINE
    # ----------------------
    def predict(self, data):
        x = self.preprocess(data)
        y = self.infer(x)
        print("Raw model output:", y)
        return self.postprocess(y)


def test_model_using_class():
    classifier = CassavaONNXClassifier(model_path="./artefacts/models/onnx/model.onnx")

    # Example raw input (simulate the same input dict format)
    data_path = "test_img.jpg" #"./data/train/cgm/train-cgm-10.jpg" #"./data/test/0/test-img-19.jpg"
    with open(data_path, "rb") as f:
        data = [{"body": f.read()}]

    predictions = classifier.predict(data)
    print(predictions)

if __name__ == "__main__":
    # main()
    # test_basic_inference()
    test_model_using_class()



