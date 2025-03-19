import runpod
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import io
import base64

# Download model checkpoints
print("Downloading model checkpoints...")
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")
print("Model checkpoints downloaded successfully.")

# Initialize Leffa Predictor
class LeffaPredictor(object):
    def __init__(self):
        print("Initializing Leffa models...")
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )
        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict(self, src_image, ref_image, control_type, vt_model_type="viton_hd", vt_garment_type="upper_body"):
        assert control_type in ["virtual_tryon", "pose_transfer"], f"Invalid control type: {control_type}"
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask prediction
        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
        else:
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        # DensePose prediction
        if control_type == "virtual_tryon":
            src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_seg_array)
        else:
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_iuv_array)

        # Transform and inference
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        inference = self.vt_inference_hd if control_type == "virtual_tryon" else self.pt_inference
        output = inference(data, num_inference_steps=50, guidance_scale=2.5, seed=42)
        gen_image = output["generated_image"][0]
        return gen_image

# Instantiate the model
leffa_predictor = LeffaPredictor()
print("Leffa Predictor initialized successfully.")

# -------------------------
# RunPod Handler Definition
# -------------------------
def handler(event):
    print("Worker Started")

    input_data = event.get("input", {})
    src_image_data = input_data.get("src_image", "")
    ref_image_data = input_data.get("ref_image", "")
    control_type = input_data.get("control_type", "virtual_tryon")
    vt_model_type = input_data.get("vt_model_type", "viton_hd")
    vt_garment_type = input_data.get("vt_garment_type", "upper_body")

    # Decode base64 images
    try:
        src_image = Image.open(io.BytesIO(base64.b64decode(src_image_data))).convert("RGB")
        ref_image = Image.open(io.BytesIO(base64.b64decode(ref_image_data))).convert("RGB")
    except Exception as e:
        print(f"Error decoding images: {e}")
        return {"error": "Invalid image data."}

    # Generate prediction
    try:
        generated_image = leffa_predictor.leffa_predict(
            src_image, ref_image, control_type, vt_model_type, vt_garment_type
        )
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"generated_image": encoded_image}
    except Exception as e:
        print(f"Error generating image: {e}")
        return {"error": "Failed to generate the image."}

# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
