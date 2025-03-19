import os
import base64
import time
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

def download_and_decode_image(image_url_or_b64):
    """Download an image from a URL or decode from base64."""
    if image_url_or_b64.startswith("http"):
        # Download from URL using RunPod utility
        input_file = rp_download(image_url_or_b64)
        image = Image.open(input_file).convert("RGB")
        rp_cleanup(input_file)  # Clean up downloaded file
    else:
        # Decode from base64
        image_data = base64.b64decode(image_url_or_b64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    return image
def resize_and_center(image, target_width, target_height):
    """Resize and center an image while maintaining aspect ratio."""
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > target_width / target_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a blank image with the target size
    new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# Download checkpoints if they don't exist
if not os.path.exists("./ckpts"):
    snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")


class LeffaPredictor(object):
    def __init__(self):
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

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict(
        self,
        src_image,
        ref_image,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False
    ):
        assert control_type in [
            "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)

        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask
        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(
                    model_parse, keypoints, vt_garment_type)
            elif vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(
                    model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        # DensePose
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(
                    src_image_array)[:, :, ::-1]
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(
                    src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate(
                    [src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(
                src_image_array)[:, :, ::-1]
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }

        data = transform(data)

        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            elif vt_model_type == "dress_code":
                inference = self.vt_inference_dc
        elif control_type == "pose_transfer":
            inference = self.pt_inference

        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,)

        gen_image = output["generated_image"][0]

        # Convert to base64 for output
        buffered = BytesIO()
        gen_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "generated_image": img_str
        }


# Initialize the model
predictor = LeffaPredictor()


def handler(event):
    """RunPod handler function to process inference requests."""
    try:
        validated_input = validate(event["input"], INPUT_SCHEMA)

        src_image = download_and_decode_image(validated_input["src_image"])
        ref_image = download_and_decode_image(validated_input["ref_image"])

        control_type = validated_input["control_type"]
        ref_acceleration = validated_input.get("ref_acceleration", False)
        step = validated_input.get("step", 30)
        scale = validated_input.get("scale", 2.5)
        seed = validated_input.get("seed", 42)
        vt_model_type = validated_input.get("vt_model_type", "viton_hd")
        vt_garment_type = validated_input.get("vt_garment_type", "upper_body")
        vt_repaint = validated_input.get("vt_repaint", False)

        result = predictor.leffa_predict(
            src_image=src_image,
            ref_image=ref_image,
            control_type=control_type,
            ref_acceleration=ref_acceleration,
            step=step,
            scale=scale,
            seed=seed,
            vt_model_type=vt_model_type,
            vt_garment_type=vt_garment_type,
            vt_repaint=vt_repaint
        )

        return {
            "output": {
                "generated_image": result["generated_image"]
            }
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
