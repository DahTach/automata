import os
import urllib.request
import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
from groundingdino.models import build_model
from groundingdino.util.inference import Model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from PIL import Image
import utils
import operations.nms as detops
from typing import Tuple, List
import gradio as gr

# TODO: check all devices and make sure they are on the same device


def get_device():
    if torch.cuda.is_available():
        print("using cuda")
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("using mps but mps is not fully supported yet")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    else:
        print("using cpu")
        return "cpu"


class Dino:
    def __init__(self, size="big"):
        self.device = get_device()
        self.distill_cache_dir = os.path.expanduser("~/.cache/autodistill")
        self.cache = os.path.join(self.distill_cache_dir, "groundingdino")
        self.version = "B" if size == "big" else "T"
        self.config = self.get_config()
        self.checkpoint = self.get_checkpoint()
        self.load()

    def get_config(self):
        if self.version == "B":
            return os.path.join(self.cache, "GroundingDINO_SwinB_cfg.py")
        return os.path.join(self.cache, "GroundingDINO_SwinT_OGC.py")

    def get_checkpoint(self):
        if self.version == "B":
            print("using swinB")
            return os.path.join(self.cache, "groundingdino_swinb_cogcoor.pth")
        print("using swinT")
        return os.path.join(self.cache, "groundingdino_swint_ogc.pth")

    def load(self):
        try:
            print("trying to load grounding dino directly")
            self.model = DinoModel(
                model_config_path=self.config,
                model_checkpoint_path=self.checkpoint,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("downloading dino model weights")
            if not os.path.exists(self.cache):
                os.makedirs(self.cache)

            if not os.path.exists(self.checkpoint):
                if self.version == "B":
                    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
                else:
                    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(url, self.checkpoint, utils.show_progress)

            if not os.path.exists(self.config):
                if self.version == "B":
                    url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py"
                else:
                    url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                urllib.request.urlretrieve(url, self.config, utils.show_progress)

            self.model = DinoModel(
                model_config_path=self.config,
                model_checkpoint_path=self.checkpoint,
            )

    def predict(
        self,
        image: np.ndarray,
        prompts: dict[int, list[str]],
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
    ):
        """
        Args:
            - image: np.ndarray
            - prompts: dict[int, list[str]]
            - box_threshold: float
            - text_threshold: float
        """

        h, w, _ = image.shape

        mask_img = "/Users/francescotacinelli/Developer/datasets/pallets/masks/top/top_fill.png"

        mask = torch.Tensor(cv.imread(mask_img, 0))

        predictions = {
            id: (torch.tensor([]).to(self.device), torch.tensor([]).to(self.device))
            for id in prompts.keys()
        }

        for id, aliases in prompts.items():
            for alias in aliases:
                if alias == "":
                    continue
                boxes, scores = self.model.predict(
                    image=image,
                    prompt=alias,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                # Convert boxes to xyxy format
                boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")

                # Denormalize boxes coordinates
                boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32).to(
                    boxes.device
                )

                # Apply NMS
                valid_boxes, valid_scores = detops.nmsT(
                    detections=(boxes, scores), shape=(h, w), mask=mask
                )

                predictions[id] = (
                    torch.cat([predictions[id][0], valid_boxes]),
                    torch.cat([predictions[id][1], valid_scores]),
                )

        return predictions

    def predict_and_clean(
        self,
        image: np.ndarray,
        prompts: dict[int, list[str]],
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
    ):
        """
        Args:
            - image: np.ndarray
            - prompts: dict[int, list[str]]
            - box_threshold: float
            - text_threshold: float
        """

        h, w, _ = image.shape

        predictions = {
            id: (torch.tensor([]).to(self.device), torch.tensor([]).to(self.device))
            for id in prompts.keys()
        }

        mask_img = "/Users/francescotacinelli/Developer/datasets/pallets/masks/top/top_fill.png"

        mask = torch.Tensor(cv.imread(mask_img, 0))

        for id, aliases in prompts.items():
            for alias in aliases:
                if alias == "":
                    continue
                boxes, scores = self.model.predict(
                    image=image,
                    prompt=alias,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                # Convert boxes to xyxy format
                boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")

                # Denormalize boxes coordinates
                boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32).to(
                    boxes.device
                )

                # Apply NMS
                valid_boxes, valid_scores = detops.nmsT(
                    detections=(boxes, scores), shape=(h, w), mask=mask
                )

                predictions[id] = (
                    torch.cat([predictions[id][0], valid_boxes]),
                    torch.cat([predictions[id][1], valid_scores]),
                )

        # Apply oversuppression
        # predictions = detops.oversuppression(predictions, (h, w))
        predictions = detops.class_agnostic_nms(predictions)

        return predictions

    def infer(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float = 0.1,
        text_threshold: float = 0.1,
    ) -> List[torch.Tensor]:
        h, w, _ = image.shape

        predictions = [
            torch.tensor([]).to(self.device),
            torch.tensor([]).to(self.device),
        ]

        boxes, scores = self.model.predict(
            image=image,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert boxes to xyxy format
        boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")

        # Denormalize boxes coordinates
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.device)

        # Apply NMS
        valid_boxes, valid_scores = detops.nmsT(
            detections=(boxes, scores), shape=(h, w)
        )

        predictions[0] = torch.cat([predictions[0], valid_boxes])
        predictions[1] = torch.cat([predictions[1], valid_scores])

        return predictions

    def interface(self):
        from prompts import Dataset
        from gradiops.ops import to_annotations

        ds = Dataset()

        def predict(image: np.ndarray, *captions: str):
            class_captions = {i: [caption] for i, caption in enumerate(captions)}

            predictions = self.predict(image, class_captions)
            cleaned_predictions = self.predict_and_clean(image, class_captions)

            dirty_annotated_image = (
                image,
                to_annotations(predictions, list(ds.classes.keys())),
            )
            clean_annotated_image = (
                image,
                to_annotations(cleaned_predictions, list(ds.classes.keys())),
            )

            return dirty_annotated_image, clean_annotated_image

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    start = gr.Button("Predict")

                with gr.Column():
                    captions = []
                    for name, idx in list(ds.classes.items()):
                        t = gr.Textbox(
                            placeholder=name,
                            value=ds.captions.get(idx, ["no caption"])[0],
                        )
                        captions.append(t)

            with gr.Row():
                dirty_output = gr.AnnotatedImage()
            with gr.Row():
                clean_output = gr.AnnotatedImage()

            start.click(
                predict, inputs=[image, *captions], outputs=[dirty_output, clean_output]
            )

        self.demo = demo


class DinoModel(Model):
    def __init__(self, model_config_path: str, model_checkpoint_path: str):
        self.device = get_device()
        self.model = self.load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
        ).to(self.device)

    def load_model(self, model_config_path: str, model_checkpoint_path: str):
        args = SLConfig.fromfile(model_config_path)
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple:
        processed_image = self.preprocess_image(image_bgr=image).to(self.device)
        caption = self.preprocess_caption(caption=prompt)
        boxes, scores = self.infer(
            model=self.model,
            image=processed_image,
            captions=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        return boxes, scores

    def preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def infer(
        self,
        model,
        image: torch.Tensor,
        captions: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple:
        model = model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = model(image[None], captions=[captions])

        pred_logits = outputs["pred_logits"].sigmoid()[0]
        scores = pred_logits.max(dim=1)[0]
        mask = scores > box_threshold
        boxes = outputs["pred_boxes"][0][mask]

        return boxes, scores[mask]


def demo():
    dino = Dino(size="small")
    dino.interface()
    dino.demo.launch()


if __name__ == "__main__":
    demo()
