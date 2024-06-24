import os
import urllib.request
from typing import List, Tuple
import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
from groundingdino.models import build_model
from groundingdino.util.inference import Model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from PIL import Image
import utils


class Dino:
    def __init__(self, size="big"):
        self.device = self.get_device()
        self.distill_cache_dir = os.path.expanduser("~/.cache/autodistill")
        self.cache = os.path.join(self.distill_cache_dir, "groundingdino")
        self.config = os.path.join(self.cache, "GroundingDINO_SwinT_OGC.py")
        self.checkpoint = os.path.join(self.cache, "groundingdino_swint_ogc.pth")
        self.version = "B" if size == "big" else "T"
        self.load()

    def get_config(self):
        if self.version == "B":
            return "GroundingDINO_SwinB_cfg.py"
        return "GroundingDINO_SwinT_CogCoor.py"

    def get_checkpoint(self):
        if self.version == "B":
            return "groundingdino_swinb_cogcoor.pth"
        return "groundingdino_swint_ogc.pth"

    def get_device(self):
        if torch.cuda.is_available():
            print("using cuda")
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("using cpu because mps is not fully supported yet")
            # TODO: replace with cpu when device fixed
            return "mps"
        else:
            print("using cpu")
            return "cpu"

    def load(self):
        try:
            print("trying to load grounding dino directly")
            self.model = DinoModel(
                model_config_path=self.config,
                model_checkpoint_path=self.checkpoint,
            )
        except Exception as e:
            print(f"Occured error: {e}")
            print("downloading dino model weights")
            if not os.path.exists(self.cache):
                os.makedirs(self.cache)

            if not os.path.exists(self.checkpoint):
                if self.version == "B":
                    url = "https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
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

    def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        confidence: float = 0.5,
    ):
        # initialize empty lists for boxes, scores and class_ids on self.device
        detections = [torch.tensor([]).to(self.device) for _ in range(3)]
        boxes, scores = self.model.predict(
            image=image,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections[0] = torch.cat((detections[0], boxes), dim=0)
        # TODO: check if confidences is a tensor and what is inside
        detections[1] = torch.cat((detections[1], scores), dim=0)

        # extend class_ids with the same class for each box
        detections[2] = torch.tensor([prompt] * len(boxes)).to(self.device)
        # detections[2].extend([prompt] * len(detections[0]))

        filtered_detections = self.nms(detections)

        return boxes

    def nms(
        self,
        detections,
        iou_threshold=0.5,
        containment_threshold=0.8,
        size_deviation_threshold=1.5,
    ):
        all_boxes = detections[0]
        all_scores = detections[1]
        all_prompts = detections[2]

        # Perform NMS
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, iou_threshold)

        # TODO: roi align the boxes with the pallet
        # torchvision.ops.roi_align

        # Remove boxes that are bigger than average by size deviation threshold
        areas = torchvision.ops.box_area(all_boxes)
        avg_area = torch.mean(areas)
        for i, area in enumerate(areas):
            if area > avg_area * size_deviation_threshold:
                # Remove this index from keep_indices
                keep_indices = keep_indices[keep_indices != i]

        # Remove boxes with high containment in others
        remove_indices = []
        for i in keep_indices:
            box_i = all_boxes[i]
            for j in keep_indices:
                if i == j:  # Skip self-comparison
                    continue
                box_j = all_boxes[j]

                # Calculate intersection area
                inter_width = torch.max(
                    torch.tensor(0),
                    torch.min(box_i[2], box_j[2]) - torch.max(box_i[0], box_j[0]),
                )
                inter_height = torch.max(
                    torch.tensor(0),
                    torch.min(box_i[3], box_j[3]) - torch.max(box_i[1], box_j[1]),
                )
                inter_area = inter_width * inter_height
                box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

                # Check for high containment
                containment_ratio = inter_area / box_i_area
                if containment_ratio > containment_threshold:
                    remove_indices.append(i)
                    break

        # Update keep_indices, removing those with high containment
        keep_indices = [idx for idx in keep_indices if idx not in remove_indices]

        # Reconstruct the final list of detections with original class IDs
        filtered_detections = [[], [], []]
        for index in keep_indices:
            filtered_detections[0].append(all_boxes[index].tolist())
            filtered_detections[1].append(all_scores[index].item())
            filtered_detections[2].append(all_prompts[index])

        return filtered_detections


class DinoModel(Model):
    def __init__(self, model_config_path: str, model_checkpoint_path: str):
        self.device = utils.get_device()
        self.model = self.load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
        ).to(self.device)

    def load_model(self, model_config_path: str, model_checkpoint_path: str):
        args = SLConfig.fromfile(model_config_path)
        # args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
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
            caption=caption,
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
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple:
        model = model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        pred_logits = outputs["pred_logits"].sigmoid()[0]
        scores = pred_logits.max(dim=1)[0]
        mask = scores > box_threshold
        boxes = outputs["pred_boxes"][0][mask]

        # logits = pred_logits[mask]  # logits.shape = (n, 256)

        # tokenizer = model.tokenizer
        # tokenized = tokenizer(caption)

        #
        # phrases = [
        #     get_phrases_from_posmap(
        #         logit > text_threshold, tokenized, tokenizer
        #     ).replace(".", "")
        #     for logit in logits
        # ]

        return boxes, scores[mask]
