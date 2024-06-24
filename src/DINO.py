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
from transformers.models.auto.configuration_auto import re
import utils
from tqdm import tqdm


class Dino:
    def __init__(self, size="big"):
        self.device = self.get_device()
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

    def get_device(self):
        if torch.cuda.is_available():
            print("using cuda")
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
        detections = [
            torch.tensor([]).to(self.device),
            torch.tensor([]).to(self.device),
            [],
        ]
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
        # detections[2] = torch.tensor([prompt] * len(boxes)).to(self.device)
        detections[2].extend([prompt] * len(detections[0]))

        filtered_detections = self.nms_tensor(detections)

        return filtered_detections

    def predict_batch(
        self,
        image: np.ndarray,
        prompts: dict,
        box_threshold: float = 0.2,
        text_threshold: float = 0.25,
    ):
        """
        Predicts bounding boxes for a batch of prompts.
        """
        predictions = {}

        for key, value in tqdm(prompts.items()):
            detections = [
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                [],
            ]

            prompt = " . ".join(value)

            boxes, scores = self.model.predict(
                image=image,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            detections[0] = torch.cat((detections[0], boxes), dim=0)
            detections[1] = torch.cat((detections[1], scores), dim=0)

            # extend class_ids with the same class for each box
            # detections[2] = torch.tensor([prompt] * len(boxes)).to(self.device)
            detections[2].extend([prompt] * len(detections[0]))

            filtered_detections = self.nms(detections)

            predictions[key] = filtered_detections


        # return predictions
        filtered_predictions = self.filter_classes(predictions)

        return filtered_predictions

    def predict_slow(
        self,
        image: np.ndarray,
        prompts: dict,
        box_threshold: float = 0.15,
        text_threshold: float = 0.25,
    ):
        """
        Predicts bounding boxes for a batch of prompts.
        """
        predictions = {}

        for key, value in tqdm(prompts.items()):
            detections = [
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                [],
            ]
            for prompt in value:

                boxes, scores = self.model.predict(
                    image=image,
                    prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                detections[0] = torch.cat((detections[0], boxes), dim=0)
                detections[1] = torch.cat((detections[1], scores), dim=0)

                # extend class_ids with the same class for each box
                # detections[2] = torch.tensor([prompt] * len(boxes)).to(self.device)
                detections[2].extend([prompt] * len(detections[0]))

            filtered_detections = self.diomerda(detections)

            predictions[key] = filtered_detections

        filtered_predictions = self.filter_classes(predictions)
        # FIX: fix filtered classes so that no boxes with same class are overlapping
        # TODO: lower box threshold but implement more aggressive nms to avoid cuttered boxes

        return filtered_predictions

        # return predictions

    def filter_classes(self, predictions, iou_threshold=0.3):
        # FIX: this performs as shit
        all_boxes = []
        all_scores = []
        all_labels = []
        keys_idxs = {}
        for key, value in predictions.items():
            keys_idxs.setdefault(key, {"start": 0, "end": 0})
            boxes = value[0]
            keys_idxs[key]["start"] = len(all_boxes)
            all_boxes.extend(boxes)
            keys_idxs[key]["end"] = len(all_boxes)
            scores = value[1]
            all_scores.extend(scores)
            labels = value[2]
            all_labels.extend(labels)

        # class dependant nms, keep the highest confidence score if different classes overlap
        keep_indices = torchvision.ops.nms(
            torch.stack(all_boxes), torch.stack(all_scores), iou_threshold
        )

        filtered_predictions = {}
        for key, value in keys_idxs.items():
            start = value["start"]
            end = value["end"]
            # Use torch.logical_and to combine conditions
            key_keep_indices = torch.where((keep_indices >= start) & (keep_indices <= end))[0]

            key_boxes = []
            key_scores = []
            key_labels = []
            for idx in key_keep_indices:
                key_boxes.append(all_boxes[idx])
                key_scores.append(all_scores[idx])
                key_labels.append(all_labels[idx])

            filtered_predictions[key] = key_boxes, key_scores, key_labels

        return filtered_predictions


    def diomerda(
        self,
        detections,
        iou_threshold=0.3,
        containment_threshold=0.5,
        size_deviation_threshold=1.1,
    ):
        all_boxes = torchvision.ops.box_convert(
            detections[0], in_fmt="cxcywh", out_fmt="xyxy"
        )

        all_scores = detections[1]
        all_prompts = detections[2]

        # Perform NMS
        keep_indices = (
            torchvision.ops.nms(all_boxes, all_scores, iou_threshold)
            .long()
            .to(self.device)
        ).tolist()

        # Remove boxes that are bigger than average by size deviation threshold
        areas = torchvision.ops.box_area(all_boxes)
        avg_area = torch.mean(areas)
        for i, area in enumerate(areas):
            if area > avg_area * size_deviation_threshold:
                # Remove this index from keep_indices
                if i in keep_indices:
                    keep_indices.remove(i)

        # Remove boxes with high containment in others
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
                    # remove_indices.append(i)
                    if i in keep_indices:
                        keep_indices.remove(i)
                    break

        valid_boxes = []
        valid_scores = []
        valid_prompts = []

        for idx in keep_indices:
            valid_boxes.append(all_boxes[idx])
            valid_scores.append(all_scores[idx])
            valid_prompts.append(all_prompts[idx])

        return valid_boxes, valid_scores, valid_prompts




    def nms(
        self,
        detections,
        iou_threshold=0.3,
        containment_threshold=0.6,
        size_deviation_threshold=1.1,
    ):
        all_boxes = torchvision.ops.box_convert(
            detections[0], in_fmt="cxcywh", out_fmt="xyxy"
        )
        # all_boxes = detections[0]

        all_scores = detections[1]
        all_prompts = detections[2]

        # Perform NMS
        keep_indices = (
            torchvision.ops.nms(all_boxes, all_scores, iou_threshold)
            .long()
            .to(self.device)
        )

        remove_indices = torch.tensor([], dtype=torch.long).to(self.device)

        # Remove boxes that are bigger than average by size deviation threshold
        areas = torchvision.ops.box_area(all_boxes)
        avg_area = torch.mean(areas)
        for i, area in enumerate(areas):
            if area > avg_area * size_deviation_threshold:
                # Remove this index from keep_indices
                remove_indices = torch.cat(
                    (remove_indices, torch.tensor([i]).to(self.device)), 0
                )

        # Remove boxes with high containment in others
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
                    # remove_indices.append(i)
                    remove_indices = torch.cat(
                        (remove_indices, torch.tensor([i]).to(self.device)), 0
                    )
                    break

        remove_mask = torch.zeros(
            all_boxes.size(0), dtype=torch.bool, device=all_boxes.device
        )
        remove_mask.scatter_(0, remove_indices, True)

        # Invert the mask to get the elements to keep
        keep_mask = ~remove_mask

        # Filter 'all_boxes' using the mask
        valid_boxes = all_boxes[keep_mask]
        valid_scores = all_scores[keep_mask]
        valid_prompts = []
        for idx in keep_mask:
            valid_prompts.append(all_prompts[idx])

        return valid_boxes.tolist(), valid_scores.tolist(), valid_prompts

    def nms_tensor(
        self,
        detections,
        iou_threshold=0.5,
        containment_threshold=0.8,
        size_deviation_threshold=1.5,
    ):
        all_boxes = torchvision.ops.box_convert(
            detections[0], in_fmt="cxcywh", out_fmt="xyxy"
        )
        # all_boxes = detections[0]

        all_scores = detections[1]
        all_prompts = detections[2]

        # Perform NMS
        keep_indices = (
            torchvision.ops.nms(all_boxes, all_scores, iou_threshold)
            .long()
            .to(self.device)
        )
        remove_indices = torch.tensor([], dtype=torch.long).to(self.device)

        # Remove boxes that are bigger than average by size deviation threshold
        areas = torchvision.ops.box_area(all_boxes)
        avg_area = torch.mean(areas)
        for i, area in enumerate(areas):
            if area > avg_area * size_deviation_threshold:
                # Remove this index from keep_indices
                # remove_indices.append(i)
                remove_indices = torch.cat(
                    (remove_indices, torch.tensor([i]).to(self.device)), 0
                )

        # Remove boxes with high containment in others
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
                    # remove_indices.append(i)
                    remove_indices = torch.cat(
                        (remove_indices, torch.tensor([i]).to(self.device)), 0
                    )
                    break

        remove_mask = torch.zeros(
            all_boxes.size(0), dtype=torch.bool, device=all_boxes.device
        )
        remove_mask.scatter_(0, remove_indices, True)

        # Invert the mask to get the elements to keep
        keep_mask = ~remove_mask

        # Filter 'all_boxes' using the mask
        valid_boxes = all_boxes[keep_mask]

        return valid_boxes


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
