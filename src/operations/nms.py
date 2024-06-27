import os
import urllib.request
from typing import List, Tuple
import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
import torchvision.ops as ops
from groundingdino.models import build_model
from groundingdino.util.inference import Model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from PIL import Image
from transformers.models.auto.configuration_auto import re
import utils
from tqdm import tqdm
import copy


from collections import defaultdict
import itertools


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
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


def nmsT(
    detections: tuple[torch.Tensor, torch.Tensor],
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform Non-Maximum Suppression on the detections
    Args:
        detections: tuple of detections (boxes, scores, class_ids)
        iou_threshold: IoU threshold for NMS
    Returns:
        list of detections after NMS
    """
    boxes, scores = detections

    if boxes.device != scores.device:
        prefered_device = (
            boxes.device if boxes.device != torch.device("cpu") else scores.device
        )
        boxes = boxes.to(prefered_device)

    # Perform NMS
    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold).long()

    # Filter the detections with the keep_indices
    valid_boxes = boxes[keep_indices]
    valid_scores = scores[keep_indices]

    return valid_boxes, valid_scores


# def overSuppression(
#     predictions: dict[int, Tuple[torch.Tensor, torch.Tensor]], threshold: float = 0.5
# ):
#     """Remove overlapping boxes from predictions
#     Args:
#         predictions: dict[int, Tuple[torch.Tensor[list[tuple[float, float, float, float]]], torch.Tensor[list[float]]
#     Returns:
#         dict of predictions after removing overlapping boxes
#     """
#
#     keys = predictions.keys()
#
#     combinations = list(itertools.combinations(keys, 2))
#
#     for key1, key2 in combinations:
#         boxes1, scores1 = predictions[key1]
#         boxes2, scores2 = predictions[key2]
#         # Perform IOU
#         ious = ops.box_iou(boxes1, boxes2)
#         # Get the indices where iou is greater than threshold
#         over = torch.where(ious > threshold)
#         # Keep only the boxes with higher scores
#         scores1 = normalize(scores1)
#         scores2 = normalize(scores2)
#         clean1, clean2 = torch.where(scores1[over[0]] < scores2[over[1]])
#         bad1 = over[0][clean1]
#         bad2 = over[1][~clean2]
#
#         if bad1 in predictions[key1]:
#             boxes1.remove(bad1)
#             scores1.remove(boxes1.index(bad1))
#         if bad2 in predictions[key2]:
#             boxes2.remove(bad2)
#             scores2.remove(boxes2.index(bad2))
#
#     return predictions


def oversuppression(
    predictions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    threshold: float = 0.5,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Removes overlapping boxes with lower normalized scores.

    Args:
        predictions: Dictionary mapping class IDs to tuples of (boxes, scores) tensors.
        threshold: IoU threshold for determining overlap.

    Returns:
        The updated predictions dictionary.
    """

    predictions = copy.deepcopy(predictions)
    keys = list(predictions.keys())

    for key1, key2 in itertools.combinations(keys, 2):
        boxes1, scores1 = predictions[key1]
        boxes2, scores2 = predictions[key2]

        # Calculate IoU
        ious = ops.box_iou(boxes1, boxes2)

        # Find overlapping pairs (over threshold)
        keep_mask = ious <= threshold

        # Normalize scores and find lower scores in each overlapping pair
        scores1_norm = scores1 / scores1.max()
        scores2_norm = scores2 / scores2.max()
        lower_score_mask = scores1_norm.unsqueeze(1) < scores2_norm.unsqueeze(0)

        # Combine the masks to find the indices of boxes to keep
        combined_mask = keep_mask | lower_score_mask

        # Keep only the valid boxes and scores for each class
        valid_indices1 = combined_mask.any(dim=1)
        valid_indices2 = combined_mask.any(dim=0)

        predictions[key1] = (boxes1[valid_indices1], scores1[valid_indices1])
        predictions[key2] = (boxes2[valid_indices2], scores2[valid_indices2])

        # FIX: Why are input predictions being modified in place? or is it a copy?

    return predictions


def remove_all_overlaps(self, predictions, iou_threshold=0.3):
    all_boxes = []
    all_scores = []
    all_prompts = []

    for key, value in predictions.items():
        all_boxes.extend(value[0])
        all_scores.extend(value[1])
        all_prompts.extend(value[2])

    all_boxes = torch.tensor(all_boxes).to(self.device)
    all_scores = torch.tensor(all_scores).to(self.device)

    # Perform NMS
    keep_indices = (
        torchvision.ops.nms(all_boxes, all_scores, iou_threshold)
        .long()
        .to(self.device)
        .tolist()
    )

    valid_boxes = []
    valid_scores = []
    valid_prompts = []
    for i in keep_indices:
        valid_boxes.append(all_boxes[i])
        valid_scores.append(all_scores[i])
        valid_prompts.append(all_prompts[i])

    return valid_boxes, valid_scores, valid_prompts


def remove_overlapping(
    self,
    predictions,
    iou_threshold=0.3,
    containment_threshold=0.5,
    size_deviation_threshold=1.1,
):
    all_boxes = []
    all_scores = []
    all_prompts = []

    for key, value in predictions.items():
        all_boxes.extend(value[0])
        all_scores.extend(value[1])
        all_prompts.extend(value[2])

    all_boxes = torch.stack(all_boxes).to(self.device)
    all_scores = torch.stack(all_scores).to(self.device)

    # Perform NMS
    keep_indices = (
        torchvision.ops.nms(all_boxes, all_scores, iou_threshold).long().to(self.device)
    )

    valid_boxes = []
    valid_scores = []
    valid_prompts = []
    for idx in keep_indices:
        valid_boxes.append(all_boxes[idx])
        valid_scores.append(all_scores[idx])
        valid_prompts.append(all_prompts[idx])

    return valid_boxes, valid_scores, valid_prompts


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
        key_keep_indices = torch.where((keep_indices >= start) & (keep_indices <= end))[
            0
        ]

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
        torchvision.ops.nms(all_boxes, all_scores, iou_threshold).long().to(self.device)
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
        torchvision.ops.nms(all_boxes, all_scores, iou_threshold).long().to(self.device)
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
        torchvision.ops.nms(all_boxes, all_scores, iou_threshold).long().to(self.device)
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
