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


def oversuppression(
    input_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    shape: Tuple[int, int],
    threshold: float = 0.5,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Removes overlapping boxes with lower normalized scores.

    Args:
        predictions: Dictionary mapping class IDs to tuples of (boxes, scores) tensors.
        img_shape: Shape of the image.
        threshold: IoU threshold for determining overlap.

    Returns:
        The updated predictions dictionary.
    """

    # TODO: check why some object disappear (like the rollbars)

    predictions = copy.deepcopy(input_predictions)
    keys = list(predictions.keys())

    for key1, key2 in itertools.combinations(keys, 2):
        boxes1, scores1 = predictions[key1]
        boxes2, scores2 = predictions[key2]

        # Normalize the scores
        scores1 = scores1 / scores1.max()
        scores2 = scores2 / scores2.max()

        # Remove contained boxes before NMS
        keep1 = remove_contained(boxes1)
        keep2 = remove_contained(boxes2)
        boxes1 = boxes1[keep1]
        scores1 = scores1[keep1]
        boxes2 = boxes2[keep2]
        scores2 = scores2[keep2]

        # Filter the original boxes and scores using valid_indices
        # (This will include boxes from both classes that survived NMS)
        all_boxes = torch.cat((boxes1, boxes2), dim=0)
        all_scores = torch.cat((scores1, scores2), dim=0)

        valid_indices = torchvision.ops.nms(all_boxes, all_scores, threshold)
        # Tensor[N]: the indices of the elements that have been kept by NMS

        # Separate the boxes and scores
        valid_indices1 = valid_indices < len(boxes1)
        valid_indices2 = valid_indices >= len(boxes1)

        # Filter the boxes and scores using the valid_indices
        filtered_boxes1 = all_boxes[valid_indices][valid_indices1]
        filtered_scores1 = all_scores[valid_indices][valid_indices1]
        filtered_boxes2 = all_boxes[valid_indices][valid_indices2]
        filtered_scores2 = all_scores[valid_indices][valid_indices2]

        # Perform outlier removal on the filtered boxes
        liers1 = remove_outliers(filtered_boxes1, shape)
        liers2 = remove_outliers(filtered_boxes2, shape)

        # Filter again to get the final boxes and scores after outlier removal
        predictions[key1] = (filtered_boxes1[liers1], filtered_scores1[liers1])
        predictions[key2] = (filtered_boxes2[liers2], filtered_scores2[liers2])

    return predictions


def remove_outliers(boxes: torch.Tensor, shape: Tuple[int, int], threshold=0.6):
    """Remove boxes that are outliers in terms of size
    Args:
        boxes: list of boxes
        shape: shape of the image (height, width)
        threshold: threshold for removing outliers
    Returns:
        list of boxes after removing outliers
    """

    # remove boxes with width or height more than 0.4 of the image size
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    box_width = box_width / shape[1]
    box_height = box_height / shape[0]

    bigger_indices = (box_width > threshold) | (box_height > threshold)
    return ~bigger_indices


def remove_contained(boxes: torch.Tensor, threshold=0.1):
    """Remove boxes that are contained in other boxes over a certain threshold
    Args:
        boxes: list of boxes
        threshold: threshold for removing inside boxes
    Returns:
        list of boxes after removing inside boxes
    """
    keep = torch.ones(len(boxes), dtype=torch.bool)
    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i != j:
                if (
                    box1[0] >= box2[0] * threshold
                    and box1[1] >= box2[1] * threshold
                    and box1[2] <= box2[2] * threshold
                    and box1[3] <= box2[3] * threshold
                ):
                    keep[i] = False
                    break
    return keep
