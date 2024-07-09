import copy
import itertools
from typing import Tuple

import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
from PIL import Image


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
    shape: Tuple[int, int],
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

    # # TODO: make another method for batch NMS, this overrides the batched overhead
    # # Perform outlier removal on the filtered boxes
    # keep_indices = remove_outliers(valid_boxes, shape)
    #
    # valid_boxes = valid_boxes[keep_indices]
    # valid_scores = valid_scores[keep_indices]
    #
    # # Perform containment removal on the filtered boxes
    # keep_indices = remove_contained(valid_boxes)
    #
    # valid_boxes = valid_boxes[keep_indices]
    # valid_scores = valid_scores[keep_indices]

    return valid_boxes, valid_scores


def roi(
    detections: tuple[torch.Tensor, torch.Tensor],
    h_lines: Tuple[int, int] | None = None,
    v_lines: Tuple[int, int] | None = None,
    rect: Tuple[int, int, int, int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform Region of Interest (ROI) filtering on the detections
    Args:
        detections: tuple of detections (boxes, scores)
        lines: tuple of lines (point1, point2) defining the ROI
        rect: tuple of rectangle (x1, y1, x2, y2) defining the ROI
    Returns:
        list of detections within the ROI
    """
    boxes, scores = detections

    device = boxes.device

    if boxes.device != scores.device:
        prefered_device = (
            boxes.device if boxes.device != torch.device("cpu") else scores.device
        )
        boxes = boxes.to(prefered_device)

    # Perform ROI filtering
    keep_indices = torch.ones(len(boxes), dtype=torch.bool)
    # given 2 lines, get only the boxes which coordinates are within the lines (boxes are in xyxy format)
    if h_lines is not None:  # horizontal lines (y1, y2)
        box_condition = (boxes[:, 1] > h_lines[0]) & (boxes[:, 3] < h_lines[1])
        keep_indices = torch.where(
            box_condition,
            torch.tensor([True]).to(device),
            torch.tensor([False]).to(device),
        )
        # draw the lines on the image
        # cv.line(image, (0, h_lines[0]), (image.shape[1], h_lines[0]), (0, 255, 0), 2)
        # cv.line(image, (0, h_lines[1]), (image.shape[1], h_lines[1]), (0, 255, 0), 2)
    elif v_lines is not None:  # vertical lines (x1, x2)
        box_condition = (boxes[:, 0] > v_lines[0]) & (boxes[:, 2] < v_lines[1])
        keep_indices = torch.where(
            box_condition,
            torch.tensor([True]).to(device),
            torch.tensor([False]).to(device),
        )
        # draw the lines on the image
        # cv.line(image, (v_lines[0], 0), (v_lines[0], image.shape[0]), (0, 255, 0), 2)
        # cv.line(image, (v_lines[1], 0), (v_lines[1], image.shape[0]), (0, 255, 0), 2)
    elif rect is not None:
        box_condition = (
            (boxes[:, 0] > rect[0])
            & (boxes[:, 1] > rect[1])
            & (boxes[:, 2] < rect[2])
            & (boxes[:, 3] < rect[3])
        )
        keep_indices = torch.where(
            box_condition,
            torch.tensor([True]).to(device),
            torch.tensor([False]).to(device),
        )
        # draw the rectangle on the image
        # cv.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    if keep_indices.any():
        # Find the index of the biggest box within the defined area
        biggest_box_index = torch.argmax(
            (boxes[keep_indices][:, 2] - boxes[keep_indices][:, 0])
            * (boxes[keep_indices][:, 3] - boxes[keep_indices][:, 1])
        )

        # Update keep_indices to only keep the index of the biggest box within the defined area
        keep_indices = keep_indices.nonzero()[biggest_box_index]

    # Filter the detections with the updated keep_indices
    valid_box = boxes[keep_indices]
    valid_score = scores[keep_indices]

    return valid_box, valid_score


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

    # FIX: might need to tune the score normalization for detecting classes like the water crate

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
    # FIX: Does not remove boxes that are contained in other boxes
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


# TODO: add Pallets ROI
