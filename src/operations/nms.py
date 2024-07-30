import copy
import itertools
from typing import Tuple

import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
import torchvision.ops as ops
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


def nmsTest(
    detections: tuple[torch.Tensor, torch.Tensor],
    shape: Tuple[int, int],
    mask: torch.Tensor,
    iou_threshold: float = 0.7,
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
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]

    # Perform outlier removal on the filtered boxes
    keep_indices = remove_outliers(boxes, shape)

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]

    # Perform containment removal on the filtered boxes
    keep_indices = remove_contained(boxes)

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]

    # perform roi mask filtering
    keep_indices = roi_mask(boxes, mask)

    valid_boxes = boxes[keep_indices]
    valid_scores = scores[keep_indices]

    return valid_boxes, valid_scores


def remove_outliers_scores(scores: torch.Tensor) -> torch.Tensor:
    # Remove lower scores outliers
    mean = scores.mean()
    std = scores.std()
    threshold = mean - 1.6 * std
    keep_indices = scores > threshold
    return keep_indices


def nmsT(
    detections: tuple[torch.Tensor, torch.Tensor],
    shape: Tuple[int, int],
    mask: torch.Tensor,
    iou_threshold: float = 0.7,
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

    # TODO: make another method for batch NMS, this overrides the batchd

    # Perform outlier removal on the filtered boxes
    keep_indices = remove_outliers(valid_boxes, shape)

    valid_boxes = valid_boxes[keep_indices]
    valid_scores = valid_scores[keep_indices]

    # Remove scores outliers
    keep_indices = remove_outliers_scores(valid_scores)
    valid_boxes = valid_boxes[keep_indices]
    valid_scores = valid_scores[keep_indices]

    # Perform containment removal on the filtered boxes
    keep_indices = remove_contained(valid_boxes)

    valid_boxes = valid_boxes[keep_indices]
    valid_scores = valid_scores[keep_indices]

    # perform roi mask filtering
    keep_indices = roi_mask(valid_boxes, mask)

    valid_boxes = valid_boxes[keep_indices]
    valid_scores = valid_scores[keep_indices]

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


def roi_mask(
    boxes: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Perform Mask based Region of Interest (ROI) filtering on the detections
    Args:
        boxes: list of boxes (x1, y1, x2, y2) torch tensor
        mask: mask defining the ROI (0 for outside, 1 for inside), same size as the image
        threshold: threshold for the box to be considered within the mask
    Returns:
        keep_indices: indices of the boxes within the mask
    """

    # Calculate the mean values of the mask regions for each box
    mean_values = torch.tensor(
        [mask[box[1] : box[3], box[0] : box[2]].float().mean() for box in boxes.int()],
        device=boxes.device,
    )

    # Determine which boxes to keep based on the mean values and the threshold
    keep_indices = mean_values >= threshold

    return keep_indices


def class_agnostic_nms(
    input_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    threshold: float = 0.9,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    # FIX: this removes good boxes

    preds = copy.deepcopy(input_predictions)

    list_boxes, list_scores = zip(*preds.values())

    # Concatenate the boxes and scores
    all_boxes = torch.cat(list_boxes, dim=0)
    all_scores = torch.cat(list_scores, dim=0)

    # Perform NMS
    valid_indices = torchvision.ops.nms(all_boxes, all_scores, threshold)
    valid = torch.zeros(len(all_boxes), dtype=torch.bool)
    valid[valid_indices] = True

    # Calculate the split sizes based on the lengths of boxes for each class
    split_sizes = [len(boxes) for boxes in list_boxes]

    # Split the valid_indices to separate the boxes from the various classes
    split_indices = torch.split(valid, split_sizes, dim=0)

    # Filter the boxes and scores using the valid_indices
    for i, (key, (boxes, scores)) in enumerate(preds.items()):
        keep_mask = split_indices[i]
        filtered_boxes = boxes[keep_mask]
        filtered_scores = scores[keep_mask]
        preds[key] = (filtered_boxes, filtered_scores)

    return preds


def oversuppression(
    input_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    shape: Tuple[int, int],
    threshold: float = 0.7,
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

        # Normalize the scores (handle empty tensors)
        if scores1.numel() > 0:  # Check if scores1 is not empty
            scores1 = scores1 / scores1.max()
        if scores2.numel() > 0:  # Check if scores2 is not empty
            scores2 = scores2 / scores2.max()

        # Remove outliers and contained boxes before NMS

        # Perform outlier removal on the filtered boxes
        liers1 = remove_outliers(boxes1, shape)
        liers2 = remove_outliers(boxes2, shape)

        boxes1 = boxes1[liers1]
        scores1 = scores1[liers1]
        boxes2 = boxes2[liers2]
        scores2 = scores2[liers2]

        # Join the boxes
        keep = remove_contained(torch.cat((boxes1, boxes2), dim=0))

        # Split the keep indices to separate the boxes from the two classes
        keep1 = keep[: len(boxes1)]
        keep2 = keep[len(boxes1) :]

        # Filter the boxes and scores using the valid_indices
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

        # Filter again to get the final boxes and scores after outlier removal
        predictions[key1] = (filtered_boxes1, filtered_scores1)
        predictions[key2] = (filtered_boxes2, filtered_scores2)

    return predictions


def remove_big_boxes(boxes: torch.Tensor, max_size: float) -> torch.Tensor:
    """
    Remove every box from ``boxes`` which contains at least one side length
    that is bigger than ``max_size``.

    .. note::
        For sanitizing a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.SanitizeBoundingBoxes` instead.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        max_size (float): maximum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        smaller than ``max_size``
    """

    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    keep = (ws <= max_size) & (hs <= max_size)
    keep = torch.where(keep)[0]

    return keep


def remove_outliers(boxes: torch.Tensor, shape: Tuple[int, int], threshold=0.6):
    """Remove boxes that are outliers in terms of size
    Args:
        boxes: list of boxes
        shape: shape of the image (height, width)
        threshold: threshold for removing outliers
    Returns:
        list of boxes after removing outliers
    """

    max_size = min(shape) * threshold

    return remove_big_boxes(boxes, max_size)


def box_ioa(boxes1: torch.Tensor, boxes2: torch.Tensor, area_type="min"):
    """
    Return intersection-over-area (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoA values for every element in boxes1 and boxes2
    """

    if not isinstance(boxes1, torch.Tensor) or not isinstance(boxes2, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")
    if (
        boxes1.dim() != 2
        or boxes2.dim() != 2
        or boxes1.shape[1] != 4
        or boxes2.shape[1] != 4
    ):
        raise ValueError("Inputs must be 2D tensors with shape (N, 4)")
    if area_type not in ["min", "max"]:
        raise ValueError("area_type must be 'min' or 'max'")

    def _upcast(t: torch.Tensor) -> torch.Tensor:
        # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()

    area1 = ops.box_area(boxes1)
    area2 = ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1_expanded = area1[:, None].expand(area1.size(0), area2.size(0))
    area2_expanded = area2[None, :].expand(area1.size(0), area2.size(0))

    if area_type == "max":
        area = torch.max(area1_expanded, area2_expanded)
    else:  # area_type == "min"
        area = torch.min(area1_expanded, area2_expanded)

    ioa = inter / area

    return ioa


def remove_contained(boxes: torch.Tensor, threshold=0.9):
    """Remove boxes that are contained in other boxes over a certain threshold
    Args:
        boxes: list of boxes
        threshold: threshold for removing inside boxes
    Returns:
        list of boxes indices after removing inside boxes
    """
    # TODO: there's room for improvement here

    if len(boxes) < 2:
        return torch.ones(len(boxes), dtype=torch.bool)

    # Compute the IoAmin between all pairs of boxes
    ioas = box_ioa(boxes, boxes, "min")

    # Remove the diagonal (self-intersection)
    ioas.fill_diagonal_(0)

    # Find the boxes that are contained in other boxes
    inter = ioas > threshold

    # Find the boxes that are not contained in any other boxes
    keep = ~inter.any(dim=1)

    return keep


# TODO: add Pallets ROI
