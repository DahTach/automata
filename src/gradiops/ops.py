from typing import List, Tuple, Dict
import torch
import numpy as np
import cv2 as cv


def to_annotations(
    predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]], classes: List[str]
) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """
    Args:
        predictions: Dict[int, Tuple[Tensor, Tensor]] = {id: (boxes, scores)}
    Returns:
        annotations: List[Annotation] = [(Mask, str)] where mask: Tuple[int, int, int, int] and str is the class name
    """

    def safe_list_get(lis, idx, default="unknown"):
        try:
            return lis[idx]
        except IndexError:
            return default

    annotations = []
    for id, (boxes, scores) in predictions.items():
        if boxes.numel() == 0:
            return []
        elif boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        class_name = safe_list_get(classes, id)
        for i, box in enumerate(boxes):
            annotations.append((box.int().tolist(), f"{class_name} {i}"))

    return annotations


def to_labels(
    predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]], classes: List[str]
) -> Dict[str, float]:
    """
    Args:
        predictions: Dict[int, Tuple[Tensor, Tensor]] = {id: (boxes, scores)}
    Returns:
        labels: dict[str, float] of classes and confidences
    """
    labels = {}
    for id, (boxes, scores) in predictions.items():
        class_name = classes[id]
        for i, score in enumerate(scores):
            labels[f"{class_name} {i}"] = score
    return labels


def to_dataframe(data: dict[int, list[int]], legend: List[str]):
    """
    Args:
        data (dict[int, list[int]]): old_id -> [new_id]
    Returns:
        dataframe: (Union[pandas.core.frame.DataFrame, pandas.io.formats.style.Styler, numpy.ndarray, polars.DataFrame, List[List], List, Dict[str, Any], str]): Data to display.
    """

    dataframe = {
        "data": [],
        "headers": ["before", "after"],
    }

    for id, pred_ids in data.items():
        for pred_id in pred_ids:
            dataframe["data"].append((legend[id], legend[pred_id]))

    return dataframe

    def refit_masks(self, target_img: np.ndarray, results, crop_box: list[int]):
        """Refits the masks to the original image size."""

        # get the target image size
        targ_height, targ_width = target_img.shape[:2]

        # get absolute coordinates of the bounding box
        x1, y1, x2, y2 = crop_box

        for result in results:
            # Get the cropped mask data
            msk = result.masks.data
            # Create a binary mask of the same size as the target size mask
            tm = torch.zeros((targ_height, targ_width), dtype=torch.bool)
            # Apply the cropped mask to the target binary mask within the bounding box region
            tm[y1:y2, x1:x2] = msk
            # Replace the results mask with the shifted mask
            result.masks.data = tm

        return results


def ultra_masks(masks: torch.Tensor, img_shape: Tuple[int, int], label: str = "obj"):
    """
    Args:
        masks: torch.Tensor of masks (n, h, w)
        img_shape: Tuple[int, int] = (h, w)
        label: str = "obj"
    Returns:
        annotations: List[Tuple[np.ndarray, str]] = [(mask, label)]
    """
    # FIX: masks are extremely slow to process, overheating
    # FIX: memory usage is extremely high, why?
    # maybe dealllocate masks after the models are used

    annotations = []

    max = 3
    i = 0
    empty = 0

    print(f"Image shape: {img_shape}")

    for msk in masks:
        mask = msk.cpu().numpy().squeeze()
        if mask.sum() == 0:
            print(f"Empty mask {empty}")
            empty += 1
            continue
        annotations.append((mask, label))
        i += 1
        if i == max:
            print(f"Stopping at {max} masks")
            break

    return annotations
