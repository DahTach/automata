from gradio.external import re
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch
import os


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


class Segmenter:
    def __init__(self):
        self.device = get_device()
        self.model = FastSAM("FastSAM-s.pt").to(self.device)  # or FastSAM-x.pt

    def infer(self, image: np.ndarray):
        predictions = self.model(
            image,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.3,
            iou=0.7,
        )
        return predictions

    def predict_box(self, image: np.ndarray, box: torch.Tensor):
        box_img = self.crop_box(image, box.int().tolist())

        results = self.infer(box_img)

        # results = self.mask_box(box, results)

        # results is now based on the cropped image, so we need to convert the masks to the original image size
        results = self.refit_masks(image, results, box.int().tolist())

        return results

    def crop_box(self, image: np.ndarray, box: list[int]):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def refit_masks(self, target_img: np.ndarray, results, crop_box: list[int]):
        """Refits the masks to the original image size."""

        # get the target image size
        targ_height, targ_width = target_img.shape[:2]

        # get absolute coordinates of the bounding box
        x1, y1, x2, y2 = crop_box

        empty_masks = 0

        mask_results = torch.tensor([]).to(self.device)

        for res in results:
            # Get the cropped mask data: res.masks.data is a tensor of shape (n, h, w)
            msk = res.masks.data
            # check if the mask are empty
            if msk.sum() == 0:
                continue
            # iterate over the masks in the batch
            for m in msk:
                # check if the mask is empty
                if m.sum() == 0:
                    print(f"Empty mask {empty_masks}")
                    empty_masks += 1
                    continue
                # Create a binary mask of the same size as the target size mask
                tm = torch.ones((targ_height, targ_width), dtype=torch.bool).to(
                    self.device
                )
                tm = ~tm
                tm = tm * 0
                tm = torch.zeros((targ_height, targ_width), dtype=torch.bool).to(
                    self.device
                )
                tm = tm.int()
                # Apply the cropped mask to the target binary mask within the bounding box region
                tm[y1:y2, x1:x2] = m.to(self.device)
                # Replace the results mask with the shifted mask
                mask_results = torch.cat((mask_results, tm.unsqueeze(0)), dim=0)

        return mask_results

    def mask_box(self, bbox: torch.Tensor, results):
        """Modifies the bounding box properties and calculates IoU between masks and bounding box."""
        # TODO: check every step of this function (was ai generated)
        if results[0].masks is not None:
            assert bbox[2] != 0 and bbox[3] != 0

            # Extract masks data and original shape information from the results
            masks = results[0].masks.data
            target_height, target_width = results[0].orig_shape
            h = masks.shape[1]
            w = masks.shape[2]

            # Adjust bounding box coordinates based on target dimensions
            if h != target_height or w != target_width:
                bbox = torch.round(
                    bbox
                    * torch.tensor(
                        [
                            w / target_width,
                            h / target_height,
                            w / target_width,
                            h / target_height,
                        ]
                    )
                )

                # Ensure bounding box coordinates are within image boundaries
                bbox[0] = torch.clamp(bbox[0], 0)
                bbox[1] = torch.clamp(bbox[1], 0)
                bbox[2] = torch.min(torch.round(bbox[2]), w)
                bbox[3] = torch.min(torch.round(bbox[3]), h)

            # Calculate IoU between masks and bounding box
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

            masks_area = torch.sum(
                masks[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])],
                dim=(1, 2),
            )
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            iou = masks_area / union

            # Find the index of the mask with maximum IoU and update masks data
            max_iou_index = torch.argmax(iou)

            results[0].masks.data = torch.unsqueeze(masks[max_iou_index], dim=0)

        return results

    def box_prompt(self, results, bbox):
        """Modifies the bounding box properties and calculates IoU between masks and bounding box."""
        if results[0].masks is not None:
            assert bbox[2] != 0 and bbox[3] != 0
            masks = results[0].masks.data
            target_height, target_width = results[0].orig_shape
            h = masks.shape[1]
            w = masks.shape[2]
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height),
                ]
            bbox[0] = max(round(bbox[0]), 0)
            bbox[1] = max(round(bbox[1]), 0)
            bbox[2] = min(round(bbox[2]), w)
            bbox[3] = min(round(bbox[3]), h)

            # IoUs = torch.zeros(len(masks), dtype=torch.float32)
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

            masks_area = torch.sum(
                masks[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])],
                dim=(1, 2),
            )
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            iou = masks_area / union
            max_iou_index = torch.argmax(iou)

            results[0].masks.data = torch.tensor(
                np.array([masks[max_iou_index].cpu().numpy()])
            )
        return results

    def cpu_mask_box(self, bbox: list[int], results):
        """Modifies the bounding box properties and calculates IoU between masks and bounding box."""

        if results[0].masks is not None:
            assert bbox[2] != 0 and bbox[3] != 0

            # Extract masks data and original shape information from the results
            masks = results[0].masks.data
            target_height, target_width = results[0].orig_shape
            h = masks.shape[1]
            w = masks.shape[2]

            # Adjust bounding box coordinates based on target dimensions
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height),
                ]

            # Ensure bounding box coordinates are within image boundaries
            bbox[0] = max(round(bbox[0]), 0)
            bbox[1] = max(round(bbox[1]), 0)
            bbox[2] = min(round(bbox[2]), w)
            bbox[3] = min(round(bbox[3]), h)

            # Calculate IoU between masks and bounding box
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            masks_area = torch.sum(
                masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2)
            )
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            iou = masks_area / union

            max_iou_index = torch.argmax(iou)
            results[0].masks.data = torch.tensor(
                np.array([masks[max_iou_index].cpu().numpy()])
            )

        return results

    def predict(self, image: np.ndarray):
        predictions = self.infer(image)
