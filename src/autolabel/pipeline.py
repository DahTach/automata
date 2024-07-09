"""
AutoLabel pipeline
1. Crop pallet from the image
2. Grounding Dino object detection
3. PaliGemma classification
"""

from automata.prophet import Prophet
import re
import numpy as np
import torch
from prompts import Dataset
import os
from models.pgemma import PaliGemma
from models.dino import Dino
from operations.nms import roi
import torchvision.ops as ops
import cv2 as cv
from pathlib import Path
from tqdm import tqdm


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


ds = Dataset()


class Pipeline:
    def __init__(self):
        self.prompts = ds.captions
        self.device = get_device()
        self.dino = Dino(size="small")
        self.gemma = PaliGemma()

    def crop(self, image: np.ndarray, box: list[int]):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def pallet(self, image: np.ndarray, prompt: str):
        """Returns the bounding box of the pallet in the image.
        Args:
            image: numpy array of the image
            prompt: prompt for the object detection model
        Returns:
            box: tensor of the pallet bounding box in xyxy format
        """
        boxes, scores = self.dino.infer(image, prompt)
        box, _ = roi(detections=(boxes, scores), v_lines=(0, 2200))
        return box

    def guess_id(self, prophecy: str, options: dict[str, int]):
        prophecy = re.sub(r"[^a-zA-Z0-9\s]", "", prophecy).lower().strip()

        for key, val in options.items():
            if prophecy in key:
                return val
        return len(options)

    def classify(
        self,
        image: np.ndarray,
        detections: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ):
        prophesies = {
            i: (torch.tensor([]).to(self.device), torch.tensor([]).to(self.device))
            for i in range(len(self.gemma.options.keys()) + 1)
        }

        for _, (boxes, scores) in detections.items():
            for idx in range(boxes.shape[0]):
                box = boxes[idx]
                score = scores[idx]

                img = self.crop(image, box.int().tolist())
                prophecy = self.gemma.generate(img)
                class_id = self.guess_id(prophecy, self.gemma.options)

                prophesies[class_id] = (
                    torch.cat((prophesies[class_id][0], box.view(1, -1))),
                    torch.cat((prophesies[class_id][1], score.view(1))),
                )

        return prophesies

    def pipe(self, image: np.ndarray, prompts: dict[int, list[str]]):
        pallet = self.pallet(image, "loaded pallet")
        refsys = pallet[0].int().tolist()
        pallet_img = self.crop(image, refsys)
        detections = self.dino.predict(
            pallet_img, prompts
        )  # prompts: dict[int, list[str]],

        classified = self.classify(pallet_img, detections)
        orig_sysref = self.det_refsys(classified, refsys)
        return orig_sysref

    def pipeline(self, images: list[Path], prompts: dict[int, list[str]]):
        for image in tqdm(images):
            img = cv.imread(str(image))
            detections = self.pipe(img, prompts)
            self.write_labels(detections, image)

    def det_refsys(
        self,
        detections: dict[int, tuple[torch.Tensor, torch.Tensor]],
        refsys: list[int],
    ):
        for id, (boxes, scores) in detections.items():
            detections[id] = (self.box_refsys(boxes, refsys), scores)
        return detections

    def box_refsys(self, boxes: torch.Tensor, refsys: list[int]):
        """
        Changes the reference system of the boxes.

        Args:
            boxes: tensor of boxes in xyxy format
            refsys: reference system (x1, y1, x2, y2)

        Returns:
            tensor of boxes in xyxy format for the new reference system
        """

        if boxes.numel() == 0:  # Check if boxes is empty
            return torch.tensor([]).to(boxes.device)

        x1, y1, _, _ = refsys
        adjustments = torch.tensor([x1, y1, x1, y1]).to(boxes.device)
        return boxes + adjustments.unsqueeze(0)  # Adds the adjustments to each box

    def draw(
        self,
        image: np.ndarray,
        predictions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ):
        for id, (boxes, scores) in predictions.items():
            for idx in range(boxes.shape[0]):
                box = boxes[idx].int().tolist()
                # score = scores[idx]
                cv.rectangle(
                    img=image,
                    pt1=(box[0], box[3] - 20),
                    pt2=(box[2], box[1]),
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv.putText(
                    img=image,
                    text=str(id),
                    org=(box[0], box[3] - 5),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2,
                )
        return image

    def show(self, img: np.ndarray):
        try:
            cv.imshow("results", img)
        except KeyboardInterrupt:
            exit(0)
        key = cv.waitKey(0) & 0xFF
        if key == 27 or key == ord("q") or key == 3:
            exit(0)

    def write_labels(
        self, detections: dict[int, tuple[torch.Tensor, torch.Tensor]], img_path: Path
    ):
        yololines = []
        for id, (boxes, scores) in detections.items():
            if boxes.numel() == 0:  # Check if boxes is empty
                continue
            boxes = ops.box_convert(boxes, "xyxy", "cxcywh")
            for idx in range(boxes.shape[0]):
                box = boxes[idx]
                score = scores[idx]
                yololines.append(f"{id} {box[0]} {box[1]} {box[2]} {box[3]}")

        label_path = img_path.with_suffix(".txt")
        with open(label_path, "w") as file:
            file.writelines(yololines)
        print(f"Labels written to {label_path}")


def get_images_path(top: Path):
    images = []
    for root, dirs, files in top.walk():
        for name in files:
            if name.endswith((".jpg", ".png")):
                images.append(root / name)
    return images


def run():
    import sys

    pipeline = Pipeline()
    images = get_images_path(Path(sys.argv[1]))
    for image in tqdm(images):
        img = cv.imread(str(image))
        predictions = pipeline.pipe(img, pipeline.prompts)
        pipeline.write_labels(predictions, image)


if __name__ == "__main__":
    run()
