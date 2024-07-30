from automata.prophet import Prophet
import numpy as np
import torch
import os
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


class Pipeline:
    def __init__(self):
        self.device = get_device()
        self.prophet = Prophet(models=["dino", "oracle"])

    def predict(self, image: np.ndarray):
        prophecies = self.prophet.predict(image)
        return prophecies

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
        self,
        detections: dict[int, tuple[torch.Tensor, torch.Tensor]],
        img_path: Path,
        shape: tuple[int, int],
    ):
        lines = []
        for id, (boxes, scores) in detections.items():
            if boxes.numel() == 0:  # Check if boxes is empty
                continue
            # convert to yolo format
            boxes = ops.box_convert(boxes, "xyxy", "cxcywh")
            # normalize the boxes between 0 and 1
            boxes = boxes / torch.tensor([shape[1], shape[0], shape[1], shape[0]]).to(
                boxes.device
            )

            for idx in range(boxes.shape[0]):
                box = boxes[idx]
                # score = scores[idx]
                line = f"{id} {box[0]} {box[1]} {box[2]} {box[3]}"
                print("line", line)
                lines.append(line)

        label_path = img_path.with_suffix(".txt")
        with open(label_path, "w") as file:
            file.write("\n".join(lines))
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
        predictions = pipeline.predict(img)
        h, w = img.shape[:2]
        pipeline.write_labels(predictions, image, (h, w))


if __name__ == "__main__":
    run()
