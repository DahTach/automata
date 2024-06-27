import os
import urllib.request
from typing import List, Tuple, Dict
import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
from groundingdino.models import build_model
from groundingdino.util.inference import Model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from PIL import Image
import utils
import operations.nms as detops
import gradio as gr


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

    def predict_class(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        boxes, scores = self.model.predict(
            image=image,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert boxes to xyxy format
        boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")

        filtered_detections = detops.nmsT(detections=(boxes, scores))

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


class GUI:
    def __init__(self):
        self.classes = [
            "bitter pack",
            "bottle pack",
            "box",
            "can pack",
            "crate",
            "keg",
        ]
        self.model = None
        self.image = None

    def load_model(self):
        try:
            self.model = Dino(size="small")
            print("DINO Model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def annotate_preds(
        self,
        predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Args:
            predictions: Dict[int, Tuple[Tensor, Tensor]] = {id: (boxes, scores)}
        Returns:
            annotations: List[Annotation] = [(Mask, str)] where mask: Tuple[int, int, int, int] and str is the class name
        """
        annotations = []
        h, w, _ = self.image.shape
        for id, (boxes, scores) in predictions.items():
            class_name = self.classes[id]
            for i, box in enumerate(boxes):
                box = box * torch.tensor([w, h, w, h], dtype=torch.float32).to(
                    box.device
                )
                annotations.append((box.int().tolist(), f"{class_name} {i}"))

        return annotations

    def label_preds(
        self, predictions: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Args:
            predictions: Dict[int, Tuple[Tensor, Tensor]] = {id: (boxes, scores)}
        Returns:
            labels: dict[str, float] of classes and confidences
        """
        labels = {}
        for id, (boxes, scores) in predictions.items():
            class_name = self.classes[id]
            for i, score in enumerate(scores):
                labels[f"{class_name} {i}"] = score
        return labels

    def update_img(self, img):
        self.image = img

    def predict(self, *prompts):
        predictions = {}

        for i, prompt in enumerate(prompts):
            if prompt == "":
                continue
            predictions[i] = self.model.predict_class(self.image, prompt)

        # Oversuppression
        cleaned_predictions = detops.oversuppression(predictions)

        nms_image = [self.image, self.annotate_preds(predictions)]
        over_image = [self.image, self.annotate_preds(cleaned_predictions)]
        nms_labels = self.label_preds(predictions)
        over_labels = self.label_preds(cleaned_predictions)

        # FIX: nms and over are the same

        return nms_image, over_image, nms_labels, over_labels

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    image.upload(self.update_img, inputs=image)
                    start = gr.Button(value="Predict")

                prompts = []
                with gr.Column():
                    for i in range(len(self.classes)):
                        prompt = gr.Textbox(label=self.classes[i])
                        prompts.append(prompt)

            with gr.Row():
                with gr.Column():
                    nms = gr.AnnotatedImage(label="NMS")
                    nms_scores = gr.Label(label="NMS Scores")
                with gr.Column():
                    over = gr.AnnotatedImage(label="Oversuppression")
                    over_scores = gr.Label(label="Oversuppression Scores")

            demo.load(self.load_model)

            start.click(
                self.predict,
                inputs=prompts,
                outputs=[nms, over, nms_scores, over_scores],
                trigger_mode="once",
            )

        self.demo = demo


def run():
    gui = GUI()
    gui.load_model()
    gui.interface()
    gui.demo.launch()


if __name__ == "__main__":
    run()
