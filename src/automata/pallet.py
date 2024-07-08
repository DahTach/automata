import gradio as gr
from automata.prophet import Prophet
import numpy as np
import gradiops.ops as grops
from prompts import Dataset
from operations.nms import roi

ds = Dataset()


class Interface:
    def __init__(self):
        self.interface()

    def load_model(self, progress=gr.Progress(track_tqdm=True)):
        progress(0, "Loading models...", 100)
        self.prophet = Prophet(models=["dino", "fastsam"])
        progress(100, "Models loaded!")
        print("Models loaded!")

    def predict(self, image: np.ndarray, prompt: str):
        boxes, scores = self.prophet.dino.infer(image, prompt)

        boxes, scores, image = roi(
            detections=(boxes, scores), v_lines=(450, 2200), image=image
        )

        masks = self.prophet.sam.predict_box(image, boxes[0])

        h, w = image.shape[:2]

        mask_ann = grops.ultra_masks(masks=masks, img_shape=(h, w), label="pallet")

        predictions = {0: (boxes, scores)}

        annotations = grops.to_annotations(predictions, [prompt])

        annotations.extend(mask_ann)

        return image, annotations

    def interface(self):
        with gr.Blocks() as demo:
            image = gr.Image()
            prompt = gr.Textbox(
                value="loaded pallet", placeholder="Enter prompt here..."
            )
            start = gr.Button("Predict")

            output = gr.AnnotatedImage()

            demo.load(self.load_model)

            start.click(self.predict, inputs=[image, prompt], outputs=[output])

        self.demo = demo


def run():
    gui = Interface()
    gui.demo.launch()


if __name__ == "__main__":
    run()
