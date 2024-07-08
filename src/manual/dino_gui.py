import gradio as gr
from automata.prophet import Prophet
import numpy as np
import gradiops.ops as grops
from prompts import Dataset
from operations.nms import roi
import cv2 as cv
from tqdm import tqdm


class Interface:
    def __init__(self):
        self.ds = Dataset()
        self.sample_images = self.ds.sample_images
        self.interface()

    def load_model(self, progress=gr.Progress(track_tqdm=True)):
        progress(0, "Loading models...", 100)
        self.prophet = Prophet(models=["dino"])
        progress(100, "Models loaded!")
        print("Models loaded!")
        return gr.Button("Predict", variant="primary", interactive=True)

    def predict(
        self,
        images: list[np.ndarray],
        prompt: str,
        progress=gr.Progress(track_tqdm=True),
    ):
        annotated = []

        progress(0, "Predicting...")

        for image, _ in tqdm(images):
            # image = cv.imread(img)

            boxes, scores = self.prophet.dino.infer(image, prompt)

            predictions = {0: (boxes, scores)}

            annotations = grops.to_annotations(predictions, [prompt])

            annotated.append((image, annotations))

        return self.annotated_gallery(annotated)

    def annotated_gallery(self, annotated):
        gallery = []
        for image, annotation in annotated:
            res = gr.AnnotatedImage(value=(image, annotation))
            gallery.append(res)

        return gallery

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                input = gr.Gallery(
                    value=self.sample_images,
                    type="numpy",
                    columns=5,
                )

            with gr.Row():
                prompt = gr.Textbox(
                    value="loaded pallet", placeholder="Enter prompt here..."
                )
                start = gr.Button(
                    "Loading Models...", variant="primary", interactive=False
                )

            with gr.Row():
                results = []

            for _ in range(len(self.sample_images)):
                with gr.Row():
                    res = gr.AnnotatedImage()
                results.append(res)

            start.click(self.predict, inputs=[input, prompt], outputs=results)

            demo.load(self.load_model, outputs=[start])

        self.demo = demo


def run():
    gui = Interface()
    gui.demo.launch(allowed_paths=gui.sample_images)


if __name__ == "__main__":
    run()
