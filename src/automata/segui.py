import gradio as gr
from automata.prophet import Prophet
import numpy as np
import gradiops.ops as grops
from prompts import Dataset
from gradiops.ops import mask_to_annotatedimage, ultra_masks

ds = Dataset()


class Interface:
    def __init__(self):
        self.prophet = None
        self.interface()
        self.classes = ds.legend

    def load_model(self, progress=gr.Progress(track_tqdm=True)):
        progress(0, "Loading models...", 100)
        self.prophet = Prophet(models=["sam"])
        print("Models loaded!")

    def predict_ground(self, image: np.ndarray):
        preds_with_masks = self.prophet.pred_seg(image)

        # prophecies_annotations = grops.to_annotations(prophecies, ds.legend)
        # comparison_dataframe = grops.to_dataframe(comparison, ds.legend)

        # return (image, prophecies_annotations), comparison_dataframe

    def predict(self, image: np.ndarray):
        results = self.prophet.sam.infer(image)
        annotations = ultra_masks(results[0])

        return image, annotations

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    start = gr.Button("Predict")

            with gr.Row():
                output = gr.AnnotatedImage()

            demo.load(self.load_model)

            start.click(self.predict, inputs=[image], outputs=[output])

        self.demo = demo


def run():
    gui = Interface()
    gui.demo.launch()


if __name__ == "__main__":
    run()
