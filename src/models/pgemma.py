from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import os
import numpy as np
from prompts import Dataset
import gradio as gr


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


class PaliGemma:
    def __init__(self):
        self.device = get_device()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            revision="bfloat16",
        ).eval()
        self.ds = Dataset()
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
        self.options = self.ds.categories
        self.prompt = f"which of these categories: {self.options.keys()} better describes the object in the image?"

    def generate(self, image: np.ndarray):
        model_inputs = self.processor(
            text=self.prompt, images=image, return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def crop_box(self, image: np.ndarray, box: list[int]):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def interface(self):
        from prompts import Dataset
        from gradiops.ops import to_annotations

        ds = self.ds

        def generate(image: np.ndarray, prompt: str):
            model_inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.model.device)
            input_len = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(
                    **model_inputs, max_new_tokens=100, do_sample=False
                )
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)

            print("decoded", decoded)
            return str(decoded)

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    start = gr.Button("Predict")

                with gr.Column():
                    prompt = gr.Textbox(label="Prompt")

            with gr.Row():
                output = gr.Textbox(label="Output", interactive=False)

            start.click(generate, inputs=[image, prompt], outputs=[output])

        self.demo = demo


def demo():
    gemma = PaliGemma()
    gemma.interface()
    gemma.demo.launch()


if __name__ == "__main__":
    demo()
