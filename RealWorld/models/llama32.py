from .base import BaseModel

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoProcessor, MllamaForConditionalGeneration


class Llama32(BaseModel):
    MODEL_NAME = "Llama32"

    def __init__(self, model_path: str, **model_kwargs) -> None:
        super().__init__(model_path, **model_kwargs)
        self.device = self.model.device


    def _init_model(self, model_path: str, **model_kwargs) -> None:
        self.model = MllamaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)
    

    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]
        prompt = f"<|image|><|begin_of_text|>{inputs['prompt']}"

        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        return inputs

    
    def post_process(self, text: str) -> str:
        text = text.split("<|end_header_id|>")[-1]
        text = text.replace("<|eot_id|>","")

        return text


    def generate(self, inputs: dict, **generation_kwargs) -> str:
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
            outputs = self.processor.decode(outputs[0], skip_special_tokens=True)
            response = self.post_process(outputs)

        return response
    