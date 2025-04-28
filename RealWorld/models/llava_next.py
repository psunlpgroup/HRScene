from .base import BaseModel

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


class LlavaNext(BaseModel):
    MODEL_NAME = "LlavaNext"

    def __init__(self, model_path: str, **model_kwargs) -> None:
        super().__init__(model_path, **model_kwargs)
        self.device = self.model.device


    def _init_model(self, model_path: str, **model_kwargs) -> None:
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.processor = LlavaNextProcessor.from_pretrained(model_path)


    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]

        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": inputs["prompt"]},
                {"type": "image"},
            ],
        }]
        prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        return inputs
    

    def post_process(self, text: str) -> str:
        text = text.split("assistant\n\n\n")[-1]

        return text
    
    
    def generate(self, inputs: dict, **generation_kwargs) -> str:
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            response = self.post_process(response)
        
        return response
