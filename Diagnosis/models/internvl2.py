from .base import BaseModel
from .utils import internvl2_load_image

import torch
from transformers import AutoModel, AutoTokenizer


class Internvl2(BaseModel):
    MODEL_NAME = "Internvl2"
    
    def __init__(self, model_path: str, **model_kwargs) -> None:
        super().__init__(model_path, **model_kwargs)
        self.device = self.model.device


    def _init_model(self, model_path: str, **model_kwargs) -> None:
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]
        pixel_values = internvl2_load_image(image)
        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=self.device)

        inputs = {
            "pixel_values": pixel_values,
            "prompt": inputs["prompt"]
        }

        return inputs
    
    
    def generate(self, inputs: dict, **generation_kwargs) -> str:
        with torch.inference_mode():
            prompt, pixel_values = inputs["prompt"], inputs["pixel_values"]
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_kwargs)
            
            return response
