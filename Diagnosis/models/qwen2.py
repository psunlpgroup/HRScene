from .base import BaseModel

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class Qwen2VL(BaseModel):
    MODEL_NAME = "Qwen2VL"

    def __init__(self, model_path: str, **model_kwargs):
        super().__init__(model_path, **model_kwargs)
        self.device = self.model.device

    
    def _init_model(self, model_path: str, **model_kwargs):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, **model_kwargs)


    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]

        message = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": inputs["prompt"]},
            ],
        }]
        message = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=[message], images=image, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        return inputs
    

    def generate(self, inputs: dict, **generation_kwargs) -> str:
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return response
