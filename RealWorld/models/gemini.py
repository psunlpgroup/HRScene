import io

from .base import BaseModel

from google import genai
from google.genai import types


class Gemini(BaseModel):
    MODEL_NAME = "Gemini"

    def __init__(self, model_path: str, **model_kwargs) -> None:
        super().__init__(model_path, **model_kwargs)
        self.model = model_path


    def _init_model(self, model_path: str, **model_kwargs) -> None:
        self.client = genai.Client(**model_kwargs)


    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        
        content = [
            inputs["prompt"],
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        ]

        return content


    def generate(self, inputs: dict, **generation_kwargs) -> str:
        output = self.client.models.generate_content(model=self.model, contents=inputs, **generation_kwargs)
        response = output.text

        return response
