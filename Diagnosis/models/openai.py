import base64
import io

from .base import BaseModel

from openai import OpenAI


class GPT(BaseModel):
    MODEL_NAME = "GPT"

    def __init__(self, model_path: str, **model_kwargs) -> None:
        super().__init__(model_path, **model_kwargs)
        self.model = model_path


    def _init_model(self, model_path: str, **model_kwargs) -> None:
        self.client = OpenAI(**model_kwargs)


    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": inputs["prompt"]}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]

        return messages
    

    def generate(self, inputs: dict, **generation_kwargs) -> str:
        output = self.client.chat.completions.create(model=self.model, messages=inputs, **generation_kwargs)
        response = output.choices[0].message.content

        return response
