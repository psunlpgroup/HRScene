import base64
import io

from .base import BaseModel

from anthropic import Anthropic


class Claude(BaseModel):
    MODEL_NAME = "Claude"

    def __init__(self, model_path: str, **model_kwargs) -> None:
        super().__init__(model_path, **model_kwargs)
        self.model = model_path

    
    def _init_model(self, model_path: str, **model_kwargs) -> None:
        self.client = Anthropic(**model_kwargs)


    def process_inputs(self, inputs: dict) -> dict:
        image = inputs["image"]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": 'image/jpeg', "data": base64_image}},
                {"type": "text", "text": inputs["prompt"]}
            ],
        }]

        return messages
    

    def generate(self, inputs: dict, **generation_kwargs) -> str:
        max_tokens = generation_kwargs.get("max_tokens", 200)

        output = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, messages=inputs, **generation_kwargs
        )
        response = output.content[0].text

        return response
