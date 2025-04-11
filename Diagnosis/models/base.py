class BaseModel:
    def __init__(self, model_path: str, **model_kwargs) -> None:
        self._init_model(model_path, **model_kwargs)


    def _init_model(self, model_path: str, **model_kwargs) -> None:
        raise NotImplementedError
    

    def process_inputs(self, inputs: dict) -> dict:
        raise NotImplementedError


    def generate(self, inputs: dict) -> str:
        raise NotImplementedError
