from typing import Callable

from models.base import BaseModel


class BaseTester:
    def __init__(self, model: BaseModel, dataset_name: str, split: str, num_samples: int = 500) -> None:
        self.model = model
        self.task = "RealWorld"
        self.split = split
        self.num_samples = num_samples
        self._init_dataset(dataset_name, split)


    def _init_dataset(self, dataset_name: str, split: str) -> None:
        raise NotImplementedError


    def run(self, **generation_kwargs) -> None:
        raise NotImplementedError
    

    def eval(self, eval_results_dir: str, metrics: str | Callable) -> None:
        raise NotImplementedError
