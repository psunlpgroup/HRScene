from typing import Callable

from models.base import BaseModel


class BaseTester:
    def __init__(self, model: BaseModel, dataset_name: str, num_samples: int = 500) -> None:
        self.model = model
        self._init_dataset(dataset_name)
        self.task = dataset_name.split('_')[0]
        self.grid_size = int(dataset_name.split('_')[1].split('x')[0])
        self.num_samples = num_samples * self.grid_size * self.grid_size


    def _init_dataset(self, dataset_name: str) -> None:
        raise NotImplementedError


    def run(self, **generation_kwargs) -> None:
        raise NotImplementedError
    

    def eval(self, eval_results_dir: str, metrics: str | Callable) -> None:
        raise NotImplementedError
