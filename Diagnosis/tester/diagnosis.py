import importlib
import json
import os
import time

from .base import BaseTester
from evaluators import draw_heatmap
from models.base import BaseModel

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


class DiagnosisTester(BaseTester):
    def __init__(self, model: BaseModel, dataset_name: str, num_samples: int = 500) -> None:
        super().__init__(model, dataset_name, num_samples)
        self.responses = []
        self.labels = []
        self.model_name = getattr(self.model, "MODEL_NAME", "Custom Model")
        self.experiment_name = f"{self.task} {self.model_name} {self.grid_size}x{self.grid_size}"
    

    def _init_dataset(self, dataset_name: str) -> None:
        self.dataset = load_dataset("Wenliang04/HRScene", dataset_name)
        self.dataset = self.dataset['test']


    def run(self, **generation_kwargs) -> None:
        for i, sample in tqdm(enumerate(self.dataset), total=self.num_samples, desc=self.experiment_name):
            if i >= self.num_samples:
                break

            prompt_template = getattr(importlib.import_module("prompts"), f"{self.task}_prompt")()
            if self.task == "whitebackground":
                prompt = prompt_template.format(question=sample["question"])
            elif self.task == "complexgrid":
                prompt = prompt_template.format(caption=sample["caption"])
            sample['prompt'] = prompt

            inputs = self.model.process_inputs(sample)
            response = self.model.generate(inputs, **generation_kwargs)
            # remove the question from the response if it exists
            response = response.split(sample["prompt"])[-1]
            response = {
                "response": response,
                "metadata": sample
            }

            self.responses.append(response)
            self.labels.append(sample["answer"])


    def eval(self, eval_results_dir: str | None = None) -> None:
        if not eval_results_dir:
            eval_results_dir = os.path.join("results", "complexgrid", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(eval_results_dir, exist_ok=True)

        metrics = getattr(importlib.import_module("evaluators"), f"default_{self.task}_metrics")

        eval_results = metrics(self.responses, self.labels)
        scores = np.array([result["score"] for result in eval_results])
        
        with open(os.path.join(eval_results_dir, "eval_results.jsonl"), "w") as f:
            for result in eval_results:
                f.write(json.dumps(result) + "\n")
        
        with open(os.path.join(eval_results_dir, "eval_scores.txt"), "w") as f:
            f.write(f"Average score: {scores.mean()}\n")
            f.write(f"Median score: {np.median(scores)}\n")
            f.write(f"Standard deviation: {scores.std()}\n")
        
        draw_heatmap(eval_results, self.grid_size, self.experiment_name, eval_results_dir)

        print(f"Finished evaluation for experiment: {self.experiment_name}")
        print(f'- Model: {self.model_name}')
        print(f'- Dataset: ComplexGrid {self.grid_size}x{self.grid_size}')
        print(f'- Average score: {float(scores.mean()):.2f}')
        print(f'- Median score: {float(np.median(scores)):.2f}')
        print(f'- Standard deviation: {float(scores.std()):.2f}')
        print(f'- Results saved in: {eval_results_dir}')
