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
    def __init__(
            self, model: BaseModel, dataset_name: str, num_samples: int = 500, cache_dir: str | None = None
        ) -> None:
        super().__init__(model, dataset_name, num_samples)
        self.model_name = getattr(self.model, "MODEL_NAME", "Custom Model")
        self.experiment_name = f"{self.task} {self.model_name} {self.grid_size}x{self.grid_size}"
        if cache_dir and os.path.exists(os.path.join(cache_dir, "responses.jsonl")):
            self.save_dir = cache_dir
            self.finished_items = set(
                [json.loads(line)["id"] for line in open(os.path.join(cache_dir, "responses.jsonl"), "r")]
            )
        else:
            self.save_dir = os.path.join("results", "diagnosis", time.strftime("%Y%m%d_%H%M%S"))
            self.finished_items = set()
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Results will be saved in: {self.save_dir}")
    

    def _init_dataset(self, dataset_name: str) -> None:
        self.dataset = load_dataset("Wenliang04/HRScene", dataset_name)
        self.dataset = self.dataset['test']


    def run(self, **generation_kwargs) -> None:
        for i, sample in tqdm(enumerate(self.dataset), total=self.num_samples, desc=self.experiment_name):
            if i >= self.num_samples:
                break
            if sample["id"] in self.finished_items:
                continue
            prompt_template = getattr(importlib.import_module("prompts"), f"{self.task}_prompt")()
            if self.task == "whitebackground":
                prompt = prompt_template.format(question=sample["question"])
            elif self.task == "complexgrid":
                prompt = prompt_template.format(caption=sample["caption"])
            sample['prompt'] = prompt

            inputs = self.model.process_inputs(sample)
            response = self.model.generate(inputs, **generation_kwargs)

            # extract the response if output format is list
            response = response if isinstance(response, str) else response[0]
            # remove the prompt from the response if it exists
            response = response.split(prompt)[-1]
            # llava-next returns full history of conversation, clean it up
            response = response.split('assistant\n')[-1].strip()

            if self.task == "whitebackground":
                response = {
                    "id": sample["id"],
                    "question": sample["question"],
                    "response": response,
                    "answer": sample["answer"]
                }
            elif self.task == "complexgrid":
                response = {
                    "id": sample["id"],
                    "caption": sample["caption"],
                    "response": response,
                    "answer": sample["answer"]
                }

            with open(os.path.join(self.save_dir, "responses.jsonl"), "a") as f:
                f.write(json.dumps(response) + "\n")


    def eval(self) -> None:
        metrics = getattr(importlib.import_module("evaluators"), f"default_{self.task}_metrics")

        with open(os.path.join(self.save_dir, "responses.jsonl"), "r") as f:
            if self.task == "whitebackground":
                responses = [{
                    "id": json.loads(line)["id"],
                    "question": json.loads(line)["question"],
                    "response": json.loads(line)["response"],
                    "answer": json.loads(line)["answer"]
                } for line in f]
            elif self.task == "complexgrid":
                responses = [{
                    "id": json.loads(line)["id"],
                    "caption": json.loads(line)["caption"],
                    "response": json.loads(line)["response"],
                    "answer": json.loads(line)["answer"]
                } for line in f]
        eval_results = metrics(responses)
        scores = np.array([result["score"] for result in eval_results])
        
        with open(os.path.join(self.save_dir, "eval_results.jsonl"), "w") as f:
            for result in eval_results:
                f.write(json.dumps(result) + "\n")
        
        with open(os.path.join(self.save_dir, "eval_scores.txt"), "w") as f:
            f.write(f"Average score: {scores.mean()}\n")
            f.write(f"Median score: {np.median(scores)}\n")
            f.write(f"Standard deviation: {scores.std()}\n")
        
        draw_heatmap(eval_results, self.grid_size, self.experiment_name, self.save_dir)

        print(f"Finished evaluation for experiment: {self.experiment_name}")
        print(f'- Model: {self.model_name}')
        print(f'- Dataset: ComplexGrid {self.grid_size}x{self.grid_size}')
        print(f'- Average score: {float(scores.mean()):.2f}')
        print(f'- Median score: {float(np.median(scores)):.2f}')
        print(f'- Standard deviation: {float(scores.std()):.2f}')
        print(f'- Results saved in: {self.save_dir}')
