import importlib
import json
import os
import time
from typing import Callable

from .base import BaseTester
from models.base import BaseModel

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


class RealWorldTester(BaseTester):
    def __init__(self, model: BaseModel, dataset_name: str, split: str, num_samples: int = 10000) -> None:
        super().__init__(model, dataset_name, split, num_samples)
        self.responses = []
        self.labels = []
        self.model_name = getattr(self.model, "MODEL_NAME", "Custom Model")
        self.experiment_name = f"RealWorld {self.model_name}"
    

    def _init_dataset(self, dataset_name: str, split: str) -> None:
        self.dataset = load_dataset("Wenliang04/HRScene", dataset_name)
        self.dataset = self.dataset[split]
        self.num_samples = min(len(self.dataset), self.num_samples)


    def run(self, **generation_kwargs) -> None:
        for i, sample in tqdm(enumerate(self.dataset), total=self.num_samples, desc='Inferencing RealWorld'):
            if i >= self.num_samples:
                break
            prompt_template = getattr(importlib.import_module("prompts"), f"realworld_prompt")()
            prompt = prompt_template.format(question=sample["question"])
            sample["prompt"] = prompt
            inputs = self.model.process_inputs(sample)
            response = self.model.generate(inputs, **generation_kwargs)

            # extract the response if output format is list
            response = response if isinstance(response, str) else response[0]
            # remove the question from the response if it exists
            response = response.split(sample["question"])[-1]
            # llava-next returns full history of conversation, clean it up
            response = response.split('assistant\n')[-1].strip()
            
            response = {
                "response": response,
                "metadata": sample
            }

            self.responses.append(response)
            self.labels.append(sample["answer"])


    def eval(self, eval_results_dir: str | None = None, metrics: str | Callable = "default") -> None:
        if not eval_results_dir:
            eval_results_dir = os.path.join("results", "realworld", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(eval_results_dir, exist_ok=True)
        if metrics == "default":
            metrics = getattr(importlib.import_module("evaluators"), "default_realworld_metrics")
        else:
            metrics = metrics

        eval_results = metrics(self.responses, self.labels)
        
        if not self.split == "test":
            scores = np.array([result["score"] for result in eval_results])

            with open(os.path.join(eval_results_dir, "eval_results.jsonl"), "w") as f:
                for result in eval_results:
                    f.write(json.dumps(result) + "\n")
            
            with open(os.path.join(eval_results_dir, "eval_scores.txt"), "w") as f:
                f.write(f"Average score: {scores.mean()}\n")
                f.write(f"Median score: {np.median(scores)}\n")
                f.write(f"Standard deviation: {scores.std()}\n")

                print(f"Finished evaluation for experiment: {self.experiment_name}")
                print(f'- Model: {self.model_name}')
                print(f'- Average score: {float(scores.mean()):.2f}')
                print(f'- Median score: {float(np.median(scores)):.2f}')
                print(f'- Standard deviation: {float(scores.std()):.2f}')
                print(f'- Results saved in: {eval_results_dir}')
        else:
            submission = {str(item['id']): item['parsed_response'] for item in eval_results}

            with open(os.path.join(eval_results_dir, "predictions.jsonl"), "a") as f:
                for result in eval_results:
                    result.pop("score")
                    result.pop("answer")
                    f.write(json.dumps(result) + "\n")

            with open(os.path.join(eval_results_dir, "submission.json"), "w") as f:
                json.dump(submission, f)

            print(f"Finished parsing, results saved in: {eval_results_dir}. Ready for submission.")
