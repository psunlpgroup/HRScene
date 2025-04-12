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
    def __init__(
            self, model: BaseModel, dataset_name: str, split: str, num_samples: int = 10000, cache_dir: str | None = None
        ) -> None:
        super().__init__(model, dataset_name, split, num_samples)
        self.model_name = getattr(self.model, "MODEL_NAME", "Custom Model")
        self.experiment_name = f"RealWorld {self.model_name}"
        if cache_dir and os.path.exists(os.path.join(cache_dir, "responses.jsonl")):
            self.save_dir = cache_dir
            self.finished_items = set(
                [json.loads(line)["id"] for line in open(os.path.join(cache_dir, "responses.jsonl"), "r")]
            )
        else:
            self.save_dir = os.path.join("results", "realworld", time.strftime("%Y%m%d_%H%M%S"))
            self.finished_items = set()
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Results will be saved in: {self.save_dir}")


    def _init_dataset(self, dataset_name: str, split: str) -> None:
        self.dataset = load_dataset("Wenliang04/HRScene", dataset_name)
        self.dataset = self.dataset[split]
        self.num_samples = min(len(self.dataset), self.num_samples)


    def run(self, **generation_kwargs) -> None:
        for i, sample in tqdm(enumerate(self.dataset), total=self.num_samples, desc='Inferencing RealWorld'):
            if i >= self.num_samples:
                break
            if sample["id"] in self.finished_items:
                continue
            prompt_template = getattr(importlib.import_module("prompts"), f"realworld_prompt")()
            prompt = prompt_template.format(question=sample["question"])
            sample["prompt"] = prompt
            inputs = self.model.process_inputs(sample)
            response = self.model.generate(inputs, **generation_kwargs)

            # extract the response if output format is list
            response = response if isinstance(response, str) else response[0]
            # remove the prompt from the response if it exists
            response = response.split(prompt)[-1]
            # llava-next returns full history of conversation, clean it up
            response = response.split('assistant\n')[-1].strip()
            
            response = {
                "id": sample["id"],
                "question": sample["question"],
                "response": response,
                "answer": sample["answer"]
            }

            with open(os.path.join(self.save_dir, "responses.jsonl"), "a") as f:
                f.write(json.dumps(response) + "\n")


    def eval(self, metrics: str | Callable = "default") -> None:
        if metrics == "default":
            metrics = getattr(importlib.import_module("evaluators"), "default_realworld_metrics")
        else:
            metrics = metrics
        
        with open(os.path.join(self.save_dir, "responses.jsonl"), "r") as f:
            responses = [{
                "id": json.loads(line)["id"],
                "question": json.loads(line)["question"],
                "response": json.loads(line)["response"],
                "answer": json.loads(line)["answer"]
            } for line in f]

        eval_results = metrics(responses)
        
        if not self.split == "test":
            scores = np.array([result["score"] for result in eval_results])

            with open(os.path.join(self.save_dir, "eval_results.jsonl"), "w") as f:
                for result in eval_results:
                    f.write(json.dumps(result) + "\n")
            
            with open(os.path.join(self.save_dir, "eval_scores.txt"), "w") as f:
                f.write(f"Average score: {scores.mean()}\n")
                f.write(f"Median score: {np.median(scores)}\n")
                f.write(f"Standard deviation: {scores.std()}\n")

                print(f"Finished evaluation for experiment: {self.experiment_name}")
                print(f'- Model: {self.model_name}')
                print(f'- Average score: {float(scores.mean()):.2f}')
                print(f'- Median score: {float(np.median(scores)):.2f}')
                print(f'- Standard deviation: {float(scores.std()):.2f}')
                print(f'- Results saved in: {self.save_dir}')
        else:
            submission = {str(item['id']): item['parsed_response'] for item in eval_results}

            with open(os.path.join(self.save_dir, "submission.json"), "w") as f:
                json.dump(submission, f)

            print(f"Finished parsing, results saved in: {self.save_dir}. Ready for submission.")
