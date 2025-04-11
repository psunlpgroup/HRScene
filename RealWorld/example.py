# python code to set env variables for huggingface download directory
# import os
# os.environ['HF_HOME'] = '/scratch1/wmz5132/HRScenne/data/huggingface'
# os.environ['HF_DATASETS_CACHE'] = '/scratch1/wmz5132/HRScenne/data/huggingface'

from models import GPT, Qwen2VL
from tester import RealWorldTester
import torch


# example 1: use api based model to run 5 samples of ArtBench subset
model = GPT(model_path="gpt-4o-mini")
tester = RealWorldTester(model=model, dataset_name="ArtBench", split="validation", num_samples=5)
tester.run(max_tokens=100)
tester.eval()

# example 2: use local Qwen2VL model to run full realworld dataset
model = Qwen2VL(
    model_path="Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda"
)
tester = RealWorldTester(model=model, dataset_name="realworld_combined", split="test")
# feel free to set generation kwargs here or leave it empty
tester.run(max_new_tokens=100, temperature=1e-5)
tester.eval()
