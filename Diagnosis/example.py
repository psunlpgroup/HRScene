# python code to set env variables for huggingface download directory
# import os
# os.environ['HF_HOME'] = '/scratch1/wmz5132/psunlp/HRScene/data'
# os.environ['HF_DATASETS_CACHE'] = '/scratch1/wmz5132/psunlp/HRScene/data'

from models import GPT, Llama32
from tester import DiagnosisTester
import torch


# Example 1: Run 150 complexgrid_3x3 samples on local model
model = Llama32(model_path="meta-llama/Llama-3.2-11B-Vision-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
tester = DiagnosisTester(model=model, dataset_name="complexgrid_5x5", num_samples=100)
# feel free to set generation kwargs here or leave it empty
tester.run(max_new_tokens=100, temperature=1e-5)
tester.eval()

# Example 2: Run full whitebackground_7x7 samples on api based model
model = GPT(model_path="gpt-4o-mini")
tester = DiagnosisTester(model=model, dataset_name="whitebackground_7x7")
tester.run(max_tokens=100)
tester.eval()
