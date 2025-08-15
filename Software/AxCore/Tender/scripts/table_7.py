from utils import *
import os

tasks=[
   "hellaswag" ,
   "lambada_openai",
   "arc_challenge",
   "arc_easy",
   "wic", 
   "anli_r2", 
   "winogrande", 
   "hendrycksTest-college_computer_science",
   "hendrycksTest-international_law",
   "hendrycksTest-jurisprudence"
]
task_str = ",".join(tasks)

os.chdir("../lm-evaluation-harness")

## OPT
# Baseline
print('='*10 + ' OPT Baseline ' + '='*10)
set_symlink_opt('modeling_opt_orig.py')
cmd = ("python main.py " 
    "--model hf-causal " 
    "--model_args pretrained=facebook/opt-6.7b,dtype=float32,scheme=base "
    f"--tasks {task_str} "
    "--num_fewshot 0 "
    "--no_cache "
    "--device cuda:0 ")
print(cmd)
os.system(cmd)

# Tender
print('='*10 + ' OPT Tender-INT4 ' + '='*10)
set_symlink_opt('modeling_opt_tender.py')
cmd = ("python main.py " 
    "--model hf-causal " 
    "--model_args pretrained=facebook/opt-6.7b,dtype=float32,scheme=tender "
    f"--tasks {task_str} "
    "--num_fewshot 0 "
    "--no_cache "
    "--device cuda:0 ")
print(cmd)
os.system(cmd)

## Llama
# Baseline
print('='*10 + ' LLaMA Baseline ' + '='*10)
set_symlink_llama('modeling_llama_orig.py')
cmd = ("python main.py " 
    "--model hf-causal " 
    "--model_args pretrained=baffo32/decapoda-research-llama-7B-hf,dtype=float32,scheme=base "
    f"--tasks {task_str} "
    "--num_fewshot 0 "
    "--no_cache "
    "--device cuda:0 ")
print(cmd)
os.system(cmd)

# Tender
print('='*10 + ' LLaMA Tender-INT4 ' + '='*10)
set_symlink_llama('modeling_llama_tender.py')
cmd = ("python main.py " 
    "--model hf-causal " 
    "--model_args pretrained=baffo32/decapoda-research-llama-7B-hf,dtype=float32,scheme=tender "
    f"--tasks {task_str} "
    "--num_fewshot 0 "
    "--no_cache "
    "--device cuda:0 ")
print(cmd)
os.system(cmd)
