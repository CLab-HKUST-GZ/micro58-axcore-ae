from utils import *
import os

# If you set EVAL_SAMPLES 0, evaluate over full dataset. (no sampling)
EVAL_SAMPLES = 0
llama_2_dir = os.environ['LLAMA2_PATH']

print('='*10 + ' OPT baseline ' + '='*10)
set_symlink_opt('modeling_opt_orig.py')
for SIZE in ['6.7b', '13b', '66b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", 'ptb']:
            cmd = "CUDA_VISIBLE_DEVICES=0 python opt.py "
            cmd += "--model facebook/opt-%s "%(SIZE)
            cmd += "--eval_dataset %s "%(DATASET)
            cmd += "--seq_len %d "%(SEQLEN)
            cmd += "--eval_samples %d "%(EVAL_SAMPLES)
            print(cmd)
            os.system(cmd)
            print("-------------------------------------------")

print('='*10 + ' OPT Tender-INT4 ' + '='*10)
set_symlink_opt('modeling_opt_tender.py')
for SIZE in ['6.7b', '13b', '66b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", 'ptb']:
            for BITS in [4, 8]:
                DECOMP = opt_decomp_params(SIZE, BITS)
                cmd = "CUDA_VISIBLE_DEVICES=0 python opt.py "
                cmd += "--model facebook/opt-%s "%(SIZE)
                cmd += "--eval_dataset %s "%(DATASET)
                cmd += "--seq_len %d "%(SEQLEN)
                cmd += "--eval_samples %d "%(EVAL_SAMPLES)
                cmd += "--q_bits %d "%(BITS)
                cmd += "--decomp_factor %d "%(DECOMP)
                cmd += "--chunk_size %d "%(256)
                cmd += "--scale_factor %s "%(f"../calibration/opt/scale/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
                cmd += "--bias %s "%(f"../calibration/opt/bias/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

print('='*10 + ' Llama-2 baseline ' + '='*10)
set_symlink_llama('modeling_llama_orig.py')
for SIZE in ['7b', '13b', '70b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", "ptb"]:
            cmd = "CUDA_VISIBLE_DEVICES=0 python llama.py "
            cmd += "--model %s/llama-2-%s "%(llama_2_dir, SIZE)
            cmd += "--eval_dataset %s "%(DATASET)
            cmd += "--seq_len %d "%(SEQLEN)
            cmd += "--eval_samples %d "%(EVAL_SAMPLES)
            print(cmd)
            os.system(cmd)
            print("-------------------------------------------")

print('='*10 + ' Llama-2 Tender-INT4 ' + '='*10)
set_symlink_llama('modeling_llama_tender.py')
for SIZE in ['7b', '13b', '70b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", 'ptb']:
            for BITS in [4, 8]:
                DECOMP = llama2_decomp_params(SIZE, BITS)
                cmd = "CUDA_VISIBLE_DEVICES=0 python llama.py "
                cmd += "--model %s/llama-2-%s "%(llama_2_dir, SIZE)
                cmd += "--eval_dataset %s "%(DATASET)
                cmd += "--seq_len %d "%(SEQLEN)
                cmd += "--eval_samples %d "%(EVAL_SAMPLES)
                cmd += "--q_bits %d "%(BITS)
                cmd += "--decomp_factor %d "%(DECOMP)
                cmd += "--chunk_size %d "%(256)
                cmd += "--scale_factor %s "%(f"../calibration/llama/llama-2-scale/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
                cmd += "--bias %s "%(f"../calibration/llama/llama-2-bias/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

print('='*10 + ' Llama-1 baseline ' + '='*10)
set_symlink_llama('modeling_llama_orig.py')
model_name = {"7b": "baffo32/decapoda-research-llama-7B-hf",
              "13b": "JG22/decapoda-research-llama-13b"}
for SIZE in ['7b', '13b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", "ptb"]:
            cmd = "CUDA_VISIBLE_DEVICES=0 python llama.py "
            cmd += "--model %s "%(model_name[SIZE])
            cmd += "--eval_dataset %s "%(DATASET)
            cmd += "--seq_len %d "%(SEQLEN)
            cmd += "--eval_samples %d "%(EVAL_SAMPLES)
            print(cmd)
            os.system(cmd)
            print("-------------------------------------------")

print('='*10 + ' Llama-1 Tender-INT4 ' + '='*10)
set_symlink_llama('modeling_llama_tender.py')
for SIZE in ['7b', '13b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", 'ptb']:
            for BITS in [4, 8]:
                DECOMP = 14
                cmd = "CUDA_VISIBLE_DEVICES=0 python llama.py "
                cmd += "--model %s "%(model_name[SIZE])
                cmd += "--eval_dataset %s "%(DATASET)
                cmd += "--seq_len %d "%(SEQLEN)
                cmd += "--eval_samples %d "%(EVAL_SAMPLES)
                cmd += "--q_bits %d "%(BITS)
                cmd += "--decomp_factor %d "%(DECOMP)
                cmd += "--chunk_size %d "%(256)
                cmd += "--scale_factor %s "%(f"../calibration/llama/llama-1-scale/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
                cmd += "--bias %s "%(f"../calibration/llama/llama-1-bias/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")
