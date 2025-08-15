from utils import *

# If you set EVAL_SAMPLES 0, evaluate over full dataset. (no sampling)
EVAL_SAMPLES = 0

# print('='*10 + ' OPT baseline ' + '='*10)
# set_symlink_opt('modeling_opt_orig.py')
# for SIZE in ['6.7b', '13b', '30b']:
#     for SEQLEN in [2048]:
#         for DATASET in ["wikitext2"]:
#             cmd = "CUDA_VISIBLE_DEVICES=0 python opt.py "
#             cmd += "--model facebook/opt-%s "%(SIZE)
#             cmd += "--eval_dataset %s "%(DATASET)
#             cmd += "--seq_len %d "%(SEQLEN)
#             cmd += "--eval_samples %d "%(EVAL_SAMPLES)
#             print(cmd)
#             os.system(cmd)
#             print("-------------------------------------------")

# print('='*10 + ' OPT Tender-INT4 ' + '='*10)
# set_symlink_opt('modeling_opt_tender.py')
# for SIZE in ['6.7b', '13b', '30b']:
#     for SEQLEN in [2048]:
#         for DATASET in ["wikitext2"]:
#             for BITS in [4, 8]:
#                 DECOMP = opt_decomp_params(SIZE, BITS)
#                 cmd = "CUDA_VISIBLE_DEVICES=0 python opt.py "
#                 cmd += "--model facebook/opt-%s "%(SIZE)
#                 cmd += "--eval_dataset %s "%(DATASET)
#                 cmd += "--seq_len %d "%(SEQLEN)
#                 cmd += "--eval_samples %d "%(EVAL_SAMPLES)
#                 cmd += "--q_bits %d "%(BITS)
#                 cmd += "--decomp_factor %d "%(DECOMP)
#                 cmd += "--chunk_size %d "%(256)
#                 cmd += "--scale_factor %s "%(f"../calibration/opt/scale/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
#                 cmd += "--bias %s "%(f"../calibration/opt/bias/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp.pt")
#                 print(cmd)
#                 os.system(cmd)
#                 print("-------------------------------------------")

print('='*10 + ' OPT Tender-INT4 (all)' + '='*10)
for SIZE in ['6.7b', '13b', '30b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2"]:
            for BITS in [4, 8]:
                DECOMP = opt_decomp_params(SIZE, BITS)
                cmd = "CUDA_VISIBLE_DEVICES=3 python opt.py "
                cmd += "--model facebook/opt-%s "%(SIZE)
                cmd += "--eval_dataset %s "%(DATASET)
                cmd += "--seq_len %d "%(SEQLEN)
                cmd += "--eval_samples %d "%(EVAL_SAMPLES)
                cmd += "--q_bits %d "%(BITS)
                cmd += "--decomp_factor %d "%(DECOMP)
                cmd += "--chunk_size %d "%(256)
                cmd += "--quant_mha "
                cmd += "--scale_factor %s "%(f"../calibration/opt/scale/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp_mha.pt")
                cmd += "--bias %s "%(f"../calibration/opt/bias/2048_{SIZE}_128_{BITS}bit_{DECOMP}decomp_mha.pt")
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")
