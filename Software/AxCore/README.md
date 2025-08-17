# AxCore Quantization & Approximation

We evaluate the results with models in perplexity and zero-shot evaluations.

## Paper's hardware configuration

+ INTEL(R) XEON(R) GOLD 6544Y
+ 4 * NVIDIA RTX 6000 Ada GPUs (48GB)

## Prerequisite

+ Ubuntu 22.04.5 LTS
+ Conda 25.1.1
+ Python 3.9
+ gcc 11.4.0
+ CUDA 12.4

## Environment

### Conda
```bash
conda create -n axcore python=3.9
conda activate axcore
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Evaluation

### Note

The scripts will automatically download the required models and datasets from the Hugging Face Hub (if not cached) and then perform the AxCore evaluation.

For llama2 family models, you can also download them from [here](https://llama.meta.com/llama-downloads). You may also need to convert the model to huggingface format using the `convert_llama_weight_to_hf.py` in `transformers/src/transformers/models/llama`.

### Perplexity Evaluation

For perplexity evaluation in Table 2, you can run the following script to evaluation all models.
```bash
sh scripts/batch_evaluation_wikitext.sh
```
Or you can use
```bash
sh scripts/evaluation_wikitext_opt_2_7b.sh      # About 30 minutes, 6GB VRAM
sh scripts/evaluation_wikitext_opt_6_7b.sh      # About 1 hour, 15 GB VRAM
sh scripts/evaluation_wikitext_opt_13b.sh       # About 2 hours, 30 GB VRAM
sh scripts/evaluation_wikitext_opt_30b.sh       # About 4 hours, 70 GB VRAM
sh scripts/evaluation_wikitext_llama2_7b.sh     # About 2 hours, 16 GB VRAM
sh scripts/evaluation_wikitext_llama2_70b.sh    # About 20 hours, 160GB VRAM
```
to evaluate a single model.

**Remember to set the `device` variable in the script to the GPU you want to use.**

The results will be saved in `results/ppl_results.txt`.

The perplexity results under our configuration are listed in the following table.
| Model | opt-2.7b | opt-6.7b | opt-13b | opt-30b | llama2-7b | llama2-70b |
| --- | --- | --- | --- | --- | --- | --- |
| PPL | 12.87 | 11.01 | 10.20 | 9.60 | 5.65 | 3.40 |

Results of perplexity evaluation can be reproduced with slight random error.


### Zero-shot Evaluation

For zero-shot evaluation in Table 3, you can run the following script to evaluation all models.
```bash
sh scripts/batch_evaluation_lm_eval.sh
```
Or you can use
```bash
sh scripts/evaluation_lm_eval_opt_30b.sh        # About 36 hours, 70GB VRAM
sh scripts/evaluation_lm_eval_llama2_70b.sh     # About 90 hours, 170GB VRAM 
```
to evaluate a single model.

**Remember to set the `device` variable in the script to the GPU you want to use.**

The results will be saved in `results/lm_eval_results.txt`.

The zero-shot results under our configuration are listed in the following table.
| Model | Arc-e | Hella. | Piqa | Wino. | Avg. |
| --- | --- | --- | --- | --- | --- |
| opt-30b | 64.86 | 72.08 | 78.07 | 68.03 | 70.76 |
| llama2-70b | 82.11 | 83.79 | 82.59 | 78.61 | 81.78 |

Results of zero-shot evaluation can be reproduced with slight random error.

### Ablation Study: mpFPMA, mpFPMA+S, and mpFPMA+S+C

For perplexity evaluation of mpFPMA/mpFPMA+S/mpFPMA+S+C in Table 2, run the corresponding script for each method:
```bash
# Method: mpFPMA
sh scripts/mpFPMA/evaluation_wikitext_opt_2_7b_mpFPMA.sh   # Approx. 25 minutes, 6GB VRAM
sh scripts/mpFPMA/evaluation_wikitext_opt_6_7b_mpFPMA.sh   # Approx. 50 minutes, 15GB VRAM
sh scripts/mpFPMA/evaluation_wikitext_opt_13b_mpFPMA.sh    # Approx. 100 minutes, 30GB VRAM
sh scripts/mpFPMA/evaluation_wikitext_opt_30b_mpFPMA.sh    # Approx. 200 minutes, 70GB VRAM

# Method: mpFPMA+S
sh scripts/mpFPMA_S/evaluation_wikitext_opt_2_7b_mpFPMA_S.sh   # Approx. 25 minutes, 6GB VRAM
sh scripts/mpFPMA_S/evaluation_wikitext_opt_6_7b_mpFPMA_S.sh   # Approx. 50 minutes, 15GB VRAM
sh scripts/mpFPMA_S/evaluation_wikitext_opt_13b_mpFPMA_S.sh    # Approx. 100 minutes, 30GB VRAM
sh scripts/mpFPMA_S/evaluation_wikitext_opt_30b_mpFPMA_S.sh    # Approx. 200 minutes, 70GB VRAM

# Method: mpFPMA+S+C
sh scripts/mpFPMA_S_C/evaluation_wikitext_opt_2_7b_mpFPMA_S_C.sh   # Approx. 25 minutes, 6GB VRAM
sh scripts/mpFPMA_S_C/evaluation_wikitext_opt_6_7b_mpFPMA_S_C.sh   # Approx. 50 minutes, 15GB VRAM
sh scripts/mpFPMA_S_C/evaluation_wikitext_opt_13b_mpFPMA_S_C.sh    # Approx. 100 minutes, 30GB VRAM
sh scripts/mpFPMA_S_C/evaluation_wikitext_opt_30b_mpFPMA_S_C.sh    # Approx. 200 minutes, 70GB VRAM
```
to evaluate each single model.

**Remember to set the `device` variable in the script to the GPU you want to use.**

The perplexity results under our configuration are listed in the following table.
| Model | Method | opt-2.7b | opt-6.7b | opt-13b | opt-30b |
| --- | --- | --- | --- | --- | --- |
| PPL | mpFPMA | 13.83 | 11.83 | 10.80 | 9.99 |
| PPL | mpFPMA+S | 13.24 | 11.45 | 10.49 | 9.86 |
| PPL | mpFPMA+S+C | 13.12 | 11.14 | 10.25 | 9.74 |

Results of perplexity evaluation can be reproduced with slight random error.

### AxCore-KV Evaluation

For perplexity evaluation of AxCore-KV in Table 2, you can use
```bash
sh scripts/axcore_kv/evaluation_wikitext_opt_6_7b_kv4.sh     # About 1.5 hours, 15GB VRAM
sh scripts/axcore_kv/evaluation_wikitext_opt_13b_kv4.sh      # About 3 hours, 30GB VRAM
sh scripts/axcore_kv/evaluation_wikitext_opt_30b_kv4.sh      # About 6 hours, 70GB VRAM
sh scripts/axcore_kv/evaluation_wikitext_llama2_7b_kv4.sh    # About 3 hours, 16GB VRAM
sh scripts/axcore_kv/evaluation_wikitext_llama2_70b_kv4.sh   # About 30 hours, 160GB VRAM
```
to evaluate each single model.

**Remember to set the `device` variable in the script to the GPU you want to use.**

The perplexity results under our configuration are listed in the following table.
| Model | opt-6.7b | opt-13b | opt-30b | llama2-7b | llama2-70b |
| --- | --- | --- | --- | --- | --- |
| PPL | 11.18 | 10.59 | 9.79 | 5.82 | 3.48 |

Results of perplexity evaluation can be reproduced with slight random error.

### Tender baseline

For Tender baseline, you can run following the REAMME.md in `./Tender` to reproduce the results in Table 2.
