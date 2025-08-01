# AxCore Quantization & Approximation

We evaluate the results with models in perplexity and zero-shot evaluations.

## Paper's hardware configuration

+ INTEL(R) XEON(R) GOLD 6544Y
+ 4 * NVIDIA RTX 6000 Ada GPUs (48GB)


## Environment

### Conda
```bash
conda create -n axcore python=3.9
conda activate axcore
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Docker
```bash
```

## Evaluation

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

Remember to set the `device` variable in the script to the GPU you want to use.

The results will be saved in `results/ppl_results.txt`.

The perplexity results under our configuration are listed in the following table.
| Model | PPL |
| --- | --- |
| opt-2.7b | 12.87 |
| opt-6.7b | 11.01 |
| opt-13b | 10.20 |
| opt-30b | 9.60 |
| llama2-7b | 5.65 |
| llama2-70b | 3.40 |

Results of perplexity evaluation can be reproduced with slight random error.


### Zero-shot Evaluation

For zero-shot evaluation in Table 3, you can run the following script to evaluation all models.
```bash
sh scripts/batch_evaluation_lm_eval.sh
```
Or you can use
```bash
sh scripts/evaluation_lm_eval_opt_30b.sh        # About 36 hours, 70GB VRAM
sh scripts/evaluation_lm_eval_llama2_70b.sh     # About 90 hours, 160GB VRAM
```
to evaluate a single model.

Remember to set the `device` variable in the script to the GPU you want to use.

The results will be saved in `results/lm_eval_results.txt`.

The zero-shot results under our configuration are listed in the following table.
| Model | Arc-e | Hella. | Piqa | Wino. | Avg. |
| --- | --- | --- | --- | --- | --- |
| opt-30b | 64.86 | 72.08 | 78.07 | 68.03 | 70.76 |
| llama2-70b | 82.11 | 83.79 | 82.59 | 78.61 | 81.78 |

Results of zero-shot evaluation can be reproduced with slight random error.