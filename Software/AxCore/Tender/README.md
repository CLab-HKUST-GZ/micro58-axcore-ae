# Tender

## Directory Structure

- calibration: Calibration scripts for generating scale factor, channel bias, and group index.
  - opt: Calibration script for OPT.
  - llama: Calibration script for Llama-2 / LLaMA
- models: Tender implementation.
- scripts: Scripts for running the perplexity and accuracy evaluation.

## Setup

### Prerequisite

Fetch llama-2 from [here](https://llama.meta.com/llama-downloads). You may also need to convert the model to huggingface format using the `convert_llama_weight_to_hf.py` in `transformers/src/transformers/models/llama`.

```sh
conda create -n tender python=3.9
conda activate tender
conda install ninja
pip install -r requirements.txt
git clone -b v4.35-release https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd .. && bash setup.sh
```

NOTE: `setup.sh` renames the original model code from `modeling_xx.py` to `modeling_xx_orig.py` (e.g., `modeling_opt.py` -> `modeling_opt_orig.py`) in the transformers library.

### Calibration

Tender requires offline calibration to determine the scale factors, biases, and channel indices. To calibrate the models, run the following command:

```sh
cd calibration/opt
bash generate_bias.sh
bash generate_scale_factor.sh
```

The above code generates channel bias, scale factor, and channel group index for each row chunk.

## Perplexity Evaluation

To reproduce Tender's results in Tables 2:

```sh
cd scripts
python table_opt.py
```
The results will be printed in the terminal.
