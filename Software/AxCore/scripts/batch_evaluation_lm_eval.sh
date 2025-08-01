#!/bin/bash
opt_models=(
    "facebook/opt-30b"
)
llama_models=(
    "meta-llama/Llama-2-70b-hf"
)

torch_dtype="float16"
quant_method="RTN"
approx_kernel="baseline"
task_group="group1"
device="0"

echo "Starting batch evaluation..."
echo "OPT models will use w_group_size=128"
echo "LLaMA models will use w_group_size=64"
echo "Results will be saved to results.txt"
echo ""

# Loop through OPT models with w_group_size=128
for model in "${opt_models[@]}"; do
    w_group_size=128
    echo "Running lm eval: device=$device, model=$model, w_group_size=$w_group_size"
    
    CUDA_VISIBLE_DEVICES=$device python evaluation/wikitext/evaluate_hf.py \
        --model ${model} \
        --approx_kernel ${approx_kernel} \
        --dtype ${torch_dtype} \
        --quant_method ${quant_method} \
        --task_group ${task_group} \
        --waquant \
        --w_group_size ${w_group_size} \
        --w_bit 4 \
        --linear_approx
    
    echo "Completed lm eval: model=$model, w_group_size=$w_group_size"
    echo "----------------------------------------"
done

# Loop through LLaMA models with w_group_size=64
for model in "${llama_models[@]}"; do
    w_group_size=64
    echo "Running lm eval: device=$device, model=$model, w_group_size=$w_group_size"
    
    CUDA_VISIBLE_DEVICES=$device python evaluation/wikitext/evaluate_hf.py \
        --model ${model} \
        --approx_kernel ${approx_kernel} \
        --dtype ${torch_dtype} \
        --quant_method ${quant_method} \
        --task_group ${task_group} \
        --waquant \
        --w_group_size ${w_group_size} \
        --w_bit 4 \
        --linear_approx
    
    echo "Completed lm eval: model=$model, w_group_size=$w_group_size"
    echo "----------------------------------------"
done


echo "Batch evaluation completed!"
echo "Check results.txt for all results"
