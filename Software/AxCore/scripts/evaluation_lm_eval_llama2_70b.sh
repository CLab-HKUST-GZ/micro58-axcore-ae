device=0

model="meta-llama/Llama-2-70b-hf" # or /Your/Llama2/Path/

torch_dtype="float16"

quant_method="RTN"

approx_kernel="baseline"

task_group="group1"

CUDA_VISIBLE_DEVICES=$device python evaluation/lmeval/lmeval.py \
    --model ${model} \
    --approx_kernel ${approx_kernel} \
    --dtype ${torch_dtype} \
    --quant_method ${quant_method} \
    --task_group ${task_group} \
    --waquant \
    --linear_approx \
    --w_bit 4 \
    --w_group_size 64