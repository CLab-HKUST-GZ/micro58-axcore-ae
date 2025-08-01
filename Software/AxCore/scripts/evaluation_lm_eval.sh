device=1

model="facebook/opt-2.7b"
# model="facebook/opt-6.7b"
# model="facebook/opt-13b"
# model="facebook/opt-30b"
# model="facebook/opt-66b"

# model="/local-ssd/jiaxiang/models/llama/llama2_7b_hf"
# model="/local-ssd/jiaxiang/models/llama/llama2_7b_hf"

torch_dtype="float16"
# torch_dtype="bfloat16"

quant_method="RTN"

approx_kernel="baseline"

task_group="group1"

CUDA_VISIBLE_DEVICES=$device python evaluation/lmeval/lmeval.py \
    --model ${model} \
    --approx_kernel ${approx_kernel} \
    --dtype ${torch_dtype} \
    --quant_method ${quant_method} \
    --task_group ${task_group} \
    # --waquant \
    # --linear_approx \
    # --w_bit 4 \
    # --w_group_size 128