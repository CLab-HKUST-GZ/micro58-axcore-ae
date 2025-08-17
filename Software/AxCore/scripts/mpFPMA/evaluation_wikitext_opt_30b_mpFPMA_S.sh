device=0

model="facebook/opt-30b"

torch_dtype="float16"

quant_method="RTN"

approx_kernel="baseline"

CUDA_VISIBLE_DEVICES=$device python evaluation/wikitext/evaluate_hf.py \
    --model ${model} \
    --approx_kernel ${approx_kernel} \
    --dtype ${torch_dtype} \
    --quant_method ${quant_method} \
    --ablation \
    --optimization 0 \
    --wquant \
    --w_group_size 128 \
    --w_bit 4 \
    --linear_approx \


    
    
