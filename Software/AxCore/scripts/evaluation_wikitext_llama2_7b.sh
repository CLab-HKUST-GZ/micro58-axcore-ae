device=0

model="meta-llama/Llama-2-7b-hf" # or /Your/Llama2/Path/

torch_dtype="float16"

quant_method="RTN"

approx_kernel="baseline"

CUDA_VISIBLE_DEVICES=$device python evaluation/wikitext/evaluate_hf.py \
    --model ${model} \
    --approx_kernel ${approx_kernel} \
    --dtype ${torch_dtype} \
    --quant_method ${quant_method} \
    --waquant \
    --w_group_size 64 \
    --w_bit 4 \
    --linear_approx \


    
    
