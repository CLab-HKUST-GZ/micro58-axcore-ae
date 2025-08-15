#!/bin/bash

CHUNK_SIZE=256

# # Llama-2 - Linear
# for nsamples in 128;do
#   for seqlen in 2048;do
#     for size in 7b 70b;do
#       for bits in 4 8;do
#         case ${size} in 
#           7b)
#             if [ ${bits} = 4 ]; then
#               decomp=14
#             elif [ ${bits} = 8 ]; then
#               decomp=8
#             fi
#           ;;
#           13b)
#             if [ ${bits} = 4 ]; then
#               decomp=16
#             elif [ ${bits} = 8 ]; then
#               decomp=14
#             fi
#           ;;
#           70b)
#             if [ ${bits} = 4 ]; then
#               decomp=20
#             elif [ ${bits} = 8 ]; then
#               decomp=16
#             fi
#           ;;
#         esac
#         echo calibrating llama-2-${size} ${bits}bit chunk size of ${CHUNK_SIZE}
#         echo linear only
#         python run_calibration.py \
#           --model-name "${LLAMA2_PATH}/llama2_${size}_hf" \
#           --target "bias" \
#           --output-path "llama-2-bias/${seqlen}_${size}_${nsamples}_${bits}bit_${decomp}decomp.pt" \
#           --dataset-path "../../data/val.jsonl.zst" \
#           --num-samples ${nsamples} \
#           --seq-len ${seqlen} \
#           --q_bits ${bits} \
#           --decomp_factor ${decomp} \
#           --chunk_size ${CHUNK_SIZE}
#       done
#     done
#   done
# done

# Llama-2 - Linear & MHA
for nsamples in 128;do
  for seqlen in 2048;do
    for size in 7b;do
      for bits in 4 8;do
        case ${size} in 
          7b)
            if [ ${bits} = 4 ]; then
              decomp=14
            elif [ ${bits} = 8 ]; then
              decomp=8
            fi
          ;;
          13b)
            if [ ${bits} = 4 ]; then
              decomp=16
            elif [ ${bits} = 8 ]; then
              decomp=14
            fi
          ;;
          70b)
            if [ ${bits} = 4 ]; then
              decomp=20
            elif [ ${bits} = 8 ]; then
              decomp=16
            fi
          ;;
        esac
        echo calibrating llama-2-${size} ${bits}bit chunk size of ${CHUNK_SIZE}
        echo linear + mha
        python run_calibration.py \
          --model-name "${LLAMA2_PATH}/llama2_${size}_hf" \
          --target "bias" \
          --output-path "llama-2-bias/${seqlen}_${size}_${nsamples}_${bits}bit_${decomp}decomp_mha.pt" \
          --dataset-path "../../data/val.jsonl.zst" \
          --num-samples ${nsamples} \
          --seq-len ${seqlen} \
          --q_bits ${bits} \
          --decomp_factor ${decomp} \
          --chunk_size ${CHUNK_SIZE} \
          --quant_mha
      done
    done
  done
done


# # Llama-1 - Linear
# for nsamples in 128;do
#   for seqlen in 2048;do
#     for size in 7b 13b;do
#       for bits in 4 8;do
#         decomp=14
#         echo calibrating llama-1-${size} ${bits}bit chunk size of ${CHUNK_SIZE}
#         echo linear only

#         case ${size} in 
#           7b)
#             model_name="baffo32/decapoda-research-llama-7B-hf"
#           ;;
#           13b)
#             model_name="JG22/decapoda-research-llama-13b"
#           ;;
#         esac
#         python run_calibration.py \
#           --model-name ${model_name} \
#           --target "bias" \
#           --output-path "llama-1-bias/${seqlen}_${size}_${nsamples}_${bits}bit_${decomp}decomp.pt" \
#           --dataset-path "../../data/val.jsonl.zst" \
#           --num-samples ${nsamples} \
#           --seq-len ${seqlen} \
#           --q_bits ${bits} \
#           --decomp_factor ${decomp} \
#           --chunk_size ${CHUNK_SIZE}
#       done
#     done
#   done
# done
