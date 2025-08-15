#!/bin/bash

start_time=$(date +%s)

# Clear previous results file to avoid appending to old data
mkdir -p results
> results/axcore_res.csv

python run_axcore.py --synth_csv ./params/systolic_array_synth_W4-FP16.csv 
python run_axcore.py --synth_csv ./params/systolic_array_synth_W4-BF16.csv 
python run_axcore.py --synth_csv ./params/systolic_array_synth_W4-FP32.csv 
python run_axcore.py --synth_csv ./params/systolic_array_synth_W8-FP16.csv 
python run_axcore.py --synth_csv ./params/systolic_array_synth_W8-BF16.csv 
python run_axcore.py --synth_csv ./params/systolic_array_synth_W8-FP32.csv

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Total execution time: $duration seconds."

python EnergyAll.py
