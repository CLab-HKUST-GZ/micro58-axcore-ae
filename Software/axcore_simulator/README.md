# AxCore Simulator

This repository contains the code for the AxCore simulator based on DNNWeaver, BitFusion and ANT.

## Prerequisite

+ Ubuntu 22.04.5 LTS
+ Conda 25.1.1
+ Python 3.9
+ gcc 11.4.0

## Getting Started

```shell
$ # Environment.
$ conda create -n axcore_sim python=3.9
$ conda activate axcore_sim  
$ pip install -r  requirements.txt
$ # Cacti for the memory simulation.
$ git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti/
$ make -C ./bitfusion/sram/cacti/
$ cp -r ./bitfusion/sram/cacti ./AxCore/sram/
$ # Run AxCore simulation.
$ sh scripts/fig_17.sh  # About 10 minutes
```

## Evaluation

The scripts/fig_17.sh script automates the reproduction of Figure 17 from the paper. It first collects performance and energy statistics into `./results/axcore_res.csv` and then uses this data to generate the final plot at `./results/fig_17.pdf`.