# Artifact Evalution for MICRO58 AxCore

This repository contains the source code for reproducing the experiments in the paper "AxCore: A Quantization-Aware Approximate GEMM Unit For LLM Inference" at MICRO'25.

`Hardware/AxCore` contains the hardware design of AxCore.

`Software/AxCore` contains the AxCore framework with PyTorch.

`Software/axcore_simulator` contains the performance and energy evaluation of AxCore. 


## Project Structure
```
AxCore_artifact/
├── README.md
├── Hardware/
│   ├── AreaTest/
│   │   ├── PE/
│   │   ├── Figure/
│   ├── FunctionalTest/
│   │   ├── AxCore/
├── Software/
│   ├── AxCore/
│   │   ├── README.md
│   │   ├── approximation_computation/
│   │   ├── evaluation/
│   │   ├── scripts/
│   ├── axcore_simulator/
│   │   ├── run_axcore.py
│   │   ├── EnergyAll.py
│   │   ├── scripts/
│   │   ├── params/
│   │   ├── AxCore/
```