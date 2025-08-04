# Artifact Evalution for MICRO58 AxCore

This repository contains the source code for reproducing the experiments in the paper "AxCore: A Quantization-Aware Approximate GEMM Unit For LLM Inference" at MICRO'25.

[`Hardware/`](./Hardware) contains the hardware design of AxCore.

[`Software/AxCore`](./Software/AxCore) contains the AxCore framework with PyTorch. (reproduces Table 2 and Table 3)

[`Software/axcore_simulator`](./Software/axcore_simulator) contains the performance and energy evaluation of AxCore. (reproduces Figure 17)


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
