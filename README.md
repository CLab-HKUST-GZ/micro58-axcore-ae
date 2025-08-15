# Artifact Evalution for MICRO58 AxCore

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16735520.svg)](https://doi.org/10.5281/zenodo.16735520)

This repository contains the source code for reproducing the experiments in the paper "AxCore: A Quantization-Aware Approximate GEMM Unit For LLM Inference" at MICRO'25.

[`Hardware/AxCore`](./Hardware/AxCore) contains the hardware design of AxCore.

[`Software/AxCore`](./Software/AxCore) contains the AxCore framework with PyTorch. (reproduces Table 2 and Table 3)

[`Software/axcore_simulator`](./Software/axcore_simulator) contains the performance and energy evaluation of AxCore. (reproduces Figure 17)

[`Profile`](./Profile) contains the gemm operations percentage of OPT and LLaMA models across various sequence lengths. (reproduces Figure 2)

## Project Structure
```
AxCore_artifact/
├── README.md
├── Hardware/
│   ├── AxCore/
│   │   ├── README.md
│   │   ├── project/
│   │   ├── hw/
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
│   │   ├── README.md
│   ├── Profile/
│   │   ├── fig2_cal.py
│   │   ├── fig2.py
│   │   ├── README.md
```
