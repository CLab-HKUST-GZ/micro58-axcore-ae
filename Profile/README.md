# AxCore Profile

This directory contains the profiling tools and scripts to reproduce **Figure 2** from the AxCore paper, which analyzes the computational workload distribution in transformer models.

## Overview

The profiling system measures and visualizes the percentage of linear operations and attention computations across different transformer architectures, providing insights into computational bottlenecks and optimization opportunities.

## Quick Start

### Step 1: Calculate Workload Distribution
Run the profiling script to analyze the computational workload:
```bash
python fig2_cal.py
```
This will analyze the transformer model and print the linear and attention operation percentages to the terminal.

### Step 2: Generate Visualization
Create the visualization chart:
```bash
python fig2.py
```
This generates `figure2.pdf` in the current directory, showing the workload distribution analysis.

## Output

- **Terminal Output**: Detailed breakdown of linear and attention percentages
- **Generated File**: `figure2.pdf` - Visual representation of the workload analysis
