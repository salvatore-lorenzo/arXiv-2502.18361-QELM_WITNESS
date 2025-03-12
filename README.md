# Quantum extreme learning machines for photonic entanglement witnessing

## Overview
This repository contains code and data for reproduce the results in [arXiv:2502.18361](https://arxiv.org/abs/2502.18361).

## Repository Structure
```
├── datajson/               # Contains JSON data related to the experiment
├── experimental_data/      # Raw and processed experimental data
├── figures/                # Generated figures from analysis
├── definitions.py          # Definitions and helper functions
├── experiment_analysis.ipynb # Jupyter notebook for analysis
├── fig_4_right.ipynb       # Specific analysis for Figure 4 (right panel)
├── isometry.py             # Code for isometry calculations
├── json_convert.ipynb      # Conversion of data to/from JSON
├── load_data.py            # Functions for loading and preprocessing data
├── plot_defs.py            # Plotting functions
```

## Usage
1. Open `experiment_analysis.ipynb` in Jupyter Notebook.
2. Select the experimental dataset with the variable "select" and the training dataset with the variable "training".
3. Run the cells sequentially to load data and generate plots.


## Analysis Features
- **Witness Plots**: Visualization of entanglement witness results.
- **Pauli Plots**: Analysis of Pauli observables in experimental data.
- **Shadow MSE**: Visualization of entanglement witness results obtained with Classical Shadow.
- **Singular Value Decomposition (SVD)**: Examination of singular values from experimental data.


