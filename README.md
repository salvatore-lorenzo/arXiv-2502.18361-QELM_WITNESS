# QELM Witness Analysis

## Overview
This repository contains code and data for analyzing experimental quantum entanglement witness measurements. The project includes tools for loading experimental data, processing it, and generating various plots related to witness operators, Pauli observables, shadow tomography, and singular values.

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

## Installation
To use this project, install the required dependencies:
```bash
pip install numpy pandas matplotlib
```

## Usage
1. Open `experiment_analysis.ipynb` in Jupyter Notebook.
2. Run the cells sequentially to load data and generate plots.
3. Modify parameters in `load_data.py` and `plot_defs.py` to customize analysis.

## Analysis Features
- **Witness Plots**: Visualization of entanglement witness measurements.
- **Pauli Plots**: Analysis of Pauli observables in experimental data.
- **Shadow Tomography**: Reconstruction of quantum states using classical shadows.
- **Singular Value Decomposition (SVD)**: Examination of singular values from experimental data.

