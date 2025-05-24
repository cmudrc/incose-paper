# incose-paper
PiperABM Paper Submission to INCOSE: "Agent-Based Modeling for the Evaluation of Community Resilience In Silico"

## Overview

This repository contains all code, data links, and analysis scripts used in the PiperABM paper submitted to INCOSE. The project is structured to allow users to run full simulations, explore existing results, and reproduce the analysis presented in the paper.

## Repository Structure
```
├── analysis/            # Analysis scripts for processing and visualizing results
├── assets/              # Extra figures and assets used in the paper
├── main/                # Main simulation code and results
│   ├── main.py          # Entry point for running full simulations
│   └── result/          # Simulation outputs: each subfolder represents a scenario
│       └── setups.csv   # Metadata describing each scenario
├── test/                # Test runs for quick checks
│   ├── single-run/      # Single-scenario test (smaller, faster)
│   └── multi-run/       # Parallel multi-scenario test (small-scale)
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Requirements

* Python 3.10 or higher
* Install dependencies:
```
pip install -r requirements.txt
```

## Running Full Simulations

* To run the main simulation suite, run `main/main.py`
```
cd main
python main.py
```
* Results will be saved under `main/result/<scenario_name>/`.
* Each folder name corresponds to a unique scenario.
* Scenario details (parameters, seeds, etc.) are listed in `main/result/setups.csv`.

## Accessing Results

* __Local:__ After running `main/main.py`, inspect the `main/result/` directory.
* __Remote:__ Precomputed results are hosted on Hugging Face: 
[https://huggingface.co/datasets/cmudrc/incose-paper-data](https://huggingface.co/datasets/cmudrc/incose-paper-data)

## Tests

Two lightweight test suites are provided to verify functionality before running full simulations:

* __single-run:__ runs one scenario end-to-end (fast sanity check)
* __multi-run:__ runs multiple scenarios in parallel on a smaller scale

## Assets

Supplementary figures and assets (e.g., diagrams, raw images) are stored in `assets/`.

*For questions or issues, please open an issue on GitHub or contact the authors.*