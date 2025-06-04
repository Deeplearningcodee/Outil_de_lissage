# Outil de Lissage

This project contains various scripts used to compute and optimise supply orders.
The main workflow merges multiple prediction files and applies smoothing
algorithms before generating the final ordering plan.

## Installation

1. Ensure **Python 3.10** or newer is available on your system.
2. Install the Python dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Some packages such as `gurobipy` may require a valid licence.

## Running the tool

Execute the main script from the repository root:

```bash
python main.py
```

The script will produce several CSV/Excel files including
`merged_predictions.csv` and `PDC_Sim_Optimized_Python.xlsx`.

