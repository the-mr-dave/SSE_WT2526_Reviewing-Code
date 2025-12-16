
# Heat Equation Solvers (1D & 2D)

This repository contains simple explicit finite-difference solvers for the 1D and 2D heat equation plus unit tests.  
It is intended for teaching and learning basic numerical methods for PDEs.

## Project structure

```text
.
├─ heat_solver.py        # 1D heat equation solver
├─ heat_solver_2d.py     # 2D heat equation solver
├─ test_heat_solvers.py  # Unit tests
├─ config_1d.json        # Example config for 1D
├─ config_2d.json        # Example config for 2D
└─ requirements.txt      # Python dependencies
```

## Installation

It is recommended to use a virtual environment:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Minimal `requirements.txt`:

```text
numpy
matplotlib
```

## Usage

### 1D solver

Run:

```bash
python heat_solver.py --config config_1d.json
```

Example `config_1d.json`:

```json
{
  "alpha": 1.0,
  "L": 1.0,
  "nx": 51,
  "dt": 0.00005,
  "t_final": 0.05,
  "output_dir": "results_1d",
  "initial_condition": {
    "type": "gaussian",
    "center": 0.5,
    "sigma": 0.1
  }
}
```

Output in `output_dir`: `results.npz`, `final_profile.csv`, `profiles.png`.

### 2D solver

Run:

```bash
python heat_solver_2d.py --config config_2d.json
```

Example `config_2d.json`:

```json
{
  "alpha": 1.0,
  "Lx": 1.0,
  "Ly": 1.0,
  "nx": 51,
  "ny": 51,
  "dt": 0.00005,
  "t_final": 0.05,
  "output_dir": "results_2d",
  "initial_condition": {
    "type": "gaussian",
    "center_x": 0.5,
    "center_y": 0.5,
    "sigma_x": 0.1,
    "sigma_y": 0.1
  }
}
```

Output in `output_dir`: `results_2d.npz`, `final_profile_2d.csv`, `final_state_2d.png`.

## Tests

Run from the project root:

```bash
python -m unittest test_heat_solvers.py
```

## Academic use

This code is for educational purposes.  
When using it in reports or assignments, please cite appropriately and follow your institution’s academic integrity guidelines.
