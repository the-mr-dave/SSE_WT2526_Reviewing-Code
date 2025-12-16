"""
Minimal 1D heat equation solver with JSON config and file output.

This module solves the 1D heat equation

    u_t = alpha * u_xx,  x in (0, L), t > 0,

with homogeneous Dirichlet boundary conditions u(0, t) = u(L, t) = 0.
It also provides helper functions to build the initial condition from a
configuration dictionary and to save results to disk.

The configuration is normally read from a JSON file; see :func:`load_config`.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def solve_heat_equation_1d(alpha=1.0,
                           L=1.0,
                           nx=51,
                           dt=1e-4,
                           t_final=0.1,
                           initial_condition=None):
    """Solve the 1D heat equation with an explicit finite difference scheme.

    The equation is

    .. math::

       u_t = \\alpha u_{xx}, \\quad x \\in (0,L), \\ t > 0,

    with boundary conditions

    .. math::

       u(0,t) = 0, \\quad u(L,t) = 0.

    A uniform spatial grid is used on :math:`[0,L]` with ``nx`` points and
    time-stepping with step size ``dt`` up to ``t_final``. The method is
    Forward Time, Centered Space (FTCS).

    Parameters
    ----------
    alpha : float, optional
        Thermal diffusivity :math:`\\alpha`.
    L : float, optional
        Length of the spatial domain :math:`[0,L]`.
    nx : int, optional
        Number of spatial grid points.
    dt : float, optional
        Time step size.
    t_final : float, optional
        Final simulation time.
    initial_condition : callable or None, optional
        Function ``ic(x)`` defining :math:`u(x,0)`. If ``None``, the default
        initial condition :math:`u(x,0) = \\sin(\\pi x / L)` is used.

    Returns
    -------
    x : numpy.ndarray
        Spatial grid of shape ``(nx,)``.
    u : numpy.ndarray
        Temperature at final time, shape ``(nx,)``.
    u_history : numpy.ndarray
        History of the temperature, shape ``(nt+1, nx)``, where ``nt`` is the
        number of time steps.
    t : numpy.ndarray
        Time grid of shape ``(nt+1,)``.

    Raises
    ------
    ValueError
        If the stability condition for the explicit scheme,
        :math:`\\alpha dt / dx^2 \\leq 0.5`, is violated.
    """
    # Create spatial grid
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]

    # Determine number of time steps
    nt = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, nt * dt, nt + 1)

    # Stability check for explicit (FTCS) scheme
    stability_number = alpha * dt / dx**2
    if stability_number > 0.5:
        raise ValueError(
            f"Unstable time step: alpha*dt/dx^2 = {stability_number:.3f} > 0.5. "
            "Decrease dt or increase nx."
        )

    # Set initial condition
    if initial_condition is None:
        # Default: sine mode satisfying homogeneous Dirichlet BCs
        u0 = np.sin(np.pi * x / L)
    else:
        u0 = initial_condition(x)

    # Allocate arrays for solution
    u = u0.copy()
    u_new = np.zeros_like(u)
    u_history = np.zeros((nt + 1, nx))
    u_history[0, :] = u0

    # Main time-stepping loop
    for n in range(nt):
        # Interior points: apply FTCS update
        u_new[1:-1] = (
            u[1:-1]
            + stability_number * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        )

        # Enforce Dirichlet boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0

        # Update solution and store in history
        u[:] = u_new[:]
        u_history[n + 1, :] = u

    return x, u, u_history, t


def load_config(config_path: Path) -> dict:
    """Load configuration parameters from a JSON file.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Dictionary containing configuration parameters.

    Raises
    ------
    ValueError
        If the file does not have a ``.json`` extension.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if config_path.suffix.lower() != ".json":
        raise ValueError("Only JSON config files are supported (use .json).")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_initial_condition(config: dict, L: float):
    """Construct an initial condition function from configuration.

    The function returns a callable ``ic(x)`` which can be passed as
    ``initial_condition`` to :func:`solve_heat_equation_1d`.

    Supported types in ``config["initial_condition"]["type"]``:

    * ``"sine"``:
      :math:`u(x,0) = \\sin(\\pi x / L)`.
    * ``"gaussian"``:
      :math:`u(x,0) = \\exp(- (x - \\text{center})^2 / (2 \\sigma^2))`.

    Parameters
    ----------
    config : dict
        Configuration dictionary, typically loaded via :func:`load_config`.
    L : float
        Domain length :math:`L`.

    Returns
    -------
    callable
        Function ``ic(x)`` defining the initial condition.

    Raises
    ------
    ValueError
        If an unknown initial condition type is requested.
    """
    ic_cfg = config.get("initial_condition", {})
    ic_type = ic_cfg.get("type", "sine").lower()

    if ic_type == "sine":
        def ic(x):
            """Sine-mode initial condition."""
            return np.sin(np.pi * x / L)

        return ic

    elif ic_type == "gaussian":
        center = float(ic_cfg.get("center", 0.5 * L))
        sigma = float(ic_cfg.get("sigma", 0.1 * L))

        def ic(x):
            """Gaussian initial condition centered at ``center``."""
            return np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))

        return ic

    else:
        raise ValueError(f"Unknown initial_condition type: {ic_type}")


def save_results(output_dir: Path,
                 x: np.ndarray,
                 t: np.ndarray,
                 u_history: np.ndarray):
    """Save numerical results to disk.

    The following files are created:

    * ``results.npz``: compressed NumPy archive containing arrays
      ``x``, ``t`` and ``u_history``.
    * ``final_profile.csv``: CSV file with two columns,
      the spatial coordinate and the final temperature.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where the files are written. Will be created if missing.
    x : numpy.ndarray
        Spatial grid of shape ``(nx,)``.
    t : numpy.ndarray
        Time grid of shape ``(nt+1,)``.
    u_history : numpy.ndarray
        Solution history of shape ``(nt+1, nx)``.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the ``results.npz`` and ``final_profile.csv`` files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all data in compressed NumPy format
    npz_path = output_dir / "results.npz"
    np.savez_compressed(npz_path, x=x, t=t, u_history=u_history)

    # Save final profile as CSV
    csv_path = output_dir / "final_profile.csv"
    u_final = u_history[-1, :]
    data = np.column_stack((x, u_final))
    header = "x,u_final"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    return npz_path, csv_path


def plot_results(output_dir: Path,
                 x: np.ndarray,
                 t: np.ndarray,
                 u_history: np.ndarray):
    """Plot initial and final temperature profiles as a PNG image.

    The plot is saved in the output directory as ``profiles.png``.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where the plot will be written.
    x : numpy.ndarray
        Spatial grid of shape ``(nx,)``.
    t : numpy.ndarray
        Time grid of shape ``(nt+1,)``.
    u_history : numpy.ndarray
        Solution history of shape ``(nt+1, nx)``.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(x, u_history[0, :], label=f"t = {t[0]:.3e}")
    plt.plot(x, u_history[-1, :], label=f"t = {t[-1]:.3e}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("1D Heat Equation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = output_dir / "profiles.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    """Command-line entry point for the 1D heat equation solver.

    This function parses the command-line arguments, loads the configuration
    from a JSON file, runs the solver and saves the results and a plot.

    The script is intended to be run as::

        python heat_solver.py --config config.json
    """
    parser = argparse.ArgumentParser(
        description="1D heat equation solver with JSON config and file output."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    # Read parameters from config with defaults
    alpha = float(config.get("alpha", 1.0))
    L = float(config.get("L", 1.0))
    nx = int(config.get("nx", 51))
    dt = float(config.get("dt", 1e-4))
    t_final = float(config.get("t_final", 0.1))

    # Build initial condition based on configuration
    initial_condition = build_initial_condition(config, L)

    # Output directory for result files
    output_dir = Path(config.get("output_dir", "output"))

    # Solve the PDE
    x, u_final, u_history, t = solve_heat_equation_1d(
        alpha=alpha,
        L=L,
        nx=nx,
        dt=dt,
        t_final=t_final,
        initial_condition=initial_condition,
    )

    # Save results and plot
    npz_path, csv_path = save_results(output_dir, x, t, u_history)
    plot_path = plot_results(output_dir, x, t, u_history)

    print("Simulation finished.")
    print(f"Results (x, t, u_history) saved to: {npz_path}")
    print(f"Final profile CSV saved to:       {csv_path}")
    print(f"Plot saved to:                   {plot_path}")


if __name__ == "__main__":
    main()