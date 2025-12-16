"""
Minimal 2D heat equation solver with JSON config and file output.

This module solves the 2D heat equation

    u_t = alpha (u_xx + u_yy),

on a rectangular domain :math:`(0, L_x) \\times (0, L_y)` with homogeneous
Dirichlet boundary conditions :math:`u = 0` on the entire boundary.

It also provides helper functions to build the initial condition, save
results to disk and generate a heatmap of the final state.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def solve_heat_equation_2d(alpha=1.0,
                           Lx=1.0,
                           Ly=1.0,
                           nx=51,
                           ny=51,
                           dt=1e-4,
                           t_final=0.1,
                           initial_condition=None):
    """Solve the 2D heat equation with an explicit finite difference scheme.

    The equation is

    .. math::

       u_t = \\alpha (u_{xx} + u_{yy}),

    inside :math:`(0, L_x) \\times (0, L_y)` with homogeneous Dirichlet
    boundary conditions :math:`u = 0` on the entire boundary.

    A uniform grid is used in both directions with ``nx`` and ``ny``
    grid points in :math:`x` and :math:`y`, respectively. The time-stepping
    scheme is an explicit FTCS method.

    Parameters
    ----------
    alpha : float, optional
        Thermal diffusivity :math:`\\alpha`.
    Lx : float, optional
        Domain length in :math:`x`.
    Ly : float, optional
        Domain length in :math:`y`.
    nx : int, optional
        Number of grid points in :math:`x`.
    ny : int, optional
        Number of grid points in :math:`y`.
    dt : float, optional
        Time step size.
    t_final : float, optional
        Final simulation time.
    initial_condition : callable or None, optional
        Function ``ic(X, Y)`` defining :math:`u(x, y, 0)`. Here ``X`` and ``Y``
        are the 2D meshgrids as returned by ``numpy.meshgrid(x, y)``.
        If ``None``, a default Gaussian bump in the center is used.

    Returns
    -------
    x : numpy.ndarray
        1D grid in :math:`x` of shape ``(nx,)``.
    y : numpy.ndarray
        1D grid in :math:`y` of shape ``(ny,)``.
    t : numpy.ndarray
        Time grid of shape ``(nt+1,)``.
    u_history : numpy.ndarray
        Solution history of shape ``(nt+1, ny, nx)``.

    Raises
    ------
    ValueError
        If the stability condition

        .. math::

           \\alpha dt (1/dx^2 + 1/dy^2) \\le 0.5

        is violated.
    """
    # Create spatial grids
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Determine number of time steps
    nt = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, nt * dt, nt + 1)

    # Stability check for explicit 2D FTCS scheme
    stability_number = alpha * dt * (1.0 / dx**2 + 1.0 / dy**2)
    if stability_number > 0.5:
        raise ValueError(
            "Unstable time step: alpha*dt*(1/dx^2 + 1/dy^2) = "
            f"{stability_number:.3f} > 0.5. Decrease dt or increase nx, ny."
        )

    # Create meshgrid for initial condition
    X, Y = np.meshgrid(x, y)  # shapes (ny, nx)

    # Set initial condition
    if initial_condition is None:
        # Default: smooth Gaussian bump in the center
        xc = 0.5 * Lx
        yc = 0.5 * Ly
        sigma_x = 0.1 * Lx
        sigma_y = 0.1 * Ly
        u0 = np.exp(-((X - xc) ** 2) / (2.0 * sigma_x ** 2)
                    - ((Y - yc) ** 2) / (2.0 * sigma_y ** 2))
    else:
        u0 = initial_condition(X, Y)

    # Allocate arrays for solution
    u = u0.copy()
    u_new = np.zeros_like(u)
    u_history = np.zeros((nt + 1, ny, nx))
    u_history[0, :, :] = u0

    # Precompute constants for update
    cx = alpha * dt / dx**2
    cy = alpha * dt / dy**2

    # Main time-stepping loop
    for n in range(nt):
        # Interior points: 2D FTCS stencil
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + cx * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2])
            + cy * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        )

        # Enforce Dirichlet boundary conditions: u = 0
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0

        # Update and store in history
        u[:, :] = u_new[:, :]
        u_history[n + 1, :, :] = u

    return x, y, t, u_history


def load_config(config_path: Path) -> dict:
    """Load configuration parameters from a JSON file.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Configuration dictionary.

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


def build_initial_condition(config: dict, Lx: float, Ly: float):
    """Construct a 2D initial condition from configuration.

    Returns a callable ``ic(X, Y)`` suitable for passing to
    :func:`solve_heat_equation_2d`.

    Supported types in ``config["initial_condition"]["type"]``:

    * ``"gaussian"``: 2D Gaussian

      .. math::

         u(x, y, 0) =
         \\exp\\Big(-\\frac{(x-c_x)^2}{2 \\sigma_x^2}
                    -\\frac{(y-c_y)^2}{2 \\sigma_y^2}\\Big).

    * ``"sine"``: product of sine functions vanishing at the boundary

      .. math::

         u(x,y,0) =
         \\sin(\\pi x / L_x) \\sin(\\pi y / L_y).

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    Lx : float
        Domain length in :math:`x`.
    Ly : float
        Domain length in :math:`y`.

    Returns
    -------
    callable
        Function ``ic(X, Y)`` defining the initial condition.

    Raises
    ------
    ValueError
        If an unknown initial condition type is requested.
    """
    ic_cfg = config.get("initial_condition", {})
    ic_type = ic_cfg.get("type", "gaussian").lower()

    if ic_type == "gaussian":
        cx = float(ic_cfg.get("center_x", 0.5 * Lx))
        cy = float(ic_cfg.get("center_y", 0.5 * Ly))
        sx = float(ic_cfg.get("sigma_x", 0.1 * Lx))
        sy = float(ic_cfg.get("sigma_y", 0.1 * Ly))

        def ic(X, Y):
            """2D Gaussian initial condition."""
            return np.exp(-((X - cx) ** 2) / (2.0 * sx**2)
                          - ((Y - cy) ** 2) / (2.0 * sy**2))

        return ic

    elif ic_type == "sine":
        def ic(X, Y):
            """Product of sine functions vanishing at all boundaries."""
            return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

        return ic

    else:
        raise ValueError(f"Unknown initial_condition type: {ic_type}")


def save_results(output_dir: Path,
                 x: np.ndarray,
                 y: np.ndarray,
                 t: np.ndarray,
                 u_history: np.ndarray):
    """Save 2D numerical results to disk.

    The following files are created:

    * ``results_2d.npz``: compressed NumPy archive containing ``x``, ``y``,
      ``t`` and ``u_history``.
    * ``final_profile_2d.csv``: CSV file with three columns, the coordinates
      :math:`(x,y)` and the final temperature.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where the files will be written.
    x : numpy.ndarray
        1D grid in :math:`x`, shape ``(nx,)``.
    y : numpy.ndarray
        1D grid in :math:`y`, shape ``(ny,)``.
    t : numpy.ndarray
        Time grid, shape ``(nt+1,)``.
    u_history : numpy.ndarray
        Solution history, shape ``(nt+1, ny, nx)``.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the ``results_2d.npz`` and ``final_profile_2d.csv`` files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all data in compressed NumPy format
    npz_path = output_dir / "results_2d.npz"
    np.savez_compressed(npz_path, x=x, y=y, t=t, u_history=u_history)

    # Save final profile as CSV (flattened)
    csv_path = output_dir / "final_profile_2d.csv"
    ny, nx = u_history.shape[1], u_history.shape[2]
    X, Y = np.meshgrid(x, y)
    u_final = u_history[-1, :, :]

    data = np.column_stack((X.ravel(), Y.ravel(), u_final.ravel()))
    header = "x,y,u_final"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    return npz_path, csv_path


def plot_final_state(output_dir: Path,
                     x: np.ndarray,
                     y: np.ndarray,
                     u_history: np.ndarray):
    """Plot the final temperature field as a 2D heatmap.

    The plot is saved in the output directory as ``final_state_2d.png``.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where the plot will be written.
    x : numpy.ndarray
        1D grid in :math:`x`, shape ``(nx,)``.
    y : numpy.ndarray
        1D grid in :math:`y`, shape ``(ny,)``.
    u_history : numpy.ndarray
        Solution history, shape ``(nt+1, ny, nx)``.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract final time step
    u_final = u_history[-1, :, :]

    plt.figure(figsize=(6, 5))
    # extent sets the coordinate ranges for the axes
    plt.imshow(
        u_final,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap="inferno",
    )
    plt.colorbar(label="u(x, y, t_final)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Heat Equation – final state")

    plot_path = output_dir / "final_state_2d.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    """Command-line entry point for the 2D heat equation solver.

    This function parses the command-line arguments, loads the configuration
    from a JSON file, runs the solver and saves numerical results and a
    heatmap of the final state.

    Typical usage::

        python heat_solver_2d.py --config config_2d.json
    """
    parser = argparse.ArgumentParser(
        description="2D heat equation solver with JSON config and file output."
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
    Lx = float(config.get("Lx", 1.0))
    Ly = float(config.get("Ly", 1.0))
    nx = int(config.get("nx", 51))
    ny = int(config.get("ny", 51))
    dt = float(config.get("dt", 1e-4))
    t_final = float(config.get("t_final", 0.1))

    # Build initial condition according to configuration
    initial_condition = build_initial_condition(config, Lx, Ly)

    # Output directory for result files
    output_dir = Path(config.get("output_dir", "output_2d"))

    # Solve the PDE
    x, y, t, u_history = solve_heat_equation_2d(
        alpha=alpha,
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        dt=dt,
        t_final=t_final,
        initial_condition=initial_condition,
    )

    # Save results and plot
    npz_path, csv_path = save_results(output_dir, x, y, t, u_history)
    plot_path = plot_final_state(output_dir, x, y, u_history)

    print("2D simulation finished.")
    print(f"Results (x, y, t, u_history) saved to: {npz_path}")
    print(f"Final profile CSV saved to:            {csv_path}")
    print(f"Final state plot saved to:             {plot_path}")


if __name__ == "__main__":
    main()