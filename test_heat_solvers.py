"""
Unit tests for the 1D and 2D heat equation solvers.

This module uses the standard library :mod:`unittest` together with NumPy
to verify basic properties of the numerical solvers implemented in
:mod:`heat_solver` (1D) and :mod:`heat_solver_2d` (2D).

The tests cover:

* stability condition checks,
* initial condition builders, and
* file output routines.
"""

import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

import heat_solver       # 1D solver file: heat_solver.py
import heat_solver_2d    # 2D solver file: heat_solver_2d.py


class TestHeatSolver1D(unittest.TestCase):
    """Tests for the 1D heat equation solver."""

    def test_stability_condition_ok(self):
        """Test that a stable time step runs without error.

        Chooses parameters such that the stability condition
        :math:`\\alpha dt / dx^2 \\le 0.5` is clearly satisfied.
        """
        alpha = 1.0
        L = 1.0
        nx = 51
        dx = L / (nx - 1)
        dt = 0.25 * dx**2  # clearly below 0.5 * dx^2

        x, u_final, u_hist, t = heat_solver.solve_heat_equation_1d(
            alpha=alpha,
            L=L,
            nx=nx,
            dt=dt,
            t_final=0.01,
            initial_condition=None,
        )
        # Basic shape checks
        self.assertEqual(x.shape[0], nx)
        self.assertEqual(u_hist.shape[1], nx)
        self.assertGreater(u_hist.shape[0], 1)  # more than one time step
        self.assertTrue(np.all(np.isfinite(u_hist)))  # no NaNs/Infs

    def test_stability_condition_raises(self):
        """Test that too large ``dt`` raises a :class:`ValueError`."""
        alpha = 1.0
        L = 1.0
        nx = 51
        dx = L / (nx - 1)
        dt = 1.0 * dx**2  # violates alpha*dt/dx^2 <= 0.5

        with self.assertRaises(ValueError):
            heat_solver.solve_heat_equation_1d(
                alpha=alpha,
                L=L,
                nx=nx,
                dt=dt,
                t_final=0.01,
                initial_condition=None,
            )

    def test_initial_condition_gaussian_builder(self):
        """Test the Gaussian initial condition builder for 1D."""
        config = {
            "initial_condition": {
                "type": "gaussian",
                "center": 0.5,
                "sigma": 0.1,
            }
        }
        L = 1.0
        ic = heat_solver.build_initial_condition(config, L)
        x = np.linspace(0.0, L, 11)
        u0 = ic(x)
        self.assertEqual(u0.shape, x.shape)
        # Gaussian is strictly positive everywhere
        self.assertTrue(np.all(u0 > 0.0))

    def test_save_results_and_files_exist(self):
        """Test that saving 1D results creates valid output files."""
        # Simple short run
        x, u_final, u_hist, t = heat_solver.solve_heat_equation_1d(
            alpha=1.0,
            L=1.0,
            nx=21,
            dt=1e-4,
            t_final=1e-3,
            initial_condition=None,
        )

        tmpdir = Path(tempfile.mkdtemp())
        try:
            npz_path, csv_path = heat_solver.save_results(tmpdir, x, t, u_hist)
            self.assertTrue(npz_path.exists())
            self.assertTrue(csv_path.exists())

            # Load NPZ and check shapes
            data = np.load(npz_path)
            self.assertIn("x", data)
            self.assertIn("t", data)
            self.assertIn("u_history", data)
            self.assertEqual(data["u_history"].shape, u_hist.shape)
        finally:
            # Clean up temporary directory
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestHeatSolver2D(unittest.TestCase):
    """Tests for the 2D heat equation solver."""

    def test_stability_condition_ok(self):
        """Test that a stable time step in 2D runs without error."""
        alpha = 1.0
        Lx = 1.0
        Ly = 1.0
        nx = 21
        ny = 21
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        # Choose a sufficiently small dt
        dt = 0.1 * min(dx**2, dy**2)

        x, y, t, u_hist = heat_solver_2d.solve_heat_equation_2d(
            alpha=alpha,
            Lx=Lx,
            Ly=Ly,
            nx=nx,
            ny=ny,
            dt=dt,
            t_final=0.01,
            initial_condition=None,
        )
        self.assertEqual(x.shape[0], nx)
        self.assertEqual(y.shape[0], ny)
        self.assertEqual(u_hist.shape[1:], (ny, nx))
        self.assertGreater(u_hist.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(u_hist)))

    def test_stability_condition_raises(self):
        """Test that too large ``dt`` in 2D raises a :class:`ValueError`."""
        alpha = 1.0
        Lx = 1.0
        Ly = 1.0
        nx = 21
        ny = 21
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        # Choose dt so that alpha*dt*(1/dx^2 + 1/dy^2) > 0.5
        dt = 1.0 * dx**2

        with self.assertRaises(ValueError):
            heat_solver_2d.solve_heat_equation_2d(
                alpha=alpha,
                Lx=Lx,
                Ly=Ly,
                nx=nx,
                ny=ny,
                dt=dt,
                t_final=0.01,
                initial_condition=None,
            )

    def test_initial_condition_builder_gaussian(self):
        """Test the Gaussian initial condition builder for 2D."""
        config = {
            "initial_condition": {
                "type": "gaussian",
                "center_x": 0.5,
                "center_y": 0.5,
                "sigma_x": 0.2,
                "sigma_y": 0.2,
            }
        }
        Lx = 1.0
        Ly = 1.0
        ic = heat_solver_2d.build_initial_condition(config, Lx, Ly)
        x = np.linspace(0.0, Lx, 11)
        y = np.linspace(0.0, Ly, 11)
        X, Y = np.meshgrid(x, y)
        u0 = ic(X, Y)
        self.assertEqual(u0.shape, X.shape)
        self.assertTrue(np.all(u0 > 0.0))

    def test_save_results_and_files_exist(self):
        """Test that saving 2D results creates valid output files."""
        x, y, t, u_hist = heat_solver_2d.solve_heat_equation_2d(
            alpha=1.0,
            Lx=1.0,
            Ly=1.0,
            nx=11,
            ny=11,
            dt=1e-4,
            t_final=5e-4,
            initial_condition=None,
        )

        tmpdir = Path(tempfile.mkdtemp())
        try:
            npz_path, csv_path = heat_solver_2d.save_results(
                tmpdir, x, y, t, u_hist
            )
            self.assertTrue(npz_path.exists())
            self.assertTrue(csv_path.exists())

            data = np.load(npz_path)
            self.assertIn("x", data)
            self.assertIn("y", data)
            self.assertIn("t", data)
            self.assertIn("u_history", data)
            self.assertEqual(data["u_history"].shape, u_hist.shape)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()