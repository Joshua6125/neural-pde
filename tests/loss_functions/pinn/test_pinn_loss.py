"""Tests for PINNLoss (loss computation)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.loss_functions.pinn import PINNLoss


class TestPINNLossInitialization:
    """Test PINNLoss initialization."""

    def test_init_with_required_args(self):
        """Can initialize with required args: u_model and c."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        assert loss.u_model is u_fn
        assert loss.c == 1.0

    def test_init_with_callable_c(self):
        """Can initialize with callable wave speed."""
        def u_fn(x):
            return jnp.sum(x)
        def c_fn(x):
            return 1.0

        loss = PINNLoss(u_model=u_fn, c=c_fn)
        assert callable(loss.c)

    def test_init_with_forcing_term(self):
        """Can initialize with forcing term f."""
        def u_fn(x):
            return jnp.sum(x)
        def f_fn(x):
            return 0.0

        loss = PINNLoss(u_model=u_fn, c=1.0, f=f_fn)
        assert loss.f is f_fn

    def test_init_with_ic_conditions(self):
        """Can initialize with IC conditions."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 0.0
        def ut0_fn(x):
            return 0.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ut0=ut0_fn)
        assert loss.u0 is u0_fn
        assert loss.ut0 is ut0_fn

    def test_init_stores_weights(self):
        """Initialization stores IC and BC weights."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0, ic_weight=2.0, bc_weight=3.0)
        assert loss.ic_weight == 2.0
        assert loss.bc_weight == 3.0

    def test_default_weights(self):
        """Default weights are 1.0."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        assert loss.ic_weight == 1.0
        assert loss.bc_weight == 1.0


class TestPINNLossUMethod:
    """Test _u method (model output)."""

    def test_u_returns_scalar_from_linear_model(self):
        """_u returns scalar from linear model."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])
        result = loss._u(x)

        # Should be scalar
        assert jnp.asarray(result).shape == ()

    def test_u_squeezes_output(self):
        """_u squeezes model output."""
        def u_fn(x):
            return jnp.array([jnp.sum(x)])  # Returns shape (1,)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])
        result = loss._u(x)

        # Should be scalar after squeeze
        assert jnp.asarray(result).shape == ()

    def test_u_with_quadratic_model(self):
        """_u works with quadratic model."""
        def u_fn(x):
            return jnp.sum(x**2)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])
        result = loss._u(x)

        expected = 0.5
        assert jnp.allclose(result, expected)

    def test_u_with_trig_model(self):
        """_u works with trigonometric model."""
        def u_fn(x):
            return jnp.sin(jnp.sum(x))

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])
        result = loss._u(x)

        expected = jnp.sin(1.0)
        assert jnp.allclose(result, expected)


class TestPINNLossPDEResidual:
    """Test _pde_residual (wave equation residual)."""

    def test_pde_residual_linear_model_1d(self):
        """Linear model u(t,x) = t+x has u_tt = u_xx = 0."""
        def u_fn(x):
            return jnp.sum(x)  # u(t,x) = t + x

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])

        # For u = t + x: u_tt = 0, u_xx = 0
        # Residual = (0 - 1^2 * 0)^2 = 0
        residual = loss._pde_residual(x)
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_pde_residual_with_scalar_wave_speed(self):
        """PDE residual with scalar wave speed."""
        def u_fn(x):
            return jnp.sum(x)

        c = 2.0
        loss = PINNLoss(u_model=u_fn, c=c)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_pde_residual_with_callable_wave_speed(self):
        """PDE residual with callable wave speed."""
        def u_fn(x):
            return jnp.sum(x)
        def c_fn(x):
            return 1.0 + 0.1 * jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=c_fn)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_pde_residual_with_forcing_term(self):
        """PDE residual includes forcing term."""
        def u_fn(x):
            return jnp.sum(x)
        def f_fn(x):
            return 1.0

        loss = PINNLoss(u_model=u_fn, c=1.0, f=f_fn)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_pde_residual_without_forcing_defaults_to_zero(self):
        """Without forcing term f, defaults to 0.0."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0, f=None)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_pde_residual_quadratic_exponential_growth(self):
        """PDE residual is squared (non-negative)."""
        def u_fn(x):
            return jnp.sin(jnp.sum(x))

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert residual >= 0.0

    def test_pde_residual_2d_spatial_laplacian(self):
        """PDE residual computes Laplacian for 2D spatial."""
        def u_fn(x):
            # u(t,x,y) = x^2 + y^2 + t^2
            return x[0]**2 + x[1]**2 + x[2]**2

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)


class TestPINNLossICResidual:
    """Test _ic_residual (initial condition residual)."""

    def test_ic_residual_with_both_conditions(self):
        """IC residual includes both displacement and velocity terms."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 0.0
        def ut0_fn(x):
            return 0.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ut0=ut0_fn, ic_weight=1.0)
        x = jnp.array([0.0, 0.5])

        residual = loss._ic_residual(x)
        assert jnp.isfinite(residual)
        assert residual >= 0.0

    def test_ic_residual_displacement_only(self):
        """IC residual with only u0 (ut0 defaults to 0)."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 1.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ut0=None)
        x = jnp.array([0.0, 0.0])

        residual = loss._ic_residual(x)
        assert jnp.isfinite(residual)

    def test_ic_residual_velocity_only(self):
        """IC residual with only ut0 (u0 defaults to 0)."""
        def u_fn(x):
            return jnp.sum(x)
        def ut0_fn(x):
            return 0.5

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=None, ut0=ut0_fn)
        x = jnp.array([0.0, 0.0])

        residual = loss._ic_residual(x)
        assert jnp.isfinite(residual)

    def test_ic_residual_both_none_defaults_to_zero(self):
        """IC residual with both None uses zero defaults."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=None, ut0=None, ic_weight=1.0)
        x = jnp.array([0.0, 0.5])

        residual = loss._ic_residual(x)
        assert jnp.isfinite(residual)

    def test_ic_residual_weight_modulation(self):
        """IC residual is modulated by ic_weight."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return jnp.asarray(1.0)

        loss_w1 = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ic_weight=1.0)
        loss_w2 = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ic_weight=2.0)

        x = jnp.array([0.0, 0.5])
        r1 = loss_w1._ic_residual(x)
        r2 = loss_w2._ic_residual(x)

        assert jnp.allclose(r2, 2.0 * r1)

    def test_ic_residual_zero_weight(self):
        """IC residual with zero weight returns zero."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 1.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ic_weight=0.0)
        x = jnp.array([0.0, 0.5])

        residual = loss._ic_residual(x)
        assert jnp.allclose(residual, 0.0)

    def test_ic_residual_includes_time_derivative(self):
        """IC residual includes ∂_t u term."""
        def u_fn(x):
            # u(t,x) = t^2 + x
            # ut = 2t, so at t=0: ut=0
            return x[0]**2 + x[1]
        def u0_fn(x):
            return x[1]  # u0(x) = x
        def ut0_fn(x):
            return 0.0  # ut0(x) = 0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ut0=ut0_fn, ic_weight=1.0)
        x = jnp.array([0.0, 0.5])

        residual = loss._ic_residual(x)
        # Should be small since IC conditions match
        assert jnp.isfinite(residual)


class TestPINNLossSpatialBCResidual:
    """Test _spatial_bc_residual (boundary condition residual)."""

    def test_spatial_bc_residual_homogeneous_dirichlet(self):
        """BC residual is bc_weight * u^2 (homogeneous Dirichlet)."""
        def u_fn(x):
            return jnp.asarray(0.5)

        loss = PINNLoss(u_model=u_fn, c=1.0, bc_weight=1.0)
        x = jnp.array([0.5, 0.0])  # At spatial boundary

        residual = loss._spatial_bc_residual(x)
        expected = 1.0 * (0.5**2)

        assert jnp.allclose(residual, expected)

    def test_spatial_bc_residual_zero_displacement(self):
        """BC residual is zero when u=0."""
        def u_fn(x):
            return jnp.asarray(0.0)

        loss = PINNLoss(u_model=u_fn, c=1.0, bc_weight=1.0)
        x = jnp.array([0.5, 0.0])

        residual = loss._spatial_bc_residual(x)
        assert jnp.allclose(residual, 0.0)

    def test_spatial_bc_residual_weight_modulation(self):
        """BC residual is modulated by bc_weight."""
        def u_fn(x):
            return jnp.asarray(1.0)

        loss_w1 = PINNLoss(u_model=u_fn, c=1.0, bc_weight=1.0)
        loss_w2 = PINNLoss(u_model=u_fn, c=1.0, bc_weight=3.0)

        x = jnp.array([0.5, 0.0])
        r1 = loss_w1._spatial_bc_residual(x)
        r2 = loss_w2._spatial_bc_residual(x)

        assert jnp.allclose(r2, 3.0 * r1)

    def test_spatial_bc_residual_zero_weight(self):
        """BC residual is zero when bc_weight=0."""
        def u_fn(x):
            return jnp.asarray(1.0)

        loss = PINNLoss(u_model=u_fn, c=1.0, bc_weight=0.0)
        x = jnp.array([0.5, 0.0])

        residual = loss._spatial_bc_residual(x)
        assert jnp.allclose(residual, 0.0)


class TestPINNLossInteriorLoss:
    """Test loss_interior (vectorized PDE loss)."""

    def test_interior_loss_single_point(self):
        """Interior loss works with single point."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x_interior = jnp.array([[0.5, 0.5]])  # Shape (1, 2)

        result = loss.loss_interior(x_interior)
        assert result.shape == (1,)
        assert jnp.all(jnp.isfinite(result))

    def test_interior_loss_multiple_points(self):
        """Interior loss works with multiple points."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x_interior = jnp.array([
            [0.5, 0.5],
            [0.2, 0.3],
            [0.8, 0.1],
        ])  # Shape (3, 2)

        result = loss.loss_interior(x_interior)
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_interior_loss_large_batch(self):
        """Interior loss works with large batch."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x_interior = jnp.zeros((100, 2))  # Shape (100, 2)

        result = loss.loss_interior(x_interior)
        assert result.shape == (100,)

    def test_interior_loss_with_forcing(self):
        """Interior loss includes forcing term."""
        def u_fn(x):
            return jnp.sum(x)
        def f_fn(x):
            return jnp.sin(jnp.sum(x))

        loss = PINNLoss(u_model=u_fn, c=1.0, f=f_fn)
        x_interior = jnp.array([[0.5, 0.5]])

        result = loss.loss_interior(x_interior)
        assert result.shape == (1,)
        assert jnp.isfinite(result[0])

    def test_interior_loss_2d_spatial(self):
        """Interior loss works for 2D spatial."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x_interior = jnp.array([
            [0.5, 0.5, 0.5],  # [t, x, y]
            [0.3, 0.2, 0.7],
        ])  # Shape (2, 3)

        result = loss.loss_interior(x_interior)
        assert result.shape == (2,)


class TestPINNLossBoundaryLoss:
    """Test loss_boundary (conditional IC/BC routing)."""

    def test_boundary_loss_ic_detection(self):
        """Boundary loss detects IC points (normal_vector[:, 0] < 0)."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 0.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ic_weight=1.0)

        x_boundary = jnp.array([[0.0, 0.5]])  # IC point
        normal_vector = jnp.array([[-1.0, 0.0]])  # Negative t-component

        result = loss.loss_boundary(x_boundary, normal_vector)
        assert result.shape == (1,)
        assert jnp.isfinite(result[0])

    def test_boundary_loss_bc_detection(self):
        """Boundary loss detects BC points (normal_vector[:, 0] == 0)."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0, bc_weight=1.0)

        x_boundary = jnp.array([[0.5, 0.0]])  # Spatial BC point
        normal_vector = jnp.array([[0.0, 1.0]])  # Zero t-component

        result = loss.loss_boundary(x_boundary, normal_vector)
        assert result.shape == (1,)
        assert jnp.isfinite(result[0])

    def test_boundary_loss_mixed_points(self):
        """Boundary loss handles mixed IC and BC points."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 0.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ic_weight=1.0, bc_weight=1.0)

        x_boundary = jnp.array([
            [0.0, 0.5],   # IC point
            [0.5, 0.0],   # BC point
        ])
        normal_vector = jnp.array([
            [-1.0, 0.0],  # IC normal
            [0.0, 1.0],   # BC normal
        ])

        result = loss.loss_boundary(x_boundary, normal_vector)
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_boundary_loss_interior_points_return_zero(self):
        """Boundary loss returns zero for interior points."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)

        x_boundary = jnp.array([[0.5, 0.5]])  # Interior point
        normal_vector = jnp.array([[1.0, 0.0]])  # Positive t-component

        result = loss.loss_boundary(x_boundary, normal_vector)
        assert jnp.allclose(result, 0.0)

    def test_boundary_loss_batch_mixed_ic_bc(self):
        """Boundary loss handles batch with mixed IC and BC."""
        def u_fn(x):
            return jnp.sum(x)
        def u0_fn(x):
            return 0.0

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ic_weight=1.0, bc_weight=1.0)

        x_boundary = jnp.array([
            [0.0, 0.2],   # IC
            [0.0, 0.5],   # IC
            [0.5, 0.0],   # BC
            [0.7, 0.0],   # BC
        ])
        normal_vector = jnp.array([
            [-1.0, 0.0],  # IC
            [-1.0, 0.0],  # IC
            [0.0, 1.0],   # BC
            [0.0, 1.0],   # BC
        ])

        result = loss.loss_boundary(x_boundary, normal_vector)
        assert result.shape == (4,)
        assert jnp.all(jnp.isfinite(result))


class TestPINNLossFunctionsInterface:
    """Test loss_functions() interface."""

    def test_loss_functions_returns_tuple(self):
        """loss_functions() returns tuple of 2 callables."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        result = loss.loss_functions()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_functions_callables(self):
        """Loss functions are callable."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        interior_loss_fn, boundary_loss_fn = loss.loss_functions()

        assert callable(interior_loss_fn)
        assert callable(boundary_loss_fn)

    def test_loss_functions_first_is_interior(self):
        """First returned function is interior loss."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        interior_loss_fn, _ = loss.loss_functions()

        x_interior = jnp.array([[0.5, 0.5]])
        result = interior_loss_fn(x_interior)
        assert result.shape == (1,)

    def test_loss_functions_second_is_boundary(self):
        """Second returned function is boundary loss."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        _, boundary_loss_fn = loss.loss_functions()

        x_boundary = jnp.array([[0.5, 0.5]])
        normal_vector = jnp.array([[0.0, 1.0]])
        result = boundary_loss_fn(x_boundary, normal_vector)
        assert result.shape == (1,)


class TestPINNLossNumericalEdgeCases:
    """Test numerical edge cases and stability."""

    def test_very_small_displacement(self):
        """Loss computation with very small displacements."""
        def u_fn(x):
            return 1e-6 * jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_large_wave_speed(self):
        """Loss computation with large wave speed."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1000.0)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_small_wave_speed(self):
        """Loss computation with small wave speed."""
        def u_fn(x):
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=0.001)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_zero_displacement_at_origin(self):
        """Loss computation at origin with zero displacement."""
        def u_fn(x):
            return jnp.asarray(0.0)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.0, 0.0])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)


class TestPINNLossRegressionKnownSolutions:
    """Regression tests against known analytical solutions."""

    def test_zero_solution_pde_residual(self):
        """Zero solution u≡0 has zero PDE residual (no forcing)."""
        def u_fn(x):
            return jnp.asarray(0.0)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.allclose(residual, 0.0, atol=1e-10)

    def test_zero_solution_with_zero_ic(self):
        """Zero solution with u0=0, ut0=0 has zero IC loss."""
        def u_fn(x):
            return jnp.asarray(0.0)
        def u0_fn(x):
            return jnp.asarray(0.0)
        def ut0_fn(x):
            return jnp.asarray(0.0)

        loss = PINNLoss(u_model=u_fn, c=1.0, u0=u0_fn, ut0=ut0_fn, ic_weight=1.0)
        x = jnp.array([0.0, 0.5])

        residual = loss._ic_residual(x)
        assert jnp.allclose(residual, 0.0, atol=1e-10)

    def test_constant_solution_pde_residual(self):
        """Constant solution u≡u0 has zero PDE residual (no forcing)."""
        def u_fn(x):
            return jnp.asarray(1.0)  # Constant

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.allclose(residual, 0.0, atol=1e-10)


class TestPINNLossDimensionHandling:
    """Test dimension handling (1D, 2D, 3D spatial)."""

    def test_1d_spatial_plus_time(self):
        """PINNLoss works for 1D spatial + time (2D total)."""
        def u_fn(x):
            # x shape: (2,) = [t, x]
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_2d_spatial_plus_time(self):
        """PINNLoss works for 2D spatial + time (3D total)."""
        def u_fn(x):
            # x shape: (3,) = [t, x, y]
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_3d_spatial_plus_time(self):
        """PINNLoss works for 3D spatial + time (4D total)."""
        def u_fn(x):
            # x shape: (4,) = [t, x, y, z]
            return jnp.sum(x)

        loss = PINNLoss(u_model=u_fn, c=1.0)
        x = jnp.array([0.5, 0.5, 0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)


class TestPINNLossCallableIntegration:
    """Test integration of callable parameters."""

    def test_callable_c_evaluated_correctly(self):
        """Callable wave speed c(x) is evaluated at point."""
        def u_fn(x):
            return jnp.sum(x)
        def c_fn(x):
            return jnp.sum(x)  # c depends on x

        loss = PINNLoss(u_model=u_fn, c=c_fn)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_callable_f_evaluated_correctly(self):
        """Callable forcing f(x) is evaluated at point."""
        def u_fn(x):
            return jnp.sum(x)
        def f_fn(x):
            return jnp.sin(jnp.sum(x))

        loss = PINNLoss(u_model=u_fn, c=1.0, f=f_fn)
        x = jnp.array([0.5, 0.5])

        residual = loss._pde_residual(x)
        assert jnp.isfinite(residual)

    def test_all_callables_together(self):
        """All callable parameters work together."""
        def u_fn(x):
            return jnp.sin(jnp.sum(x))
        def c_fn(x):
            return 0.5 + 0.1 * jnp.sum(x**2)
        def f_fn(x):
            return jnp.cos(x[0])
        def u0_fn(x):
            return jnp.sin(x[1])
        def ut0_fn(x):
            return 0.0

        loss = PINNLoss(
            u_model=u_fn,
            c=c_fn,
            f=f_fn,
            u0=u0_fn,
            ut0=ut0_fn,
            ic_weight=1.0,
            bc_weight=1.0
        )

        # Test interior PDE
        x_int = jnp.array([0.5, 0.5])
        residual_int = loss._pde_residual(x_int)
        assert jnp.isfinite(residual_int)

        # Test IC
        x_ic = jnp.array([0.0, 0.5])
        residual_ic = loss._ic_residual(x_ic)
        assert jnp.isfinite(residual_ic)

        # Test BC
        x_bc = jnp.array([0.5, 0.0])
        normal_bc = jnp.array([0.0, 1.0])
        residual_bc = loss._spatial_bc_residual(x_bc)
        assert jnp.isfinite(residual_bc)
