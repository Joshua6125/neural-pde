"""Tests for LSLoss (loss computation)."""

import jax.numpy as jnp
import pytest

from src.loss_functions import LSLoss


pytestmark = pytest.mark.LS


class TestLSLossInitialisation:
    """Test LSLoss initialisation."""

    def test_init_with_required_args(self):
        """Can initialise with required args: v_model and sigma_model."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        assert loss.v_model is v_fn
        assert loss.sigma_model is sigma_fn

    def test_init_with_sources(self, callable_f_zero, callable_g_zero):
        """Can initialise with source terms."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, f=callable_f_zero, g=callable_g_zero)
        assert loss.f is callable_f_zero
        assert loss.g is callable_g_zero

    def test_init_with_initial_conditions(self, callable_v0_zero, callable_sigma0_zero):
        """Can initialise with initial conditions."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v0=callable_v0_zero, sigma0=callable_sigma0_zero)
        assert loss.v0 is callable_v0_zero
        assert loss.sigma0 is callable_sigma0_zero

    def test_init_with_boundary_condition(self, callable_v_boundary_zero):
        """Can initialise with a boundary condition."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v_boundary=callable_v_boundary_zero)
        assert loss.v_boundary is callable_v_boundary_zero

    def test_init_prints_warning_when_boundary_set(self, callable_v_boundary_zero, capsys):
        """A warning is emitted when v_boundary is provided."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        LSLoss(v_model=v_fn, sigma_model=sigma_fn, v_boundary=callable_v_boundary_zero)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestLSLossVMethod:
    """Test _v method (velocity field output)."""

    def test_v_returns_scalar(self):
        """_v returns a scalar for scalar-shaped model output."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        result = loss._v(jnp.array([0.5, 0.25]))
        assert jnp.asarray(result).shape == ()

    def test_v_squeezes_length_one_vector(self):
        """_v squeezes a length-one vector to a scalar."""
        def v_fn(x):
            return jnp.asarray([x[0]])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        result = loss._v(jnp.array([0.5, 0.25]))
        assert jnp.asarray(result).shape == ()


class TestLSLossSigmaMethod:
    """Test _sigma method (stress/flux output)."""

    def test_sigma_flattens_vector_output(self):
        """_sigma flattens a vector output to 1D."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([[x[1]]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        result = loss._sigma(jnp.array([0.5, 0.25]))
        assert result.shape == (1,)

    def test_sigma_returns_expected_dimension(self):
        """_sigma preserves the expected component count."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([x[1], x[1] / 2.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        result = loss._sigma(jnp.array([0.5, 0.25, 0.75]))
        assert result.shape == (2,)


class TestLSLossInteriorResidual:
    """Test _interior_residual (first-order acoustic system residual)."""

    def test_interior_residual_zero_for_exact_1d_solution(self):
        """Exact 1D solution gives zero interior residual."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        residual = loss._interior_residual(jnp.array([0.5, 0.25]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)

    def test_interior_residual_zero_for_exact_2d_solution(self):
        """Exact 2D solution gives zero interior residual."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([x[1] / 2.0, x[2] / 2.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        residual = loss._interior_residual(jnp.array([0.5, 0.25, 0.75]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)

    def test_interior_residual_includes_source_terms(self):
        """Source terms f and g appear in the residual."""
        def v_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(0.0)

        def sigma_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray([0.0])

        def f_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray([1.0])

        def g_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray([1.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, f=f_fn, g=g_fn)
        residual = loss._interior_residual(jnp.array([0.5, 0.25]))
        assert jnp.allclose(residual, 2.0, atol=1e-8)

    def test_interior_residual_without_sources_defaults_to_zero(self):
        """None source terms default to zero contributions."""
        def v_fn(x):
            return jnp.asarray(0.0)

        def sigma_fn(x):
            return jnp.asarray([0.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, f=None, g=None)
        residual = loss._interior_residual(jnp.array([0.5, 0.25]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)

    def test_loss_interior_vectorizes_over_batch(self):
        """loss_interior returns one residual per point."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        points = jnp.array([
            [0.2, 0.1],
            [0.5, 0.3],
            [0.7, 0.8],
        ])
        residuals = loss.loss_interior(points)
        assert residuals.shape == (3,)
        assert jnp.allclose(residuals, 0.0, atol=1e-8)


class TestLSLossICResidual:
    """Test _ic_residual (initial condition residual)."""

    def test_ic_residual_zero_when_conditions_match(self, callable_v0_linear, callable_sigma0_linear):
        """Matching ICs produce zero residual."""
        def v_fn(x):
            return jnp.asarray(x[1])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v0=callable_v0_linear, sigma0=callable_sigma0_linear)
        residual = loss._ic_residual(jnp.array([0.0, 0.5]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)

    def test_ic_residual_defaults_to_zero_references(self):
        """None initial conditions default to zero references."""
        def v_fn(x):
            return jnp.asarray(x[1])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        residual = loss._ic_residual(jnp.array([0.0, 0.5]))
        assert jnp.allclose(residual, 0.5, atol=1e-8)

    def test_ic_residual_with_only_v0(self, callable_v0_zero):
        """v0 alone is used while sigma0 defaults to zero."""
        def v_fn(x):
            return jnp.asarray(1.0)

        def sigma_fn(x):
            return jnp.asarray([1.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v0=callable_v0_zero)
        residual = loss._ic_residual(jnp.array([0.0, 0.5]))
        assert jnp.allclose(residual, 2.0, atol=1e-8)

    def test_ic_residual_with_only_sigma0(self, callable_sigma0_zero):
        """sigma0 alone is used while v0 defaults to zero."""
        def v_fn(x):
            return jnp.asarray(1.0)

        def sigma_fn(x):
            return jnp.asarray([1.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, sigma0=callable_sigma0_zero)
        residual = loss._ic_residual(jnp.array([0.0, 0.5]))
        assert jnp.allclose(residual, 2.0, atol=1e-8)


class TestLSLossSpatialBCResidual:
    """Test _spatial_bc_residual (Dirichlet boundary residual)."""

    def test_spatial_bc_residual_returns_zero_without_boundary_callable(self):
        """Without v_boundary the residual is zero."""
        def v_fn(x):
            return jnp.asarray(2.0)

        def sigma_fn(x):
            return jnp.asarray([0.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        residual = loss._spatial_bc_residual(jnp.array([0.2, 0.0]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)

    def test_spatial_bc_residual_uses_boundary_callable(self, callable_v_boundary_zero):
        """With v_boundary the residual is the squared boundary mismatch."""
        def v_fn(x):
            return jnp.asarray(2.0)

        def sigma_fn(x):
            return jnp.asarray([0.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v_boundary=callable_v_boundary_zero)
        residual = loss._spatial_bc_residual(jnp.array([0.2, 0.0]))
        assert jnp.allclose(residual, 4.0, atol=1e-8)


class TestLSLossBoundaryLoss:
    """Test loss_boundary (IC vs BC routing)."""

    def test_boundary_loss_routes_ic_points(self, callable_v0_linear, callable_sigma0_linear):
        """IC points use the IC residual."""
        def v_fn(x):
            return jnp.asarray(x[1])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v0=callable_v0_linear, sigma0=callable_sigma0_linear)
        points = jnp.array([[0.0, 0.5]])
        normals = jnp.array([[-1.0, 0.0]])
        residuals = loss.loss_boundary(points, normals)
        assert residuals.shape == (1,)
        assert jnp.allclose(residuals, 0.0, atol=1e-8)

    def test_boundary_loss_routes_spatial_bc_points(self, callable_v_boundary_zero):
        """Spatial boundary points use the boundary residual."""
        def v_fn(x):
            return jnp.asarray(2.0)

        def sigma_fn(x):
            return jnp.asarray([0.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn, v_boundary=callable_v_boundary_zero)
        points = jnp.array([[0.2, 0.0]])
        normals = jnp.array([[0.0, 1.0]])
        residuals = loss.loss_boundary(points, normals)
        assert residuals.shape == (1,)
        assert jnp.allclose(residuals, 4.0, atol=1e-8)

    def test_boundary_loss_returns_zero_for_exterior_points(self):
        """Points with positive time-component normals are ignored."""
        def v_fn(x):
            return jnp.asarray(2.0)

        def sigma_fn(x):
            return jnp.asarray([0.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        points = jnp.array([[0.5, 0.5]])
        normals = jnp.array([[1.0, 0.0]])
        residuals = loss.loss_boundary(points, normals)
        assert jnp.allclose(residuals, 0.0, atol=1e-8)

    def test_boundary_loss_handles_mixed_batch(self, callable_v0_linear, callable_sigma0_linear, callable_v_boundary_zero):
        """Boundary loss handles mixed IC, BC, and exterior points."""
        def v_fn(x):
            return jnp.asarray(x[1])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(
            v_model=v_fn,
            sigma_model=sigma_fn,
            v0=callable_v0_linear,
            sigma0=callable_sigma0_linear,
            v_boundary=callable_v_boundary_zero,
        )
        points = jnp.array([
            [0.0, 0.5],
            [0.2, 0.0],
            [0.5, 0.5],
        ])
        normals = jnp.array([
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        residuals = loss.loss_boundary(points, normals)
        assert residuals.shape == (3,)
        assert jnp.allclose(residuals[0], 0.0, atol=1e-8)
        assert jnp.allclose(residuals[1], 0.0, atol=1e-8)
        assert jnp.allclose(residuals[2], 0.0, atol=1e-8)


class TestLSLossInterface:
    """Test loss_functions() interface."""

    def test_loss_functions_returns_tuple(self):
        """loss_functions() returns a tuple of two callables."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        result = loss.loss_functions()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_functions_returns_callables(self):
        """loss_functions() returns callable losses."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        interior_loss_fn, boundary_loss_fn = loss.loss_functions()
        assert callable(interior_loss_fn)
        assert callable(boundary_loss_fn)

    def test_loss_functions_match_methods(self):
        """The returned functions map to the loss methods."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return x[1:]

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        interior_loss_fn, boundary_loss_fn = loss.loss_functions()
        assert interior_loss_fn.__self__ is loss
        assert boundary_loss_fn.__self__ is loss
        assert interior_loss_fn.__func__ is LSLoss.loss_interior
        assert boundary_loss_fn.__func__ is LSLoss.loss_boundary


class TestLSLossDimensionHandling:
    """Test dimension handling for 1D and 2D spatial cases."""

    def test_1d_spatial_interior_point(self):
        """The loss handles [t, x] inputs."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([x[1]])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        residual = loss._interior_residual(jnp.array([0.5, 0.25]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)

    def test_2d_spatial_interior_point(self):
        """The loss handles [t, x, y] inputs."""
        def v_fn(x):
            return jnp.asarray(x[0])

        def sigma_fn(x):
            return jnp.asarray([x[1] / 2.0, x[2] / 2.0])

        loss = LSLoss(v_model=v_fn, sigma_model=sigma_fn)
        residual = loss._interior_residual(jnp.array([0.5, 0.25, 0.75]))
        assert jnp.allclose(residual, 0.0, atol=1e-8)
