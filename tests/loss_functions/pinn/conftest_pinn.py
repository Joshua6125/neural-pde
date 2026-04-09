"""Pytest fixtures for PINN module tests.

Provides mock models, callable fixtures, and test data points.
"""

import pytest
import jax
import jax.numpy as jnp


# ============================================================================
# MOCK NEURAL NETWORK MODELS
# ============================================================================

class MockModel:
    """Base class for mock neural network models."""

    def __init__(self, compute_fn):
        """Initialize with computation function."""
        self.compute_fn = compute_fn

    def init(self, rng_key, x):
        """Initialize parameters."""
        return {"dummy": 0}  # No learnable params

    def apply(self, params, x):
        """Apply model to input."""
        return {"u": self.compute_fn(x)}


def make_mock_linear_model():
    """Mock model: u(x) = sum(x_i), returns {"u": scalar}."""
    return MockModel(lambda x: jnp.sum(x))


def make_mock_quadratic_model():
    """Mock model: u(x) = sum(x_i^2), returns {"u": scalar}."""
    return MockModel(lambda x: jnp.sum(x**2))


def make_mock_trig_model():
    """Mock model: u(x) = sin(sum(x)), returns {"u": scalar}."""
    return MockModel(lambda x: jnp.sin(jnp.sum(x)))


class MockModelInvalid:
    """Mock model with invalid output (no 'u' key)."""

    def init(self, rng_key, x):
        return {"dummy": 0}

    def apply(self, params, x):
        return {"v": jnp.sum(x)}  # Wrong key


def make_mock_invalid_model():
    """Mock model with invalid output (no 'u' key)."""
    return MockModelInvalid()


class MockModelInvalidShape:
    """Mock model with invalid output shape (not scalar)."""

    def init(self, rng_key, x):
        return {"dummy": 0}

    def apply(self, params, x):
        return {"u": jnp.array([jnp.sum(x), jnp.sum(x)])}  # Shape (2,)


def make_mock_invalid_output_shape_model():
    """Mock model with invalid output shape (not scalar)."""
    return MockModelInvalidShape()


# ============================================================================
# CALLABLE FIXTURES (c, f, u0, ut0)
# ============================================================================

@pytest.fixture
def callable_c_constant():
    """Wave speed: c(x) = 1.0 (scalar callable)."""
    return lambda x: 1.0


@pytest.fixture
def callable_c_linear():
    """Wave speed: c(x) = 0.5 + 0.1 * sum(x)."""
    return lambda x: 0.5 + 0.1 * jnp.sum(x)


@pytest.fixture
def callable_c_quadratic():
    """Wave speed: c(x) = sqrt(1 + sum(x^2))."""
    return lambda x: jnp.sqrt(1.0 + jnp.sum(x**2))


@pytest.fixture
def callable_f_zero():
    """Forcing term: f(x) = 0."""
    return lambda x: 0.0


@pytest.fixture
def callable_f_constant():
    """Forcing term: f(x) = 1.0."""
    return lambda x: 1.0


@pytest.fixture
def callable_f_linear():
    """Forcing term: f(x) = sum(x)."""
    return lambda x: jnp.sum(x)


@pytest.fixture
def callable_u0_zero():
    """Initial displacement: u0(x) = 0."""
    return lambda x: 0.0


@pytest.fixture
def callable_u0_constant():
    """Initial displacement: u0(x) = 1.0."""
    return lambda x: 1.0


@pytest.fixture
def callable_u0_linear():
    """Initial displacement: u0(x) = sum(x)."""
    return lambda x: jnp.sum(x)


@pytest.fixture
def callable_u0_trig():
    """Initial displacement: u0(x) = sin(sum(x))."""
    return lambda x: jnp.sin(jnp.sum(x))


@pytest.fixture
def callable_ut0_zero():
    """Initial velocity: ut0(x) = 0."""
    return lambda x: 0.0


@pytest.fixture
def callable_ut0_constant():
    """Initial velocity: ut0(x) = 1.0."""
    return lambda x: 1.0


@pytest.fixture
def callable_ut0_linear():
    """Initial velocity: ut0(x) = sum(x)."""
    return lambda x: jnp.sum(x)


# ============================================================================
# MOCK MODEL FIXTURES (Parameterized)
# ============================================================================

@pytest.fixture
def mock_model_linear():
    """Mock linear model: u(x) = sum(x)."""
    return make_mock_linear_model()


@pytest.fixture
def mock_model_quadratic():
    """Mock quadratic model: u(x) = sum(x^2)."""
    return make_mock_quadratic_model()


@pytest.fixture
def mock_model_trig():
    """Mock trig model: u(x) = sin(sum(x))."""
    return make_mock_trig_model()


@pytest.fixture
def mock_model_invalid():
    """Mock model with invalid output dict."""
    return make_mock_invalid_model()


@pytest.fixture
def mock_model_invalid_shape():
    """Mock model with invalid output shape."""
    return make_mock_invalid_output_shape_model()


# ============================================================================
# TEST DATA POINTS
# ============================================================================

@pytest.fixture
def interior_points_1d():
    """1D spatial interior test points: (n, 2) = [t, x]."""
    t = jnp.linspace(0, 1, 5)
    x = jnp.linspace(0, 1, 5)
    xx, tt = jnp.meshgrid(x, t)
    points = jnp.stack([tt.flatten(), xx.flatten()], axis=1)  # Shape (25, 2)
    return points


@pytest.fixture
def interior_points_2d():
    """2D spatial interior test points: (n, 3) = [t, x, y]."""
    t = jnp.array([0.5])
    x = jnp.linspace(0, 1, 3)
    y = jnp.linspace(0, 1, 3)
    xx, yy = jnp.meshgrid(x, y)
    tt = jnp.full_like(xx, 0.5)
    points = jnp.stack([tt.flatten(), xx.flatten(), yy.flatten()], axis=1)  # Shape (9, 3)
    return points


@pytest.fixture
def interior_points_3d():
    """3D spatial interior test points: (n, 4) = [t, x, y, z]."""
    t = jnp.array([0.5])
    x = jnp.linspace(0, 1, 2)
    y = jnp.linspace(0, 1, 2)
    z = jnp.linspace(0, 1, 2)
    xx, yy, zz = jnp.meshgrid(x, y, z)
    tt = jnp.full_like(xx, 0.5)
    points = jnp.stack([tt.flatten(), xx.flatten(), yy.flatten(), zz.flatten()], axis=1)  # Shape (8, 4)
    return points


@pytest.fixture
def boundary_ic_points_1d():
    """IC boundary points (t < 0): normals with negative t-component."""
    x = jnp.linspace(0, 1, 5)
    t = jnp.full_like(x, -0.1)  # t < 0
    points = jnp.stack([t, x], axis=1)  # Shape (5, 2)

    # Normal vectors: negative t-component for IC
    normals = jnp.zeros_like(points)
    normals = normals.at[:, 0].set(-1.0)  # Negative t-direction

    return points, normals


@pytest.fixture
def boundary_bc_points_1d():
    """Spatial BC boundary points (t >= 0, x at edge): normals with t=0."""
    t = jnp.linspace(0, 1, 5)
    x = jnp.zeros_like(t)  # x = 0 (spatial boundary)
    points = jnp.stack([t, x], axis=1)  # Shape (5, 2)

    # Normal vectors: t-component = 0 for spatial BC
    normals = jnp.zeros_like(points)
    normals = normals.at[:, 1].set(1.0)  # Spatial normal

    return points, normals


@pytest.fixture
def boundary_bc_points_2d():
    """Spatial BC boundary points (2D): normals with t=0."""
    n_bc = 4
    t = jnp.linspace(0, 1, n_bc)
    x = jnp.zeros(n_bc)
    y = jnp.zeros(n_bc)
    points = jnp.stack([t, x, y], axis=1)  # Shape (4, 3)

    # Normal vectors: t-component = 0
    normals = jnp.zeros_like(points)
    normals = normals.at[:, 1].set(1.0)  # Spatial normal (x-direction)

    return points, normals


# ============================================================================
# PINN CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def pinn_config_default():
    """Default PINNConfig with minimal args."""
    from src.algorithms import PINNConfig
    return PINNConfig()


@pytest.fixture
def pinn_config_with_ic():
    """PINNConfig with IC conditions and scalar wave speed."""
    from src.algorithms import PINNConfig

    def u0(x):
        return jnp.sin(jnp.sum(x[1:]))  # spatial dims

    def ut0(x):
        return 0.0

    return PINNConfig(c=1.0, u0=u0, ut0=ut0, ic_weight=1.0, bc_weight=1.0)


@pytest.fixture
def pinn_config_with_forcing():
    """PINNConfig with forcing term."""
    from src.algorithms import PINNConfig

    def f(x):
        return jnp.sin(jnp.sum(x))

    return PINNConfig(c=1.0, f=f)


@pytest.fixture
def pinn_config_variable_c():
    """PINNConfig with variable wave speed."""
    from src.algorithms import PINNConfig

    def c(x):
        return 0.5 + 0.1 * jnp.sum(x**2)

    return PINNConfig(c=c)


@pytest.fixture
def pinn_config_weighted():
    """PINNConfig with non-default weights."""
    from src.algorithms import PINNConfig
    return PINNConfig(ic_weight=2.0, bc_weight=0.5)


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def rng_key():
    """JAX PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_input_1d():
    """1D sample input: [t, x]."""
    return jnp.array([0.5, 0.5])


@pytest.fixture
def sample_input_2d():
    """2D sample input: [t, x, y]."""
    return jnp.array([0.5, 0.5, 0.5])


@pytest.fixture
def sample_input_3d():
    """3D sample input: [t, x, y, z]."""
    return jnp.array([0.5, 0.5, 0.5, 0.5])
