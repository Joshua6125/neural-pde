import jax.numpy as jnp
import pytest
from dataclasses import replace


@pytest.mark.quadrature
def test_quadrature_constant_1d(config_quadrature_1d, test_functions_1d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    const_func = test_functions_1d['constant']

    result = integrator.integrate_interior(const_func['func'])
    expected = const_func['integral']

    assert jnp.allclose(result, expected, atol=const_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_linear_1d(config_quadrature_1d, test_functions_1d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    linear_func = test_functions_1d['linear']

    result = integrator.integrate_interior(linear_func['func'])
    expected = linear_func['integral']

    assert jnp.allclose(result, expected, atol=linear_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_quadratic_1d(config_quadrature_1d, test_functions_1d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    quadratic_func = test_functions_1d['quadratic']

    result = integrator.integrate_interior(quadratic_func['func'])
    expected = quadratic_func['integral']

    assert jnp.allclose(result, expected, atol=quadratic_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_sine_1d(config_quadrature_1d, test_functions_1d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    sine_func = test_functions_1d['sine']

    result = integrator.integrate_interior(sine_func['func'])
    expected = sine_func['integral']

    assert jnp.allclose(result, expected, atol=sine_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_exponential_1d(config_quadrature_1d, test_functions_1d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    exponential_func = test_functions_1d['exponential']

    result = integrator.integrate_interior(exponential_func['func'])
    expected = exponential_func['integral']

    assert jnp.allclose(result, expected, atol=exponential_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_constant_2d(config_quadrature_2d, test_functions_2d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_2d)
    const_func = test_functions_2d['constant']

    result = integrator.integrate_interior(const_func['func'])
    expected = const_func['integral']

    assert jnp.allclose(result, expected, atol=const_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_separable_2d(config_quadrature_2d, test_functions_2d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_2d)
    seperable_func = test_functions_2d['separable']

    result = integrator.integrate_interior(seperable_func['func'])
    expected = seperable_func['integral']

    assert jnp.allclose(result, expected, atol=seperable_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_product_sine_2d(config_quadrature_2d, test_functions_2d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_2d)
    product_sine_func = test_functions_2d['product_sine']

    result = integrator.integrate_interior(product_sine_func['func'])
    expected = product_sine_func['integral']

    assert jnp.allclose(result, expected, atol=product_sine_func['tolerance'])


@pytest.mark.quadrature
@pytest.mark.slow
def test_quadrature_constant_3d(config_quadrature_3d, test_functions_3d):
    """3D integration test: integral of 1 over [0,1]^3 should be 1."""
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_3d)
    const_func = test_functions_3d['constant']

    result = integrator.integrate_interior(const_func['func'])
    expected = const_func['integral']

    assert jnp.allclose(result, expected, atol=const_func['tolerance'])


@pytest.mark.quadrature
@pytest.mark.slow
def test_quadrature_separable_3d(config_quadrature_3d, test_functions_3d):
    """3D integration test: integral of x*y*z over [0,1]^3 should be 1/8."""
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_3d)
    separable_func = test_functions_3d['separable']

    result = integrator.integrate_interior(separable_func['func'])
    expected = separable_func['integral']

    assert jnp.allclose(result, expected, atol=separable_func['tolerance'])


@pytest.mark.quadrature
def test_quadrature_boundary_dirichlet_1d(config_quadrature_1d):
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)

    # Boundary function always returns 1
    boundary_func = lambda pts, normals: jnp.ones(pts.shape[0])

    result = integrator.integrate_boundary(boundary_func)
    # 1D cube has 2 points at x=0 and x=1, each with "length" 1
    assert jnp.allclose(result, 2.0, atol=1e-10)


@pytest.mark.quadrature
def test_quadrature_invalid_dim(config_quadrature_1d):
    from src.integration import QuadratureIntegration

    bad_config = replace(config_quadrature_1d, dim=-1)
    with pytest.raises(AssertionError, match="dim must be strictly positive"):
        QuadratureIntegration(bad_config)


@pytest.mark.quadrature
def test_quadrature_invalid_bounds(config_quadrature_1d):
    from src.integration import QuadratureIntegration

    bad_config = replace(config_quadrature_1d, x_min=1.0, x_max=0.0)
    with pytest.raises(AssertionError, match="x_min must be < x_max"):
        QuadratureIntegration(bad_config)


@pytest.mark.quadrature
def test_quadrature_invalid_degree(config_quadrature_1d):
    from src.integration import QuadratureIntegration

    bad_config = replace(config_quadrature_1d, gauss_legendre_degree=0)
    with pytest.raises(AssertionError, match="degree must be strictly positive"):
        QuadratureIntegration(bad_config)


@pytest.mark.quadrature
def test_quadrature_integrate_combined_1d(config_quadrature_1d, test_functions_1d):
    """Test that interior + boundary method combines correctly."""
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    const_func = test_functions_1d['constant']

    # Compute using combined method
    interior_func = const_func['func']
    boundary_func = lambda pts, normals: jnp.ones(pts.shape[0])

    total_loss, interior_loss, boundary_loss = integrator.integrate(
        interior_func, boundary_func
    )

    # Verify: total = interior + boundary
    assert jnp.allclose(total_loss, interior_loss + boundary_loss, atol=1e-14)

    # Verify interior matches direct call
    assert jnp.allclose(interior_loss, const_func['integral'], atol=const_func['tolerance'])

    # Verify boundary is 2 (two endpoints at x=0 and x=1)
    assert jnp.allclose(boundary_loss, 2.0, atol=1e-10)


@pytest.mark.quadrature
def test_quadrature_custom_bounds_negative(config_quadrature_1d):
    """Test integration over [-1, 1] domain."""
    from src.integration import QuadratureIntegration

    config = replace(config_quadrature_1d, x_min=-1.0, x_max=1.0)

    integrator = QuadratureIntegration(config)
    result = integrator.integrate_interior(lambda x: jnp.ones(x.shape[0]))
    expected = 2.0

    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.quadrature
def test_quadrature_custom_bounds_scaled(config_quadrature_1d):
    """Test integration over [0, 2] domain with scaled interval."""
    from src.integration import QuadratureIntegration

    config = replace(config_quadrature_1d, x_min=0.0, x_max=2.0)

    integrator = QuadratureIntegration(config)
    result = integrator.integrate_interior(lambda x: x[:, 0])
    expected = 2.0

    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.quadrature
def test_quadrature_boundary_2d(config_quadrature_2d):
    """Test boundary integration over [0,1]^2."""
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_2d)

    # Boundary function always returns 1
    boundary_func = lambda pts, normals: jnp.ones(pts.shape[0])

    result = integrator.integrate_boundary(boundary_func)

    # 2D cube has 4 boundaries: 4 edges each with "length" 1
    # Total boundary measure: 4
    assert jnp.allclose(result, 4.0, atol=1e-10)


@pytest.mark.quadrature
def test_quadrature_boundary_normals_2d(config_quadrature_2d):
    """Test that boundary normals are computed correctly for 2D."""
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_2d)

    # Function returns x-component of outward normal
    boundary_func = lambda pts, normals: normals[:, 0]

    result = integrator.integrate_boundary(boundary_func)

    # Total should be 0 due to cancellation (up to numerical precision)
    assert jnp.allclose(result, 0.0, atol=1e-7)
