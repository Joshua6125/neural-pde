import jax.numpy as jnp
import pytest


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
