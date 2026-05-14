"""Tests for DRMLoss."""

import jax.numpy as jnp
import pytest

from src.loss_functions.drm import DRMLoss


pytestmark = pytest.mark.DRM


class TestDRMLossInitialisation:
    def test_init_stores_fields(self):
        def u_fn(x):
            return jnp.asarray(x[0])

        loss = DRMLoss(u_model=u_fn, boundary_weight=2.0)
        assert loss.u_model is u_fn
        assert loss.boundary_weight == 2.0


class TestDRMLossInterior:
    def test_interior_energy_linear_model(self, mock_model_valid):
        loss = DRMLoss(u_model=lambda x: jnp.asarray(x[0] + 2.0 * x[1]), A=1.0, c=0.0, f=0.0)
        result = loss.loss_interior(jnp.array([[0.5, 0.25]]))
        assert jnp.allclose(result[0], 2.5)

    def test_interior_energy_zero_model(self):
        loss = DRMLoss(u_model=lambda x: jnp.asarray(0.0), A=1.0, c=0.0, f=0.0)
        result = loss.loss_interior(jnp.array([[0.5, 0.25]]))
        assert jnp.allclose(result[0], 0.0)

    def test_interior_accepts_matrix_A(self):
        loss = DRMLoss(
            u_model=lambda x: jnp.asarray(x[0]),
            A=lambda x: jnp.eye(x.shape[0]),
            c=0.0,
            f=0.0,
        )
        result = loss.loss_interior(jnp.array([[0.5, 0.25]]))
        assert jnp.isfinite(result[0])


class TestDRMLossBoundary:
    def test_boundary_penalty_zero_when_matching_g(self):
        loss = DRMLoss(u_model=lambda x: jnp.asarray(0.0), g=0.0, boundary_weight=3.0)
        points = jnp.array([[0.0, 0.25], [0.0, 0.75]])
        normals = jnp.array([[-1.0, 0.0], [-1.0, 0.0]])
        result = loss.loss_boundary(points, normals)
        assert jnp.allclose(result, 0.0)

    def test_boundary_penalty_uses_weight(self):
        loss = DRMLoss(u_model=lambda x: jnp.asarray(1.0), g=0.0, boundary_weight=2.0)
        points = jnp.array([[0.0, 0.25]])
        normals = jnp.array([[-1.0, 0.0]])
        result = loss.loss_boundary(points, normals)
        assert jnp.allclose(result[0], 2.0)
