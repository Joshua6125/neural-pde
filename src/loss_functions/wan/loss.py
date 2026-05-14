"""Weak Adversarial Network loss for the first-order acoustic wave system."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class WANLoss(Loss):
    """Weak Adversarial Network loss for the wave equation.

    The minimax functional is:
        min_{v, sigma} max_{phi,spi} [ R_v(v, sigma,phi)^2 + R_sigma(v,sigma,psi)^2 ]

    where R_v and R_sigma are the weak residuals of the two equations.

    This class computes the INNER quantity (given fixed φ,ψ from the
    adversarial network) that the trial network minimises. The adversarial
    network maximises the same quantity with negated sign.

    The training loop must alternate:
        1. Maximise over (phi_model, psi_model) — adversarial step
        2. Minimise over (v_model, sigma_model) — trial step

    Parameters
    ----------
    v_model : Callable
        Trial network v(x) -> scalar.
    sigma_model : Callable
        Trial network sigma(x) -> [d].
    phi_model : Callable
        Adversarial test function φ(x) -> scalar (for v equation).
    psi_model : Callable
        Adversarial test function ψ(x) -> [d] (for σ equation).
    f : Callable or float
        Source term for v equation.
    g : Callable or float
        Source term for sigma equation.
    v0 : Callable or float
        Initial condition v(0,-) = v0.
    sigma0 : Callable or float
        Initial condition sigma(0,-) = sigma0.
    ic_weight : float
        Weight for initial condition terms in the weak form.
    """

    def __init__(
        self,
        v_model: Callable[[jnp.ndarray], jnp.ndarray],
        sigma_model: Callable[[jnp.ndarray], jnp.ndarray],
        phi_model: Callable[[jnp.ndarray], jnp.ndarray],
        psi_model: Callable[[jnp.ndarray], jnp.ndarray],
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ic_weight: float = 1.0,
    ):
        self.v_model = v_model
        self.sigma_model = sigma_model
        self.phi_model = phi_model
        self.psi_model = psi_model
        self.ic_weight = ic_weight

        self._f_fn = f if callable(f) else self._constant_function(f)
        self._g_fn = g if callable(g) else self._constant_function(g)
        self._v0_fn = v0 if callable(v0) else self._constant_function(v0)
        self._sigma0_fn = sigma0 if callable(sigma0) else self._constant_function(sigma0)

        self._vmapped_interior = jax.vmap(self._interior_integrand)
        self._vmapped_ic = jax.vmap(self._ic_integrand)

    def _v(self, x):
        return self.v_model(x).squeeze()

    def _sigma(self, x):
        return self.sigma_model(x).reshape(-1)

    def _phi(self, x):
        return self.phi_model(x).squeeze()

    def _psi(self, x):
        return self.psi_model(x).reshape(-1)

    def _interior_integrand(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pointwise integrand of the weak residual in the interior.

        For the v-equation (integrated by parts):
            v * ∂_t φ + σ · ∇_x φ - f * φ

        For the σ-equation (integrated by parts):
            σ · ∂_t ψ + v * div_x ψ - g · ψ

        Note: derivatives act on test functions φ,ψ — NOT on v,σ.
        This is the fundamental difference from FOSLS/SLS.
        """
        v = self._v(x)
        sigma = self._sigma(x)
        f = self._f_fn(x)
        g = self._g_fn(x)

        # Gradients of TEST functions — cheap, no second-order derivatives
        phi_grad = jax.grad(self._phi)(x)
        dt_phi = phi_grad[0]
        grad_x_phi = phi_grad[1:]

        J_psi = jax.jacobian(self._psi)(x)
        dt_psi = J_psi[:, 0]
        div_x_psi = jnp.trace(J_psi[:, 1:])

        if jnp.ndim(g) == 0:
            g = g * jnp.ones_like(sigma)

        # Weak residual integrands
        res_v = v * dt_phi + jnp.dot(sigma, grad_x_phi) - f * self._phi(x)
        res_sigma = jnp.dot(sigma, dt_psi) + v * div_x_psi - jnp.dot(g, self._psi(x))

        return res_v, res_sigma

    def _ic_integrand(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Boundary terms from integration by parts at t=0.

        After IBP in time the boundary term at t=0 is:
            -v(0) * φ(0) for the v-equation
            -σ(0) · ψ(0) for the σ-equation

        These should equal -v0*φ(0) and -σ0·ψ(0) respectively,
        so the IC residual contribution is:
            (v(0) - v0) * φ(0)  and  (σ(0) - σ0) · ψ(0)
        """
        v_val = self._v(x)
        sigma_val = self._sigma(x)
        phi_val = self._phi(x)
        psi_val = self._psi(x)

        v0_val = self._v0_fn(x)
        sigma0_val = self._sigma0_fn(x)
        if jnp.ndim(sigma0_val) == 0:
            sigma0_val = sigma0_val * jnp.ones_like(sigma_val)

        ic_v = (v_val - v0_val) * phi_val
        ic_sigma = jnp.dot(sigma_val - sigma0_val, psi_val)

        return ic_v, ic_sigma

    def weak_residuals(
        self,
        x_interior: jnp.ndarray,
        x_ic: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the two scalar weak residuals R_v and R_σ.

        These are Monte Carlo estimates of the integrals:
            R_v = ∫_Q (v ∂_t φ + σ·∇φ - fφ) dQ + ic_weight * ∫_Ω (v(0)-v0)φ(0) dΩ
            R_σ = ∫_Q (σ·∂_t ψ + v div ψ - g·ψ) dQ + ic_weight * ∫_Ω (σ(0)-σ0)·ψ(0) dΩ

        The Monte Carlo estimate of ∫_D h dD ≈ vol(D) * mean(h over samples).
        The volume factor should be handled by your sampler/trainer, so
        here we just return the means.
        """
        int_v, int_sigma = self._vmapped_interior(x_interior)
        ic_v, ic_sigma = self._vmapped_ic(x_ic)

        R_v = jnp.mean(int_v) + self.ic_weight * jnp.mean(ic_v)
        R_sigma = jnp.mean(int_sigma) + self.ic_weight * jnp.mean(ic_sigma)

        return R_v, R_sigma

    def loss_for_trial(
        self,
        x_interior: jnp.ndarray,
        x_ic: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Loss for the TRIAL network (v, sigma): minimise R_v^2 + R_sigma^2.
        Call this when updating v_model and sigma_model parameters.
        """
        R_v, R_sigma = self.weak_residuals(x_interior, x_ic)
        return jnp.square(R_v) + jnp.square(R_sigma)

    def loss_for_adversary(
        self,
        x_interior: jnp.ndarray,
        x_ic: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Loss for the ADVERSARIAL network (φ, ψ): maximise R_v² + R_sigma^2,
        i.e. minimise -(R_v² + R_sigma^2).
        Call this when updating phi_model and psi_model parameters.
        """
        return -self.loss_for_trial(x_interior, x_ic)

    # ------------------------------------------------------------------
    # Base class interface — interior/boundary split is less natural for
    # WAN since the weak form integrates everything together. We expose
    # the standard interface for compatibility but the real entry points
    # are loss_for_trial and loss_for_adversary above.
    # ------------------------------------------------------------------

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "WAN does not decompose cleanly into interior/boundary losses. "
            "Use loss_for_trial() and loss_for_adversary() directly in your trainer."
        )

    def loss_boundary(self, x_boundary: jnp.ndarray, normal_vector: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "WAN does not decompose cleanly into interior/boundary losses. "
            "Use loss_for_trial() and loss_for_adversary() directly in your trainer."
        )
