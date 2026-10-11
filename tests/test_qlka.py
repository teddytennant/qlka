import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from qlka import (
    DEFAULT_BUMP_TAU,
    DEFAULT_LAPLACE_TAU,
    DEFAULT_RIESZ_TAU,
    RIESZ_EPS,
    abs_sign_sum,
    add_bump_attention,
    add_bump_sum,
    add_laplace_attention,
    add_laplace_sum,
    add_riesz_attention,
    add_riesz_sum,
    additive_abs_sum,
    causal_abs_sum,
    causal_laplace_sum,
    phi_bump,
    phi_laplace,
    phi_riesz,
    relu_gram,
    weighted_abs_sum,
    weighted_bump_sum,
    weighted_laplace_prefix_sum,
    weighted_laplace_sum,
    weighted_riesz_sum,
)

jax.config.update("jax_enable_x64", True)


def _check(actual, expected, atol=1e-8, rtol=1e-8):
    actual = jnp.asarray(actual)
    expected = jnp.asarray(expected)
    if not jnp.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = jnp.max(jnp.abs(actual - expected))
        raise AssertionError(f"max abs diff {float(diff)}\nactual {actual}\nexpected {expected}")


def _naive_abs(s, t, v):
    return jnp.sum(jnp.abs(s[:, None] - t[None, :])[:, :, None] * v[None, :, :], axis=1)


def _naive_laplace(s, t, v):
    weights = jnp.exp(-jnp.abs(s[:, None] - t[None, :]))
    return weights @ v


def _gram(q, k, phi):
    g = jnp.zeros((q.shape[0], k.shape[0]), dtype=q.dtype)
    for d in range(q.shape[-1]):
        g = g + phi(q[:, d][:, None], k[:, d][None, :])
    return g


def _brute_attention(q, k, v, phi):
    g = _gram(q, k, phi)
    denom = jnp.sum(g, axis=1, keepdims=True)
    positive = denom > 0
    denom_safe = jnp.where(positive, denom, jnp.ones_like(denom))
    a = jnp.where(positive, g / denom_safe, jnp.zeros_like(g))
    return a @ v, a, g


def _separated(s, t, gap=1e-3):
    return float(jnp.min(jnp.abs(s[:, None] - t[None, :]))) > gap


def test_runs_on_cpu():
    assert jax.default_backend() == "cpu"


def test_default_bandwidths():
    assert DEFAULT_LAPLACE_TAU == 0.5
    assert DEFAULT_BUMP_TAU == 1.5
    assert DEFAULT_RIESZ_TAU == 1.0
    assert RIESZ_EPS == 1e-3
    q = jnp.array([[0.0, 1.0]], dtype=jnp.float64)
    k = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    v = jnp.ones((1, 1), dtype=jnp.float64)
    z = add_laplace_sum(q, k, v)
    _check(z, add_laplace_sum(q, k, v, tau=0.5))
    assert float(jnp.max(jnp.abs(z - add_laplace_sum(q, k, v, tau=6.0)))) > 1e-3
    zb = add_bump_sum(q, k, v)
    _check(zb, add_bump_sum(q, k, v, tau=1.5))
    assert float(jnp.max(jnp.abs(zb - add_bump_sum(q, k, v, tau=0.5)))) > 1e-3


def test_algorithm1_matches_naive_with_tie():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    m, n, c = 8, 13, 4
    s = jax.random.normal(k1, (m,), dtype=jnp.float64)
    t = jax.random.normal(k2, (n,), dtype=jnp.float64)
    v = jax.random.normal(k3, (n, c), dtype=jnp.float64)
    # Several keys equal to one query, and two keys equal to each other.
    t = t.at[1].set(s[0]).at[4].set(s[0]).at[7].set(s[0]).at[9].set(t[2])
    _check(weighted_abs_sum(s, t, v), _naive_abs(s, t, v))

    # Hand example with a repeated key and C = 2.
    s_h = jnp.array([0.0, 2.0])
    t_h = jnp.array([1.0, -1.0, 1.0])
    v_h = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = jnp.array([[2.0, 2.0], [2.0, 4.0]])
    _check(weighted_abs_sum(s_h, t_h, v_h), expected)
    _check(weighted_abs_sum(s_h, t_h, v_h), _naive_abs(s_h, t_h, v_h))


def test_equation10_prefix_identity():
    """Equation (10) on prefix arrays of t_sigma * v_sigma and of v_sigma."""
    key = jax.random.PRNGKey(10)
    k1, k2, k3 = jax.random.split(key, 3)
    s = jax.random.normal(k1, (6,), dtype=jnp.float64)
    t = jax.random.normal(k2, (9,), dtype=jnp.float64)
    v = jax.random.normal(k3, (9, 3), dtype=jnp.float64)
    t = t.at[0].set(s[2]).at[5].set(s[2])

    sigma = jnp.argsort(t)
    t_s = t[sigma]
    v_s = v[sigma]
    pref_tv = jnp.concatenate(
        [jnp.zeros((1, 3)), jnp.cumsum(t_s[:, None] * v_s, axis=0)], axis=0
    )
    pref_v = jnp.concatenate([jnp.zeros((1, 3)), jnp.cumsum(v_s, axis=0)], axis=0)
    p = jnp.searchsorted(t_s, s, side="right")
    a = 2.0 * pref_v[p] - pref_v[-1]
    z = pref_tv[-1] - 2.0 * pref_tv[p] + s[:, None] * a

    _check(z, _naive_abs(s, t, v))
    _check(weighted_abs_sum(s, t, v), z)
    # p includes every tie, so a is the sign sum with sign(0) = +1.
    _check(a, abs_sign_sum(s, t, v))


def test_laplace_sort_sum_matches_naive():
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    s = jax.random.normal(k1, (7,), dtype=jnp.float64)
    t = jax.random.normal(k2, (11,), dtype=jnp.float64)
    v = jax.random.normal(k3, (11, 3), dtype=jnp.float64)
    t = t.at[3].set(s[1]).at[8].set(s[1])
    naive = _naive_laplace(s, t, v)
    _check(weighted_laplace_prefix_sum(s, t, v), naive)
    _check(weighted_laplace_sum(s, t, v), naive)

    # Separated points: origin-centered exp(t) overflows, pairwise exp(-|s-t|) does not.
    s_far = jnp.array([0.0, 800.0], dtype=jnp.float64)
    t_far = jnp.array([0.0, 400.0, 800.0], dtype=jnp.float64)
    v_far = jnp.array([[1.0], [-2.0], [0.5]], dtype=jnp.float64)
    _check(weighted_laplace_sum(s_far, t_far, v_far), _naive_laplace(s_far, t_far, v_far))
    assert not jnp.all(jnp.isfinite(weighted_laplace_prefix_sum(s_far, t_far, v_far)))


def test_additive_kernels_and_attention_match_brute_force():
    key = jax.random.PRNGKey(4)
    kq, kk, kv = jax.random.split(key, 3)
    d, m, n, c = 4, 7, 11, 3
    q = jax.random.normal(kq, (m, d), dtype=jnp.float64)
    k = jax.random.normal(kk, (n, d), dtype=jnp.float64)
    v = jax.random.normal(kv, (n, c), dtype=jnp.float64)

    cases = [
        (add_laplace_sum, add_laplace_attention, phi_laplace, {}),
        (add_bump_sum, add_bump_attention, phi_bump, {}),
        (add_riesz_sum, add_riesz_attention, phi_riesz, {}),
        # Non-default bandwidths, so scaling is not a no-op.
        (add_laplace_sum, add_laplace_attention, phi_laplace, {"tau": 1.5}),
        (add_bump_sum, add_bump_attention, phi_bump, {"tau": 1.25}),
        (add_riesz_sum, add_riesz_attention, phi_riesz, {"tau": 2.0, "eps": 1e-3}),
    ]
    for kernel_sum, attend, phi, kwargs in cases:
        def univariate(s, t, phi=phi, kwargs=kwargs):
            return phi(s, t, **kwargs)

        brute_y, _, gram = _brute_attention(q, k, v, univariate)
        _check(kernel_sum(q, k, v, **kwargs), gram @ v)
        _check(attend(q, k, v, **kwargs), brute_y)


def test_bump_identity_with_algorithm1():
    for tau in (0.5, 1.0, 1.5):
        u = jnp.linspace(-3.0, 3.0, 31, dtype=jnp.float64)
        lhs = jnp.maximum(0.0, 1.0 - jnp.abs(u) / tau)
        rhs = (1.0 / (2.0 * tau)) * (
            jnp.abs(u + tau) + jnp.abs(u - tau) - 2.0 * jnp.abs(u)
        )
        _check(lhs, rhs, atol=1e-12, rtol=1e-12)

    tau = 0.5
    s = jnp.array([0.2, -0.4, 1.5, 0.0], dtype=jnp.float64)
    t = jnp.array([0.0, 0.7, -1.2, 3.0, 0.2], dtype=jnp.float64)
    u = s[:, None] - t[None, :]
    rhs = (1.0 / (2.0 * tau)) * (jnp.abs(u + tau) + jnp.abs(u - tau) - 2.0 * jnp.abs(u))
    _check(phi_bump(s[:, None], t[None, :], tau=tau), rhs, atol=1e-12, rtol=1e-12)
    v = jax.random.normal(jax.random.PRNGKey(5), (t.shape[0], 2), dtype=jnp.float64)
    _check(weighted_bump_sum(s, t, v, tau=tau), rhs @ v)


def test_bump_zero_row_is_zero():
    q = jnp.array([[0.0, 0.0], [20.0, 20.0]], dtype=jnp.float64)
    k = jnp.array([[0.1, -0.2], [0.0, 0.3], [-0.2, 0.1]], dtype=jnp.float64)
    v = jnp.array([[3.0, -1.0], [0.5, 2.0], [-4.0, 0.25]], dtype=jnp.float64)
    y = add_bump_attention(q, k, v, tau=0.5)
    brute_y, _, gram = _brute_attention(
        q, k, v, lambda s, t: phi_bump(s, t, tau=0.5)
    )
    assert float(jnp.sum(gram[1])) == 0.0
    assert float(jnp.sum(gram[0])) > 0.0
    _check(y[1], jnp.zeros((2,), dtype=jnp.float64))
    _check(y, brute_y)


def test_capacity_witnesses():
    """Closed-form capacity witness, not a reproduction of Figure 1.

    Laplace and bump in 1D separate N = 12 points, more than the ReLU
    feature dimension D = 3 checked below. Laplace uses tau = 1 so that
    L = 40 actually separates; the package default tau is 6.
    """
    finite_d = 3
    n_finite = 8
    key = jax.random.PRNGKey(6)
    kq, kk = jax.random.split(key)
    q = jnp.abs(jax.random.normal(kq, (n_finite, finite_d), dtype=jnp.float64)) + 0.2
    k = jnp.abs(jax.random.normal(kk, (n_finite, finite_d), dtype=jnp.float64)) + 0.2
    gram = relu_gram(q, k)
    rank = int(jnp.linalg.matrix_rank(gram))
    assert rank <= finite_d
    assert jnp.all(jnp.sum(gram, axis=1) > 0.0)
    denom = jnp.sum(gram, axis=1, keepdims=True)
    attention = gram / denom
    err = float(jnp.sum((attention - jnp.eye(n_finite, dtype=jnp.float64)) ** 2))
    assert err >= n_finite - rank - 1e-5
    assert err >= n_finite - finite_d - 1e-6

    # Mixed signs can zero a ReLU coordinate. Rank is still at most D.
    # Do not repair a zero row and then test A.
    q_mix = jax.random.normal(jax.random.PRNGKey(7), (9, finite_d), dtype=jnp.float64)
    k_mix = jax.random.normal(jax.random.PRNGKey(8), (9, finite_d), dtype=jnp.float64)
    assert int(jnp.linalg.matrix_rank(relu_gram(q_mix, k_mix))) <= finite_d

    n_inf = 12
    assert n_inf > finite_d
    eye = jnp.eye(n_inf, dtype=jnp.float64)

    laplace_pos = (jnp.arange(n_inf, dtype=jnp.float64) * 40.0)[:, None]
    y_lap = add_laplace_attention(laplace_pos, laplace_pos, eye, tau=1.0)
    lap_err = float(jnp.sum((y_lap - eye) ** 2))
    assert lap_err < 1e-6, lap_err

    bump_pos = (jnp.arange(n_inf, dtype=jnp.float64) * 8.0)[:, None]
    y_bump = add_bump_attention(bump_pos, bump_pos, eye, tau=1.0)
    bump_err = float(jnp.sum((y_bump - eye) ** 2))
    assert bump_err < 1e-6, bump_err


def test_causal_algorithm2_matches_masked_brute_force():
    def brute(s, t, v, pairwise):
        weights = pairwise(s[:, None], t[None, :])
        mask = jnp.arange(t.shape[0])[None, :] <= jnp.arange(s.shape[0])[:, None]
        return jnp.where(mask, weights, 0.0) @ v

    key = jax.random.PRNGKey(9)
    k1, k2, k3 = jax.random.split(key, 3)
    n, c = 8, 3
    s = jax.random.normal(k1, (n,), dtype=jnp.float64)
    t = jax.random.normal(k2, (n,), dtype=jnp.float64)
    v = jax.random.normal(k3, (n, c), dtype=jnp.float64)
    t = t.at[2].set(s[5]).at[6].set(t[1])
    _check(causal_abs_sum(s, t, v), brute(s, t, v, lambda a, b: jnp.abs(a - b)))
    _check(
        causal_laplace_sum(s, t, v),
        brute(s, t, v, lambda a, b: jnp.exp(-jnp.abs(a - b))),
    )

    # N = 1 is the masked base case, including a visible diagonal pair.
    s1 = jnp.array([0.3], dtype=jnp.float64)
    t1 = jnp.array([1.7], dtype=jnp.float64)
    v1 = jnp.array([[2.0, -1.0]], dtype=jnp.float64)
    _check(causal_abs_sum(s1, t1, v1), brute(s1, t1, v1, lambda a, b: jnp.abs(a - b)))

    # Non-power-of-two falls back to the same mask.
    s6 = s[:6]
    t6 = t[:6]
    v6 = v[:6]
    _check(causal_abs_sum(s6, t6, v6), brute(s6, t6, v6, lambda a, b: jnp.abs(a - b)))

    jit_abs = jax.jit(causal_abs_sum)
    _check(jit_abs(s, t, v), brute(s, t, v, lambda a, b: jnp.abs(a - b)))


def test_sign_convention_and_grad_away_from_ties():
    # sign(0) = +1. Three tied keys, v sums to 6.
    s_tie = jnp.array([1.0], dtype=jnp.float64)
    t_tie = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
    v_tie = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float64)
    _check(abs_sign_sum(s_tie, t_tie, v_tie), jnp.array([[6.0]]))
    _check(weighted_abs_sum(s_tie, t_tie, v_tie), jnp.array([[0.0]]))
    # jnp.sign(0) is 0, but this JAX abs VJP at 0 is +1, so the naive
    # gradient at a total tie equals the paper's sign(0) = +1 convention.
    naive_tie_grad = jax.grad(lambda s: jnp.sum(_naive_abs(s, t_tie, v_tie)))(s_tie)
    _check(naive_tie_grad, jnp.array([6.0]))

    s = jnp.array([1.0, -2.0], dtype=jnp.float64)
    t = jnp.array([0.0, 3.0, -1.0], dtype=jnp.float64)
    v = jnp.array([[1.0, 0.5], [0.0, -1.0], [2.0, 3.0]], dtype=jnp.float64)
    expected_sign = jnp.array([[3.0, 4.5], [-3.0, -2.5]], dtype=jnp.float64)
    _check(abs_sign_sum(s, t, v), expected_sign)
    naive_grad = jax.grad(lambda ss: jnp.sum(_naive_abs(ss, t, v)))(s)
    _check(naive_grad, jnp.sum(expected_sign, axis=-1))

    key = jax.random.PRNGKey(12)
    ks, kt, kv, kg = jax.random.split(key, 4)
    s_r = jax.random.normal(ks, (6,), dtype=jnp.float64)
    t_r = jax.random.normal(kt, (9,), dtype=jnp.float64) + 0.17
    v_r = jax.random.normal(kv, (9, 3), dtype=jnp.float64)
    g = jax.random.normal(kg, (6, 3), dtype=jnp.float64)
    assert _separated(s_r, t_r)

    sign = abs_sign_sum(s_r, t_r, v_r)
    grad_s = jax.grad(lambda ss: jnp.sum(_naive_abs(ss, t_r, v_r)))(s_r)
    _check(grad_s, jnp.sum(sign, axis=-1))

    def j_naive(ss, tt, vv):
        return jnp.sum(g * _naive_abs(ss, tt, vv))

    def j_alg(ss, tt, vv):
        return jnp.sum(g * weighted_abs_sum(ss, tt, vv))

    naive_grads = jax.grad(j_naive, argnums=(0, 1, 2))(s_r, t_r, v_r)
    alg_grads = jax.grad(j_alg, argnums=(0, 1, 2))(s_r, t_r, v_r)
    for actual, expected in zip(alg_grads, naive_grads):
        _check(actual, expected)


def test_equation26_vjp_matches_abs_weights():
    key = jax.random.PRNGKey(26)
    kq, kk, kv, kg = jax.random.split(key, 4)
    d, m, n, c = 4, 5, 6, 3
    q = jax.random.normal(kq, (m, d), dtype=jnp.float64)
    k = jax.random.normal(kk, (n, d), dtype=jnp.float64)
    v = jax.random.normal(kv, (n, c), dtype=jnp.float64)
    g = jax.random.normal(kg, (m, c), dtype=jnp.float64)
    for d_idx in range(d):
        assert _separated(q[:, d_idx], k[:, d_idx])

    def loss(values):
        return jnp.sum(g * additive_abs_sum(q, k, values))

    vjp = jax.grad(loss)(v)
    # Equation (26), and the paper's swap of queries with keys.
    expected = jnp.zeros_like(v)
    for d_idx in range(d):
        weights = jnp.abs(q[:, d_idx][:, None] - k[:, d_idx][None, :])
        expected = expected + weights.T @ g
    _check(vjp, expected)
    _check(vjp, additive_abs_sum(k, q, g))


def test_lemma11_permuted_basis_loss():
    n, d = 5, 3
    key = jax.random.PRNGKey(11)
    kq, kk, ks, kx = jax.random.split(key, 4)
    q = jax.random.normal(kq, (n, d), dtype=jnp.float64)
    k = jax.random.normal(kk, (n, d), dtype=jnp.float64)
    sigma = jax.random.permutation(ks, n)
    xi = jax.random.permutation(kx, n)
    eye = jnp.eye(n, dtype=jnp.float64)
    gram = add_laplace_sum(q, k, eye, tau=1.0)
    attention = gram / jnp.sum(gram, axis=1, keepdims=True)
    values = eye[xi]
    y = add_laplace_attention(q, k[sigma], values, tau=1.0)
    lhs = jnp.sum((eye[xi] - y[sigma]) ** 2)
    rhs = jnp.sum((attention - eye) ** 2)
    _check(lhs, rhs)

    # Unpermuted special case: y = A when values are the identity.
    y_id = add_laplace_attention(q, k, eye, tau=1.0)
    _check(jnp.sum((eye - y_id) ** 2), jnp.sum((attention - eye) ** 2))


def test_vmap_matches_rowwise():
    key = jax.random.PRNGKey(13)
    ks, kt, kv = jax.random.split(key, 3)
    s = jax.random.normal(ks, (3, 5), dtype=jnp.float64)
    t = jax.random.normal(kt, (3, 6), dtype=jnp.float64)
    v = jax.random.normal(kv, (3, 6, 2), dtype=jnp.float64)
    batched = jax.vmap(weighted_abs_sum)(s, t, v)
    for b in range(3):
        _check(batched[b], weighted_abs_sum(s[b], t[b], v[b]))

    batched_l = jax.vmap(weighted_laplace_sum)(s, t, v)
    for b in range(3):
        _check(batched_l[b], weighted_laplace_sum(s[b], t[b], v[b]))


def test_univariate_riesz_matches_weighted_sum():
    s = jnp.array([0.3, -1.2, 2.0], dtype=jnp.float64)
    t = jnp.array([-0.4, 0.5, 1.1, -2.0], dtype=jnp.float64)
    v = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.25, -0.25]], dtype=jnp.float64)
    weights = phi_riesz(s[:, None], t[None, :], tau=1.0, eps=1e-3)
    _check(weighted_riesz_sum(s, t, v, tau=1.0, eps=1e-3), weights @ v)


def test_jit_abs_and_attention():
    key = jax.random.PRNGKey(14)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (7, 4), dtype=jnp.float64)
    k = jax.random.normal(kk, (11, 4), dtype=jnp.float64)
    v = jax.random.normal(kv, (11, 3), dtype=jnp.float64)
    _check(jax.jit(add_laplace_sum)(q, k, v), add_laplace_sum(q, k, v))
    _check(jax.jit(add_bump_attention)(q, k, v), add_bump_attention(q, k, v))
    _check(jax.jit(weighted_abs_sum)(q[:, 0], k[:, 0], v), weighted_abs_sum(q[:, 0], k[:, 0], v))
