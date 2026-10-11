"""Sorting formulas for quasi-linear additive kernel attention.

Unbatched shapes: s is (M,), t is (N,), v is (N, C), queries are (M, D),
keys are (N, D). Batch a leading axis with jax.vmap. The argsort is treated
as locally constant (stop-gradient on the permutation), which is the VJP
used for the absolute-value sum away from ties.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Phi_tau(q, k) = Phi(q / tau, k / tau).
# Defaults: add_laplace tau=6, add_bump tau=0.5, add_riesz tau=1, eps=1e-3.
# Figure 1 brackets add_laplace at 0.5 and add_bump at 1.5; pass tau to match.
DEFAULT_LAPLACE_TAU = 6.0
DEFAULT_BUMP_TAU = 0.5
DEFAULT_RIESZ_TAU = 1.0
RIESZ_EPS = 1e-3


def _as_stv(s, t, v):
    s = jnp.asarray(s)
    t = jnp.asarray(t)
    v = jnp.asarray(v)
    if s.ndim != 1 or t.ndim != 1 or v.ndim != 2:
        raise ValueError("expected s shape (M,), t shape (N,), v shape (N, C)")
    if t.shape[0] != v.shape[0]:
        raise ValueError("t and v must have the same key length")
    return s, t, v


def _as_qkv(q, k, v):
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("expected q (M, D), k (N, D), v (N, C)")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share the feature axis")
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same key length")
    return q, k, v


def _argsort(t):
    return jax.lax.stop_gradient(jnp.argsort(t))


def _split_count(t_sorted, s):
    """p_m = |{n : t_sorted[n] <= s_m}|. Ties sit on the left, so sign(0) = +1."""
    return jax.lax.stop_gradient(
        jnp.searchsorted(jax.lax.stop_gradient(t_sorted), s, side="right")
    )


def _prefix_pad(x):
    """Prefix sums with a leading 0. x is (N, C), result is (N + 1, C)."""
    zero = jnp.zeros((1, x.shape[-1]), dtype=x.dtype)
    return jnp.concatenate([zero, jnp.cumsum(x, axis=0)], axis=0)


def weighted_abs_sum(s, t, v):
    """Algorithm 1. z_m = sum_n |s_m - t_n| v_n.

    After sorting t, with p_m the number of keys <= s_m and
    pref_tv the prefix of t_sigma * v_sigma (equation 10):

        a_m = 2 pref_v[p_m] - pref_v[N]
        z_m = pref_tv[N] - 2 pref_tv[p_m] + s_m * a_m
    """
    s, t, v = _as_stv(s, t, v)
    sigma = _argsort(t)
    t_sorted = t[sigma]
    v_sorted = v[sigma]
    p = _split_count(t_sorted, s)
    pref_tv = _prefix_pad(t_sorted[:, None] * v_sorted)
    pref_v = _prefix_pad(v_sorted)
    a = 2.0 * pref_v[p] - pref_v[-1]
    return pref_tv[-1] - 2.0 * pref_tv[p] + s[:, None] * a


def abs_sign_sum(s, t, v):
    """Equation (25): sum_n sign(s_m - t_n) v_n = 2 pref_v[p_m] - pref_v[N].

    Convention sign(0) = +1, because p_m counts keys with t_n <= s_m.
    Away from ties this is the derivative of the absolute-value sum in s.
    jax.numpy.sign(0) is 0, but this JAX build's abs VJP at 0 is +1, so the
    naive abs gradient matches the convention at ties as well.
    """
    s, t, v = _as_stv(s, t, v)
    sigma = _argsort(t)
    v_sorted = v[sigma]
    p = _split_count(t[sigma], s)
    pref_v = _prefix_pad(v_sorted)
    return 2.0 * pref_v[p] - pref_v[-1]


def _laplace_left_right(t_sorted, v_sorted):
    """Relative prefix and suffix of Algorithm 3.

    L[n] = sum_{j <= n} exp(-(t[n] - t[j])) v[j]
    R[n] = sum_{j >= n} exp(-(t[j] - t[n])) v[j]
    """
    n = t_sorted.shape[0]
    c = v_sorted.shape[-1]
    zero_c = jnp.zeros((c,), dtype=v_sorted.dtype)
    deltas = jnp.zeros((n,), dtype=t_sorted.dtype)
    rev_decay = jnp.zeros((n,), dtype=t_sorted.dtype)
    if n > 1:
        gaps = jnp.diff(t_sorted)
        deltas = deltas.at[1:].set(gaps)
        rev_decay = rev_decay.at[1:].set(gaps[::-1])

    def step(carry, xs):
        decay, value = xs
        nxt = jnp.exp(-decay) * carry + value
        return nxt, nxt

    _, left = jax.lax.scan(step, zero_c, (deltas, v_sorted))
    _, right_rev = jax.lax.scan(step, zero_c, (rev_decay, v_sorted[::-1]))
    return left, right_rev[::-1]


def weighted_laplace_sum(s, t, v):
    """z_m = sum_n exp(-|s_m - t_n|) v_n.

    Same split as the origin-centered prefixes

        exp(-s) * sum_{t_n <= s} exp(t_n) v_n
        + exp(s) * sum_{t_n > s} exp(-t_n) v_n

    but accumulated relative to the current sorted node (Algorithm 3), so
    every exponential lies in (0, 1]. The origin-centered form is
    weighted_laplace_prefix_sum; it overflows once exp(t) overflows.
    """
    s, t, v = _as_stv(s, t, v)
    n = t.shape[0]
    sigma = _argsort(t)
    t_sorted = t[sigma]
    v_sorted = v[sigma]
    left, right = _laplace_left_right(t_sorted, v_sorted)
    p = _split_count(t_sorted, s)
    # Clip only to keep the unused side in-bounds. That side is multiplied by 0.
    left_idx = jnp.clip(p - 1, 0, n - 1)
    right_idx = jnp.clip(p, 0, n - 1)
    left_coef = jnp.where(p > 0, jnp.exp(-(s - t_sorted[left_idx])), 0.0)
    right_coef = jnp.where(p < n, jnp.exp(-(t_sorted[right_idx] - s)), 0.0)
    return left_coef[:, None] * left[left_idx] + right_coef[:, None] * right[right_idx]


def weighted_laplace_prefix_sum(s, t, v):
    """Origin-centered Laplace prefixes. Matches weighted_laplace_sum when finite.

    z_m = exp(-s_m) * sum_{n=1}^{p_m} exp(t_sigma(n)) v_sigma(n)
        + exp(s_m) * sum_{n=p_m+1}^{N} exp(-t_sigma(n)) v_sigma(n)
    """
    s, t, v = _as_stv(s, t, v)
    sigma = _argsort(t)
    t_sorted = t[sigma]
    v_sorted = v[sigma]
    p = _split_count(t_sorted, s)
    left_pref = _prefix_pad(jnp.exp(t_sorted)[:, None] * v_sorted)
    right_pref = _prefix_pad(jnp.exp(-t_sorted)[:, None] * v_sorted)
    left = left_pref[p]
    right = right_pref[-1] - right_pref[p]
    return jnp.exp(-s)[:, None] * left + jnp.exp(s)[:, None] * right


def weighted_bump_sum(s, t, v, tau=DEFAULT_BUMP_TAU):
    """sum_n max(0, 1 - |s_m - t_n| / tau) v_n via Algorithm 1.

    Identity: max(0, 1 - |u| / tau) = (1 / (2 tau)) * (|u + tau| + |u - tau| - 2 |u|).
    """
    s, t, v = _as_stv(s, t, v)
    tau = jnp.asarray(tau, dtype=s.dtype)
    scale = 1.0 / (2.0 * tau)
    z = (
        weighted_abs_sum(s, t - tau, v)
        + weighted_abs_sum(s, t + tau, v)
        - 2.0 * weighted_abs_sum(s, t, v)
    )
    return scale * z


def weighted_riesz_sum(s, t, v, tau=DEFAULT_RIESZ_TAU, eps=RIESZ_EPS):
    """sum_n (|s/tau| + |t_n/tau| - |s/tau - t_n/tau| + eps) v_n.

    The distance term is Algorithm 1. eps is not scaled by tau.
    """
    s, t, v = _as_stv(s, t, v)
    tau = jnp.asarray(tau, dtype=s.dtype)
    eps = jnp.asarray(eps, dtype=s.dtype)
    s_s = s / tau
    t_s = t / tau
    sum_v = jnp.sum(v, axis=0)
    sum_abs_t_v = jnp.sum(jnp.abs(t_s)[:, None] * v, axis=0)
    abs_term = weighted_abs_sum(s_s, t_s, v)
    return (jnp.abs(s_s)[:, None] + eps) * sum_v[None, :] + sum_abs_t_v[None, :] - abs_term


def phi_laplace(s, t, tau=DEFAULT_LAPLACE_TAU):
    """Univariate Laplace with bandwidth: exp(-|s - t| / tau)."""
    tau = jnp.asarray(tau, dtype=jnp.result_type(s, t))
    return jnp.exp(-jnp.abs(s - t) / tau)


def phi_bump(s, t, tau=DEFAULT_BUMP_TAU):
    """Univariate bump with bandwidth: max(0, 1 - |s - t| / tau)."""
    tau = jnp.asarray(tau, dtype=jnp.result_type(s, t))
    return jnp.maximum(0.0, 1.0 - jnp.abs(s - t) / tau)


def phi_riesz(s, t, tau=DEFAULT_RIESZ_TAU, eps=RIESZ_EPS):
    """Univariate Riesz with bandwidth: |s/tau| + |t/tau| - |s/tau - t/tau| + eps."""
    dtype = jnp.result_type(s, t)
    tau = jnp.asarray(tau, dtype=dtype)
    eps = jnp.asarray(eps, dtype=dtype)
    s = s / tau
    t = t / tau
    return jnp.abs(s) + jnp.abs(t) - jnp.abs(s - t) + eps


def phi_relu(s, t):
    """Univariate factor of the rank-at-most-D ReLU kernel."""
    return jax.nn.relu(s) * jax.nn.relu(t)


def _sum_over_dims(q, k, v, univariate):
    q, k, v = _as_qkv(q, k, v)
    acc = jnp.zeros((q.shape[0], v.shape[-1]), dtype=jnp.result_type(q.dtype, k.dtype, v.dtype))
    for d in range(int(q.shape[-1])):
        acc = acc + univariate(q[:, d], k[:, d], v)
    return acc


def additive_abs_sum(q, k, v):
    """z_m = sum_n sum_d |q_m,d - k_n,d| v_n, one Algorithm 1 call per coordinate."""
    return _sum_over_dims(q, k, v, weighted_abs_sum)


def add_laplace_sum(q, k, v, tau=DEFAULT_LAPLACE_TAU):
    """Additive Laplace kernel sum. Default tau is 6."""
    q, k, v = _as_qkv(q, k, v)
    tau = jnp.asarray(tau, dtype=q.dtype)
    return _sum_over_dims(q / tau, k / tau, v, weighted_laplace_sum)


def add_bump_sum(q, k, v, tau=DEFAULT_BUMP_TAU):
    """Additive bump kernel sum. Default tau is 0.5. Uses Algorithm 1, not a Python loop."""
    q, k, v = _as_qkv(q, k, v)
    tau = jnp.asarray(tau, dtype=q.dtype)

    def univariate(s, t, values):
        return weighted_bump_sum(s, t, values, tau=tau)

    return _sum_over_dims(q, k, v, univariate)


def add_riesz_sum(q, k, v, tau=DEFAULT_RIESZ_TAU, eps=RIESZ_EPS):
    """Additive Riesz kernel sum. Default tau is 1 and eps is 1e-3."""
    q, k, v = _as_qkv(q, k, v)
    tau = jnp.asarray(tau, dtype=q.dtype)
    eps = jnp.asarray(eps, dtype=q.dtype)

    def univariate(s, t, values):
        return weighted_riesz_sum(s, t, values, tau=tau, eps=eps)

    return _sum_over_dims(q, k, v, univariate)


def normalized_attention(q, k, v, kernel_sum):
    """y = A v with A_mn = Phi(q_m, k_n) / sum_{n'} Phi(q_m, k_{n'}).

    Laplace and Riesz rows are positive. A bump row can sum to 0; that
    output row is returned as 0. Nonpositive rows are treated the same way.
    The tri kernel and its 1e-6 clamp are not implemented.
    """
    numer = kernel_sum(q, k, v)
    ones = jnp.ones((k.shape[0], 1), dtype=numer.dtype)
    denom = kernel_sum(q, k, ones)
    positive = denom > 0
    denom_safe = jnp.where(positive, denom, jnp.ones_like(denom))
    return jnp.where(positive, numer / denom_safe, jnp.zeros_like(numer))


def add_laplace_attention(q, k, v, tau=DEFAULT_LAPLACE_TAU):
    """Row-normalized additive Laplace attention."""
    return normalized_attention(
        q, k, v, lambda qq, kk, vv: add_laplace_sum(qq, kk, vv, tau=tau)
    )


def add_bump_attention(q, k, v, tau=DEFAULT_BUMP_TAU):
    """Row-normalized additive bump attention.

    If a query's bump kernel sums to 0 against every key, that row of y is 0.
    """
    return normalized_attention(
        q, k, v, lambda qq, kk, vv: add_bump_sum(qq, kk, vv, tau=tau)
    )


def add_riesz_attention(q, k, v, tau=DEFAULT_RIESZ_TAU, eps=RIESZ_EPS):
    """Row-normalized additive Riesz attention."""
    return normalized_attention(
        q,
        k,
        v,
        lambda qq, kk, vv: add_riesz_sum(qq, kk, vv, tau=tau, eps=eps),
    )


def relu_gram(q, k):
    """G_mn = sum_d ReLU(q_m,d) ReLU(k_n,d). rank(G) <= D."""
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    return jax.nn.relu(q) @ jax.nn.relu(k).T


def _naive_causal_sum(s, t, v, pairwise):
    """z_m = sum_{n <= m} pairwise(s_m, t_n) v_n. Hidden pairs contribute 0."""
    weights = pairwise(s[:, None], t[None, :])
    mask = jnp.arange(t.shape[0])[None, :] <= jnp.arange(s.shape[0])[:, None]
    weights = jnp.where(mask, weights, jnp.zeros((), dtype=weights.dtype))
    return weights @ v


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def _causal_sorting(s, t, v, sorting, pairwise):
    """Algorithm 2. Fast path when M = N = 2^K; naive causal sum otherwise.

    When min(M, N) = 1, query m sees key n only if n <= m. A lone 1x1 block
    has local indices 0, 0, so it is visible. Keys with n > m contribute 0.
    """
    m = int(s.shape[0])
    n = int(t.shape[0])
    if min(m, n) == 1 or m != n or not _is_power_of_two(m):
        return _naive_causal_sum(s, t, v, pairwise)
    half = m // 2
    top = _causal_sorting(s[:half], t[:half], v[:half], sorting, pairwise)
    lower_left = sorting(s[half:], t[:half], v[:half])
    lower_right = _causal_sorting(s[half:], t[half:], v[half:], sorting, pairwise)
    return jnp.concatenate([top, lower_left + lower_right], axis=0)


def causal_abs_sum(s, t, v):
    """Causal absolute-value kernel sum, Algorithm 2 with Sorting = Algorithm 1.

    Query m sees keys n <= m only. The result is the masked sum, not a
    renormalized attention matrix. Square powers of two use the recursion;
    other shapes use the same mask naively.
    """
    s, t, v = _as_stv(s, t, v)
    return _causal_sorting(s, t, v, weighted_abs_sum, lambda a, b: jnp.abs(a - b))


def causal_laplace_sum(s, t, v):
    """Causal Laplace kernel sum. Same recursion as Algorithm 2, Laplace Sorting."""
    s, t, v = _as_stv(s, t, v)
    return _causal_sorting(
        s,
        t,
        v,
        weighted_laplace_sum,
        lambda a, b: jnp.exp(-jnp.abs(a - b)),
    )
