Additive Laplace, bump, and Riesz kernel attention via sorting, in JAX on CPU (arXiv:2609.35349).

```
pip install -e .
```

```
JAX_PLATFORMS=cpu python -m pytest
```

This is not a CUDA fused kernel, and it does not do fp16, nomic-embed distillation, or a timing table. Defaults follow Section 4 (Laplace 0.5, bump 1.5), not the full L1 Laplace kernel at tau 6. The capacity check is a closed-form witness, not Figure 1. A bump row that sums to zero returns a zero output row.
