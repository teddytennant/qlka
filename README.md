Additive Laplace, bump, and Riesz kernel attention via sorting, in JAX on CPU.

```
pip install -e .
```

```
JAX_PLATFORMS=cpu python -m pytest
```

This is not a CUDA fused kernel, and it does not do fp16, nomic-embed distillation, or a timing table. The capacity check is a closed-form witness, not Figure 1. A bump row that sums to zero returns a zero output row.
