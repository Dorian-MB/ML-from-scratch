# Self-made ML

From-scratch implementations of core machine learning concepts — no PyTorch, no Keras, just NumPy and the math behind it.

---

## Notebooks

### `likelihood.ipynb` — Maximum Likelihood Estimation
Derives and visualises MLE for a Gaussian distribution by hand.
- Gaussian pdf and log-likelihood surface
- Analytical derivation of $\hat{\mu}$ and $\hat{\sigma}^2$
- Grid search over the likelihood surface vs closed-form estimator vs `scipy.stats`

---

### `optimizer.ipynb` — SGD / Momentum / Adam
Compares three optimisers on a binary classification task (logistic regression, hand-derived gradients).
- **SGD** — plain gradient descent
- **Momentum** — exponential moving average of gradients (EMA)
- **Adam** — combines momentum (1st moment) and RMSProp adaptivity (2nd moment), with bias correction for zero-initialised moments
- Full step-by-step derivation of Adam in markdown (all four steps with LaTeX)
- Convergence curves and final decision boundaries compared

---

### `graph-de-calcule.ipynb` — Autograd Engine
Builds a miniature automatic differentiation engine from scratch, scalar then numpy-array version.
- Scalar `Node`: `+`, `*`, `**`, `exp`, with topological-sort backward pass
- Array `Node`: matrix multiply (`@`), element-wise ops, `relu`, `sigmoid`, `log`, `mean`, `sum`
- `_sum_to` helper to reverse numpy broadcasting in backward passes
- Validates against `sympy` symbolic derivatives

---

### `seq-nn.ipynb` — Sequential Neural Network (manual backprop)
A modular neural network where each layer owns its own `forward` / `backward` / `get_params`.
- `Linear`, `ReLU`, `Dropout` (inverted), `BatchNorm`
- Detailed markdown derivations: ReLU, Linear (with full element-wise transpose proof), Dropout, BatchNorm (Ioffe & Szegedy 2015 backward)
- **Adam** optimiser keyed by parameter object identity (`id(node)`)
- Trained on `make_moons`, visualised with decision boundaries
- No autograd — every gradient formula is written explicitly

---

### `nn.ipynb` — Graph Neural Network (autograd backprop)
Replaces manual `backward()` methods with the autograd engine from `graph-de-calcule`, enabling non-sequential architectures.
- `Parameter` class (array `Node`) with full op set: `@`, `+`, `-`, `*`, `/`, `relu`, `sigmoid`, `log`, `exp`, `mean`, `var`, `sqrt`
- `Linear`, `Dropout`, `BatchNorm` layers — no `backward()` needed, gradients flow automatically
- **`ResidualBlock`**: `out + x` skip connection — `Parameter.__add__` routes gradient to both branches with a single line
- **`GraphNetwork`**: embed → ResBlock → BatchNorm → ResBlock → head
- `binary_cross_entropy` written in `Parameter` ops, `loss.backward()` propagates everything
- Adam updated to accept a flat list of `Parameter` objects (no pre-registration needed)

---

## Key concepts thread

```
likelihood → optimisers → manual backprop (seq-nn) → autograd engine → graph net (nn)
```

Each notebook builds on the intuition of the previous one. The progression from `seq-nn` to `nn` mirrors the design difference between a framework where you hand-write gradients and one like PyTorch where the graph is built dynamically at runtime.
