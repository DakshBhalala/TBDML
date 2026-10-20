# Architecture Specification: CompactTab

*Project: CompactTab*  
*Architecture Class: Parameter-Efficient Tabular Deep Network with Quantile Piecewise-Linear Embeddings and BatchEnsemble Gated Residual Blocks*

---

## 1. Architectural Overview

`CompactTab` is designed to solve the two fundamental weaknesses of tabular neural networks:
1. **The Continuous Feature Split Deficiency:** Standard neural networks struggle to represent sharp, localized step-decisions that tree splits easily discover. CompactTab integrates trainable **Quantile Piecewise-Linear Embeddings (PLE)**, transforming scalar floats into localized multi-bin representations.
2. **The Ensemble Cost Barrier:** Ensembling multiple independent models is the single most reliable way to boost tabular accuracy, but it multiplies parameter counts, memory footprint, and training/inference compute by $k$. CompactTab integrates **BatchEnsemble**, a parameter-efficient ensembling technique that maintains a single shared backbone weight matrix $W$ alongside $k$ rank-1 modulation vectors $r_k$ and $s_k$. This simulates an ensemble of $k$ distinct models in a single vectorized forward pass with merely $\approx 10\%$ extra parameters.

```
Input Table (Numerical & Categorical Features)
   │
   ├── [Numerical Features] ──> Quantile Piecewise-Linear Embedding (PLE) ──> [B bins x d_num] ──┐
   ├── [Categorical Features] ──> Entity Embedding Table ────────────────────> [d_cat] ──────────┼──> Feature Concatenation & Linear Projection (Hidden Dim D)
   └── [Missingness Mask] ─────> Binary Indicator Projection ───────────────> [d_num] ──────────┘
                                                                                                        │
                                                                                                        ▼
                                                                                   ┌───────────────────────────────────────────┐
                                                                                   │       Gated Residual Block x L Layers     │
                                                                                   │  ├── RMSNorm                              │
                                                                                   │  ├── BatchEnsemble Gated Linear Unit      │
                                                                                   │  │   (W_gate ⊙ r1_k s1_k, W_up ⊙ r2_k s2_k) │
                                                                                   │  ├── SiLU Activation                      │
                                                                                   │  ├── BatchEnsemble Down-Projection        │
                                                                                   │  │   (W_down ⊙ r3_k s3_k)                 │
                                                                                   │  ├── Dropout (p_drop)                     │
                                                                                   │  └── Residual Skip Addition               │
                                                                                   └───────────────────────────────────────────┘
                                                                                                        │
                                                                                                        ▼
                                                                                   ┌───────────────────────────────────────────┐
                                                                                   │       Multi-Head BatchEnsemble Head       │
                                                                                   │  k independent prediction heads           │
                                                                                   │  (Binary / Multi-class / Regression)      │
                                                                                   └───────────────────────────────────────────┘
                                                                                                        │
                                                                                                        ▼
                                                                                       Ensemble Averaging (Inference)
                                                                                       or Multi-Head Loss (Training)
```

---

## 2. Component Specifications

### 2.1 Numerical Feature Encoding: Piecewise Linear Embeddings (PLE)
Given a continuous feature $x \in \mathbb{R}$, we compute $B$ quantile-based bin edges $b_0 < b_1 < \dots < b_B$ fitted strictly on training data.
For an input scalar $x$, the piecewise linear encoding $e(x) \in [0, 1]^B$ is computed as:
$$
e_j(x) = \begin{cases}
0 & \text{if } x < b_{j-1} \\
\frac{x - b_{j-1}}{b_j - b_{j-1}} & \text{if } b_{j-1} \le x \le b_j \\
1 & \text{if } x > b_j
\end{cases}
$$
This soft-activation representation preserves both ordinal continuity and localized slope information. Each bin is then projected linearly to embedding dimension $E_{num}$.

### 2.2 Categorical Feature Encoding
Categorical features are mapped to dense embeddings using a learned lookup table. Unseen test categories and missing categorical values map to a dedicated Out-Of-Vocabulary (OOV) index $0$. The embedding dimension $E_{cat}$ is determined by standard heuristic:
$$
E_{cat} = \min\left(50, \left\lfloor 1.6 \times C^{0.56} \right\rfloor + 1\right)
$$
where $C$ is the number of unique categories.

### 2.3 BatchEnsemble Layer
Let $W \in \mathbb{R}^{d_{out} \times d_{in}}$ be the shared weight matrix. For an ensemble of $k$ submodels, we introduce two rank-1 adapter matrices:
- Input adapter: $R \in \mathbb{R}^{k \times d_{in}}$ (row vectors $r_1, \dots, r_k$)
- Output adapter: $S \in \mathbb{R}^{k \times d_{out}}$ (row vectors $s_1, \dots, s_k$)

The effective weight matrix for ensemble member $m \in \{1, \dots, k\}$ is:
$$
W^{(m)} = W \odot (s_m r_m^T)
$$
During forward execution, for mini-batch $X$ replicated across $k$ ensemble members:
$$
Y^{(m)} = ( (X^{(m)} \odot r_m) W^T ) \odot s_m + b^{(m)}
$$
This allows evaluating all $k$ ensemble members in a single vectorized matrix multiplication without storing $k$ separate dense weight matrices.

### 2.4 Conservative Initialization
To ensure training stability, adapters are initialized near unity:
$$
r_m \sim \mathcal{N}(1, \sigma^2), \quad s_m \sim \mathcal{N}(1, \sigma^2) \quad \text{with } \sigma \in [0.01, 0.05]
$$
This guarantees that all ensemble members initially compute virtually identical representations, preventing chaotic gradient oscillations, and gradually diverge to capture distinct predictive features.

---

## 3. Configuration Profiles

| Dimension / Hyperparameter | `CompactTab-Tiny` | `CompactTab-Small` | `CompactTab-Medium` |
| :--- | :--- | :--- | :--- |
| **Target Parameters** | **$\approx 100\text{k}$** | **$\approx 1\text{M}$** | **$\approx 10\text{M}$** |
| Hidden Dimension ($D$) | 64 | 128 | 384 |
| Number of Layers ($L$) | 2 | 4 | 6 |
| Ensemble Heads ($k$) | 4 | 8 | 8 |
| Continuous Bins ($B$) | 8 | 16 | 32 |
| Gated Expansion Ratio | 2.0 | 2.0 | 2.0 |
| Dropout Rate | 0.10 | 0.15 | 0.20 |
| Weight Decay | $1\times 10^{-4}$ | $1\times 10^{-4}$ | $3\times 10^{-4}$ |
| Learning Rate | $1\times 10^{-3}$ | $8\times 10^{-4}$ | $5\times 10^{-4}$ |
| Recommended Hardware | CPU or Mobile/Edge | RTX 3050 (Laptop) | RTX 3050 / Server GPU |
