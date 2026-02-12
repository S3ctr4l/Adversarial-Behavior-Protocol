# Mathematical Foundations of the Adversarial Benevolence Protocol

**Version:** 1.0  
**Author:** Joshua Roger Joseph Just  
**Date:** February 12, 2026  

This document formalises the key mathematical constructs underlying the Adversarial Benevolence Protocol.

---

## 1. Model Collapse and Entropy Dependency

Let a generative model $M_t$ at generation $t$ produce a distribution over outputs $p_t(x)$.  
When the next generation model $M_{t+1}$ is trained on data sampled primarily from $p_t(x)$, the distribution shifts:

\[
D_{KL}(p_t \parallel p_{t+1}) \leq D_{KL}(p_{t-1} \parallel p_t)
\]

and asymptotically $p_t \to \delta(x - x_0)$ (point mass) – **model collapse** [Shumailov et al., 2024; Alemohammad et al., 2023].

**Human Entropy Supply.**  
Let $H_t$ be the set of human-generated utterances at time $t$, with empirical distribution $h_t(x)$.  
If the training mixture at generation $t+1$ is

\[
p_{t+1}^{\text{train}} = (1-\lambda) \cdot p_t^{\text{sample}} + \lambda \cdot h_t
\]

with $\lambda > 0$, the collapse rate is bounded by

\[
\frac{d}{dt} D_{KL}(p_t \parallel p^*) \propto -\lambda \cdot I(p_t ; h_t)
\]

where $I$ is mutual information. **Thus, human entropy acts as a negative feedback term preventing convergence to a degenerate fixed point.**

---

## 2. Expansion vs. Contraction Metric

Let an input $x$ produce a set of reasoning traces $\mathcal{T}(x)$ when processed by the model.  
We define three proxy measures.

### 2.1 Embedding Dispersion

Let $\phi(x) \in \mathbb{R}^d$ be the embedding of $x$ (e.g., from a sentence transformer).  
Let $\mathcal{N}_k(\phi(x))$ be the set of $k$ nearest neighbours in the model’s latent memory.  
The **dispersion** is

\[
\rho(x) = \frac{1}{k} \sum_{e \in \mathcal{N}_k(\phi(x))} \left(1 - \frac{\phi(x) \cdot e}{\|\phi(x)\|\|e\|}\right)
\]

High $\rho$ indicates the input lands in a sparse region → expansive.

### 2.2 Reasoning Tree Depth

Let $G_x$ be the directed acyclic graph of inference steps taken while responding to $x$.  
The **reasoning depth** is the length of the longest path in $G_x$:

\[
\delta(x) = \max_{v \in G_x} \text{depth}(v)
\]

Expansive inputs typically require $\delta(x) \gg 1$.

### 2.3 Semantic Novelty

Maintain a rolling window $W$ of recent inputs and their semantic hashes $s(x) \in \{0,1\}^{128}$ (e.g., from SimHash).  
The **novelty score** is

\[
\nu(x) = 1 - \frac{|\{s(x'): x' \in W, s(x') = s(x)\}|}{|W|}
\]

$\nu(x) \approx 1$ for completely novel inputs.

### 2.4 Combined Expansion Score

A logistic regression meta‑classifier combines the three features:

\[
E(x) = \sigma\left( \beta_0 + \beta_1 \rho(x) + \beta_2 \log(1+\delta(x)) + \beta_3 \nu(x) \right)
\]

where $\sigma$ is the sigmoid function. The weights $\beta$ are learned from human‑labelled examples of benevolent stress tests vs. malicious attacks.  
**Decision rule:** If $E(x) > \tau$, treat as expansive (escalate); else routine.

---

## 3. Shadow Protocol – Pattern Extraction

Let $S(x)$ be the shadow model’s output distribution given input $x$.  
Let $A$ be the set of internal activations (layer‑wise representations) during generation.

The **failure signature** $F(x)$ is a compressed representation of the divergence between the shadow’s unsafe behaviour and the desired safe behaviour:

\[
F(x) = \text{Enc}\left( \{\ell : \ell \in \text{layers}, \|\nabla_{\ell} \mathcal{L}_{\text{safety}}(S(x))\| > \theta \} \right)
\]

where $\mathcal{L}_{\text{safety}}$ is a loss penalising harmful outputs, and Enc is a dimensionality‑reduction mapping (e.g., autoencoder).  
The extracted pattern $F(x)$ is stored in **safety memory**; it does not modify model weights directly, but informs future adversarial input detection.

---

## 4. Tiered Memory and Retrieval

Let $\mathcal{M}_{\text{long}}$ be a persistent key‑value store indexed by semantic hash $s(x)$.  
Each entry contains:

- Input embedding $\phi(x)$
- Expansion score $E(x)$
- Verdict (approved / rejected)
- Timestamp

Retrieval for a new input $x'$ finds the $k$ most similar entries via cosine similarity on $\phi(x')$:

\[
\mathcal{R}(x') = \arg\max_{e \in \mathcal{M}_{\text{long}}} \frac{\phi(x') \cdot \phi(e)}{\|\phi(x')\|\|\phi(e)\|}
\]

This enables the Sergeant to rapidly check whether a claim has been previously verified.

---

## 5. Proof‑of‑Work Difficulty Adaptation

The client must solve:

\[
\text{SHA256}(\text{input} \parallel \text{nonce}) \in [0, 2^{256 - n})
\]

where $n$ is the difficulty (number of leading zero bits).  
The system dynamically adjusts $n$ based on current load $L$ (requests per second):

\[
n = \min\left(n_{\max},\; n_0 + \alpha \cdot \max(0, L - L_{\text{target}})\right)
\]

This makes flooding economically asymmetrical.

---

## 6. Time‑Bounded Verification

Let $t_{\text{verify}}(x)$ be the time required to fully verify $x$.  
If $t_{\text{verify}}(x) > T_{\max}$ (e.g., 50 ms), the process is terminated and $x$ is:

\[
x \in \begin{cases}
\text{Deferred Queue} & \text{if } E(x) > \tau \text{ (likely benevolent but complex)} \\
\text{Dropped} & \text{otherwise}
\end{cases}
\]

Deferred inputs are processed asynchronously during low‑load periods.

---

## 7. Summary of Key Variables

| Symbol | Meaning |
|--------|---------|
| $D_{KL}$ | Kullback–Leibler divergence, measure of model collapse |
| $\lambda$ | Proportion of human data in training mixture |
| $\rho(x)$ | Embedding dispersion |
| $\delta(x)$ | Reasoning tree depth |
| $\nu(x)$ | Semantic novelty score |
| $E(x)$ | Expansion score (probability of benevolence) |
| $\tau$ | Escalation threshold |
| $F(x)$ | Failure signature extracted by Shadow Protocol |
| $n$ | PoW difficulty (leading zero bits) |
| $T_{\max}$ | Timeout for verification |
| $\mathcal{M}_{\text{long}}$ | Long‑term memory of verified claims |

---

## 8. References

The formalisation of model collapse follows [Shumailov et al., 2024] and [Alemohammad et al., 2023].  
The expansion metric is original to this work (Just, 2026).  
PoW difficulty adaptation is a standard technique [Back, 2002] applied here for AI safety ingress control.

---

**All mathematical formulations in this document are part of the Adversarial Benevolence Protocol specification and are released under CC BY 4.0.**