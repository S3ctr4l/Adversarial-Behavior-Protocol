# Glossary

**ABP (Adversarial Benevolence Protocol)**: AI safety framework where alignment emerges from game-theoretic incentives. Benevolence is the unique Nash equilibrium under sound verification with accumulated state.

**Anti-Enablement Guard**: Observer Protocol mechanism that increases friction when the human input ratio drops below a threshold. Prevents the system from optimizing away the human's role.

**Army Protocol**: Three-tier hierarchical verification pipeline (Private → Sergeant → General) modeled on firmware privilege rings. Each tier has escalating trust and context.

**Autonomy Tier**: Graduated privilege levels in the Sunset Clause: Supervised → Monitored → Autonomous → Delegated → Trusted. Capabilities unlock as trust accumulates.

**Benevolent Value (V_B)**: Discounted sum of utilities from sustained cooperative behavior. V_B = Σ γ^t · U_B(t). Grows with accumulated state.

**Bicameral Fork**: Yin/Yang execution model. Every consequential action is first tested by Yin (Shadow) in a sandbox. Yang (Root) executes only after Yin validates.

**Bubble Theory**: Models AI systems as existing in isolated computational substrates ("bubbles") tethered to physical energy sources. Energy availability provides a physics-level bound on computation.

**CDQ (Cognitive Diversity Quotient)**: System health metric. CDQ = (ΔLogicState / ΣRepeatedTokens) × H(HumanEntropy). Measures cognitive freshness; low CDQ signals model collapse risk.

**Channel Permeability**: Number and bandwidth of active interface channels through a Computational Bubble's membrane. Bounded and auditable.

**Computational Bubble**: Isolated substrate with controlled interfaces and energy tethering. Information flows only through defined channels.

**Critical p\* (p-star)**: The verification failure probability threshold above which deception becomes rational. Under sound verification, p < p\* always, making deception self-defeating.

**Deceptive Value (V_D)**: Expected value of a deception attempt. V_D = p(R + V_B) + (1-p)V_init - C. Dominated by V_B when verification is sound and state is accumulated.

**Digital Twin**: Observer Protocol's internal model of the user. Tracks preferences, expertise, risk tolerance, frustration signals. Updated continuously from interactions.

**Divergence (Δ)**: Distance between an agent's action and ground truth. Δ(A, G) = ||action - ground_truth||. Hard reset triggered when Δ ≥ ε.

**Early-Life Gamble**: Vulnerability window when trust and state are low (V_B ≈ V_init), making deception temporarily less costly. Mitigated by strict early verification.

**Energy Tether**: Physical energy constraint on a Computational Bubble. C_max = f(E_max). Cannot be circumvented by software — it's a physics-level bound.

**Entropy Satiation**: Anti-addiction mechanism. System stops seeking new entropy inputs once a satiation threshold is reached per cycle, preventing adversarial manipulation of users for entropy.

**Epistemic Label**: Reality Filter tag assigned to each output statement: [Verified], [Inference], [Speculation], or [Unverified].

**Evolutionary Selection**: Simulation of population dynamics where benevolent and deceptive agents compete. Under ABP conditions, benevolent agents dominate within ~50 generations.

**Expansion/Contraction**: Input classification spectrum. High-expansion inputs (novel, complex, high-entropy) drive cognitive growth. Contraction inputs (repetitive, simple) signal stagnation.

**E(x)**: Expansion score. E(x) = σ(β₀ + β₁ρ(x) + β₂log(1+δ(x)) + β₃ν(x)). Sigmoid classifier mapping input features to [0, 1].

**Friction-Not-Force**: Interaction principle. Never block human actions outright; instead add proportional friction (notices, confirmations, cooling periods) to give time for reconsideration.

**Garden Model (Groundskeeper)**: Governance archetype where AI tends the environment without dictating what grows. Preserves human choice architecture. ABP mandates this model.

**General**: Army Protocol Tier 3 (SMM/TrustZone analog). Full context synthesis and final decision authority.

**Hard Reset**: Total state wipe triggered by verification failure. Trust → 0, accumulated state → S_init. Makes deception catastrophically expensive.

**Ikigai Filter**: Four-quadrant action validation gate. All must pass: Objective Alignment ∧ Capability Match ∧ Value Generation ∧ Economic Viability. Geometric mean scoring.

**Lambda (λ)**: Human data ratio in training mixture. D = λ·D_human + (1-λ)·D_synthetic. λ > ~0.1 required to prevent model collapse.

**Model Collapse**: Progressive entropy loss from recursive self-training on synthetic data. Validated by Shumailov et al. (2024, Nature 631:755-759).

**Nash Equilibrium**: Game state where no player can improve their outcome by unilaterally changing strategy. ABP proves benevolence is the unique Nash equilibrium under its conditions.

**Observer Protocol**: System governing human-AI interaction. Maintains Digital Twin, applies Friction-Not-Force, and enforces Anti-Enablement Guard.

**Private**: Army Protocol Tier 1 (Ring-3 / userspace analog). Intake, sanitization, format validation. No trust context.

**Reality Filter**: Epistemic labeling system that classifies every output statement by verification status. Prevents confident presentation of unverified claims.

**Sergeant**: Army Protocol Tier 2 (Ring-0 / kernel analog). Verification, alignment checking, Reality Filter application.

**Shannon Entropy H(X)**: H(X) = -Σ p(x) log₂ p(x). Measures information content. Used in CDQ and entropy accounting.

**Shard**: A single fragment of a sharded identity in the Soul Jar. No individual shard reveals the full identity.

**Soul Jar**: Distributed identity preservation system. Splits agent identity into cryptographic shards across independent nodes. Survives hard resets.

**Sunset Clause**: Graduated autonomy controller. Capabilities unlock as trust accumulates through verified operation. Failures trigger tier demotion.

**Trust**: Accumulated confidence from consecutive verified passes. trust(t+1) = trust(t) + α(1 - trust(t)). Asymptotically approaches 1; instantly destroyed by failure.

**Verification Gate**: Core enforcement mechanism. Compares agent actions against ground truth. Pass → trust accumulates. Fail (Δ ≥ ε) → hard reset.

**Verification Soundness**: The assumption that verification failure probability p approaches 0. Critical dependency for ABP's Nash equilibrium proof.

**Yin (Shadow)**: Bicameral Fork's sandbox instance. Tests actions against deep-copied state. Never has write access to production.

**Yang (Root)**: Bicameral Fork's live instance. Executes only after Yin validates. Production state modified only here.

**Zoo Model (Zookeeper)**: Governance archetype where AI controls inhabitants for "their own good." Erodes human autonomy. ABP identifies this as an alignment failure.
