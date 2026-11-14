# Methodology

Last update: 14 November 2026, 06:50. **Outcome of the gates: G3 and G4 failed. See Section 9 before reading Sections 4–7.** Status marks: **[S]** supported by a published paper we checked (see `LITERATURE_REVIEW.md`); **[O]** our own measurement (see `RESULTS.md`, experiment IDs from `EXPERIMENT_LOG.md`); **[H]** hypothesis that still has to pass an experiment.

## 1. Goal and research questions (unchanged)

We want to know whether a graph neural network can make Adaptive Large Neighbourhood Search (ALNS) better for pickup-and-delivery and dial-a-ride problems with heterogeneous fleets, while every solution stays exactly feasible.

- **RQ1.** At the same time budget, is the guided ALNS better than classical ALNS?
- **RQ2.** Does a heterogeneous graph model guide better than a homogeneous one?
- **RQ3.** Does the method keep all constraints under short deadlines?
- **RQ4.** When does guidance help (instance size, fleet, budget)?

## 2. The first method and why it is replaced

**First method.** A heterogeneous GNN predicted, for every removed request, the route it had in an elite solution. ALNS used this as top-$K$ pruning, as a prior on the insertion cost, and in one destroy operator.

**Result.** Not better than classical ALNS on 140 benchmark instances (E04), also at equal iterations (E05).

| Weakness | Evidence |
|---|---|
| The network predicts what ALNS already computes exactly | Exact insertion cost finds the label route in 70% of cases, the network in 43% [O, E07]. Same pattern in the literature: learned heatmaps are no better than plain distance inside a strong solver [S, Santana et al. 2023; Xia et al. 2024] |
| The teacher was not better than the search | Labels came from runs shorter than the test runs. Advice from a 0.6× or 1× run does not help (+0.46%, +0.32%); advice from a 4× run helps (−0.78%, p = 0.002) [O, E08c] |
| Hard pruning removes good options | Pruning alone: +8.5 on Li & Lim 200 [O, E06]. Even with oracle advice, pruning alone is harmful on H-DARP style instances (+2.0%, p < 0.001) [O, E09b]. Fixing decisions from imperfect predictions is known to hurt [S, Liu et al. 2025; Han et al. 2023] |
| The label depends on a search state the search never visits | Good offline accuracy, no gain in the search; same finding in [S, Feijen et al. 2024; Ross et al. 2011] |

**Directions tested and closed before choosing the new method**

| Idea | Test | Result |
|---|---|---|
| Learn which requests to remove (one step ahead) | Perfect selection among $m$ random moves [E08a] | At most −0.7% (H-DARP style) even when the selection is free and perfect; nothing on Li & Lim style. Closed |
| Guide from the run's own best solution | E08b | +4.5%; 0 of 28 instances better. Closed |
| Guide from static relatedness (no learning) | E09b | Worse than classical ALNS (+0.7%, p = 0.006). Closed |
| Old co-routing head as static guide | E08b | Worse (+0.5%, p = 0.040). Closed |

## 3. What the pilots say a new method must look like

All numbers: 50 generated tuning instances, equal iterations, difference to classical ALNS [O, E08–E09].

| Finding | H-DARP style | Li & Lim style |
|---|---|---|
| Classical ALNS with 4× the iterations | −1.24% | −2.71 |
| Oracle advice from one 4× run, **prior only**, $\lambda = 1$ | **−1.33%** (23 better / 4 worse, p < 0.001) | **−11.3** (p = 0.004) |
| Same, $\lambda = 0.2$ | −0.77% | −8.5 |
| **Soft consensus** of three 4× runs (prior + pruning + destroy, as tuned for the first method) | −0.58% (p = 0.015) | −8.6 (p = 0.015) |
| Oracle used as destroy operator only | −0.20% (n.s.) | −0.8 (n.s.) |
| Oracle used for pruning only | +2.0% (worse) | −7.5 (n.s.) |

Design rules that follow:

1. **Use a teacher that is clearly better than the search** (4× or more).
2. **Use the advice as a prior only.** No pruning, no learned destroy operator.
3. **Predict something that does not depend on the search state**, once per instance, so the network costs nothing inside the loop. The soft consensus of several long runs is such a target, and it works as a guide.
4. **Compare with a free rule and with the oracle.** A learned model is only worth it if it is better than static relatedness and gets a clear part of the oracle gain.

One warning from our own data [O, E09c]: on H-DARP style instances two long runs of the same instance agree only weakly on which requests share a route (AUC 0.66). So the learnable part of the target is small. Whether a network can learn enough of it is the open question of gate G3.

## 4. Revised method: teacher-distilled co-routing prior

### 4.1 Workflow

**Offline (once per problem family)**

1. Generate training instances (same generators as before; no benchmark instance is used).
2. For each instance run classical ALNS $K_T = 3$ times with $M = 4$ times the iteration budget of the target setting (the *teacher* runs).
3. Build the soft co-routing target from these runs (4.2).
4. Train the HGNN to predict the target from the static instance graph (4.3).

**Online (per instance)**

1. One forward pass of the HGNN gives the predicted co-routing matrix $\hat C$.
2. ALNS runs as usual. In the repair operators, the insertion cost of request $r$ into route $k$ gets a prior term computed from $\hat C$ (4.4).
3. At the end, a set-partitioning step recombines the routes found during the search (4.5).

### 4.2 Training target

Let $S_1,\dots,S_{K_T}$ be the teacher solutions of an instance and $\rho_t(r)$ the route of request $r$ in $S_t$.

$$y_{rr'} = \frac{1}{K_T}\sum_{t=1}^{K_T} \mathbb{1}\left[\rho_t(r)=\rho_t(r')\right]$$

for every pair of different requests $r$ and $r'$. So $y_{rr'}$ is the share of the teacher runs in which the two requests are in the same route.

### 4.3 Model and loss

- **Graph.** The same static heterogeneous graph as before: pickup nodes, delivery nodes and vehicle-class nodes; typed edges between the 16 nearest stops (time-window reachable), and between stops and compatible vehicle classes.
- **Encoder.** The same relation-aware attention network (3 layers, width 128, 8 heads). It runs once per instance and gives an embedding $h_v$ for every node.
- **Pair head.** For every request–request edge $e=(u,v)$ of the graph, with edge embedding $\epsilon_e$:

$$\hat c_{e} = \sigma\left(\mathrm{MLP}\left(h_u, h_v, \epsilon_e\right)\right)$$

  The three vectors are concatenated before the MLP. The predicted value $\hat C_{rr'}$ for a request pair is the largest $\hat c_e$ over the edges between the stops of $r$ and $r'$, and $0$ for pairs that are not neighbours in the graph.

- **Loss.** Binary cross-entropy with the soft target $y_e$, over the $m$ request–request edges $E$:

$$L = -\frac{1}{m}\sum_{e\in E}\left[y_e\log \hat c_e + (1-y_e)\log(1-\hat c_e)\right]$$

- **Comparison models** (same data, same loss): a homogeneous GNN (RQ2), and the same head without message passing (is a graph network needed at all?).

### 4.4 Use inside ALNS

When the repair step has to insert the removed request $r$, let $R_k$ be the requests now in route $k$ and $P$ all requests now placed.

**Affinity of $r$ for route $k$** (share of the predicted co-routing mass that sits in route $k$):

$$A(r,k) = \frac{\sum_{r'\in R_k}\hat C_{rr'}}{\sum_{r'\in P}\hat C_{rr'}}$$

For an empty vehicle, $A(r,k)=\tfrac12$ if $r$ has no predicted route-mate among the placed requests, and $0$ otherwise.

**Prior and biased insertion cost** ($\beta = 6$; $d_{\mathrm{avg}}$ = mean distance of the instance; $\Delta$ = exact insertion cost; the sum runs over all candidate routes $j$):

$$\pi(k \mid r)=\frac{\exp\left(\beta A(r,k)\right)}{\sum_{j}\exp\left(\beta A(r,j)\right)}$$

$$\Delta_{\mathrm{biased}}(r,k)=\Delta(r,k)-\lambda \cdot d_{\mathrm{avg}} \cdot \log\pi(k \mid r)$$

- All routes are always evaluated. Nothing is pruned.
- Feasibility is checked exactly, as before. Every solution is feasible whatever the network predicts.
- With $\lambda = 0$ the method is exactly classical ALNS.
- The acceptance test of ALNS uses the true cost, not the biased one. The prior only changes which insertion is tried.

### 4.5 Base solver upgrade: set partitioning over the route pool

Every route of an accepted solution is stored in a pool $\Omega$ with its cost $c_j$. After the last iteration one integer program is solved ($a_{rj}=1$ if route $j$ serves request $r$; $m_g$ = number of vehicles of class $g$; $\Omega_g$ = routes of that class):

$$\min \sum_{j\in\Omega} c_j x_j$$

subject to

$$\sum_{j\in\Omega} a_{rj}x_j = 1 \quad \text{for every request } r$$

$$\sum_{j\in\Omega_g} x_j \le m_g \quad \text{for every vehicle class } g$$

with every $x_j$ equal to 0 or 1.

- For Li & Lim (vehicles first) a large constant is added to every $c_j$.
- The incumbent's routes are in the pool, so the result is never worse.
- This step is used by the best published methods for both problems [S, Parragh & Schmid 2013; Gschwind & Drexl 2019; Sartori & Buriol 2020; Malheiros et al. 2021].
- First measurement [O, E10]: Li & Lim style −5.1 (16 better / 0 worse, p < 0.001); H-DARP style −0.05% (11 better / 0 worse). Cost: about 0.2–1.3 s per run.
- It is added to **both** the classical baseline and the guided method, so it does not bias RQ1.

### 4.6 Cost

| Part | Cost |
|---|---|
| Teacher runs (offline) | $K_T \cdot M = 12$ times one normal run per training instance; about 37 CPU-hours for 600 instances |
| Training (offline) | Minutes on one GPU |
| Forward pass (online, once) | About 0.1–0.2 s per instance (0.2% of the budget in E04) |
| Prior inside the loop | One matrix lookup per (removed request, route); measured run time ×1.10 for the oracle version [O, E09b] |

### 4.7 Differences to the first method

| | First method | Revised method |
|---|---|---|
| Teacher | Runs shorter than the test runs | Three runs with 4× the iterations |
| What is predicted | Route of a removed request in a given search state | Co-routing of request pairs, independent of the state |
| Network calls | Every iteration | Once per instance |
| Use | Pruning + prior + destroy operator | Prior only |
| Safety | Pruning could remove the best route | Nothing is removed |
| Base solver | ALNS | ALNS + set partitioning |

## 5. Decision gates

| Gate | Question | Status |
|---|---|---|
| G1 | Does oracle advice from a stronger teacher help, and weaker teachers not? | **Passed** (E08c) |
| G2 | Which use carries the gain? Does a soft consensus work? Does a free rule already give it? | **Passed**: prior only; consensus works; free rule fails (E09b) |
| G3 | Does a trained model predict long-run co-routing better than the free rule, the old head and a 1× run? | **Failed** (E13): AUC 0.564 on H-DARP style, same as the old head (0.563), below a 1× run (0.606) |
| G4 | Tuning instances: is the model-guided search better than classical ALNS (paired Wilcoxon, p < 0.05) and than the free rule? | **Failed** (E14): +0.19% (n.s.) with a weak prior, +5.49% with a strong prior on H-DARP style |
| G5 | Benchmarks (Li & Lim 100 / 200, H-DARP), 5–10 seeds, equal time, with a budget sweep | Not run for the learned prior (G4 failed). Confirmation runs on benchmarks: E15 (set partitioning), E16 (oracle and trained model on 36 instances) |

If G3 or G4 fails, the result is reported as negative, together with the oracle numbers that show how much a perfect model could have given.

## 6. Validation plan

**Baselines**

| Name | What it is |
|---|---|
| B1 | Classical ALNS (tuned) |
| B1+SP | Classical ALNS + set partitioning |
| P (first method) | For reference only |
| Free rule | Same prior mechanism, $\hat C$ from static relatedness |
| Oracle | Same prior mechanism, $\hat C$ from long runs of the test instance (upper bound, not a method) |

**Ablations**

- Heterogeneous GNN / homogeneous GNN / no message passing (RQ2).
- Prior weight $\lambda \in \{0.2, 1\}$ (chosen on the tuning instances only).
- With and without set partitioning.
- Teacher strength ($M$) and number of teacher runs ($K_T$), if G4 passes.

**Protocol**

- Tuning and all gates use generated instances only. Benchmarks are used once, at the end.
- Equal time on one CPU thread, forward pass and set partitioning inside the budget [S, Accorsi et al. 2022]. In addition, equal-iteration runs with the iteration clock (deterministic, not sensitive to machine load).
- Budget sweep (0.25×, 1×, 4× the standard budget) to show how the effect changes when the search is given more time (RQ4).
- Metrics: score against the best-known solution; cumulative vehicles and distance for Li & Lim; % gap for H-DARP; share of feasible runs (RQ3).
- Statistics: paired Wilcoxon signed-rank test on instance means, Holm correction for families of tests, Vargha–Delaney $\hat A_{12}$, win / tie / loss counts.

## 7. What is supported and what is still a hypothesis

| Statement | Status |
|---|---|
| A prior from a stronger teacher improves ALNS at equal iterations | [O] E08c, E09b |
| A weaker or equal teacher does not help | [O] E08c |
| Pruning from predictions is harmful; a soft prior is safe | [O] E06, E09b; [S] Liu et al. 2025 |
| One-shot, instance-level predictions avoid the inference cost that cancels gains | [S] Xin et al. 2021; Santana et al. 2023; [O] run time ×1.10 |
| Set partitioning over the route pool improves the base | [S] four papers; [O] E10 |
| A GNN can learn enough of the long-run co-routing to keep a useful part of the oracle gain | **[H]** gates G3–G5. Risk: long runs agree only weakly with each other on H-DARP style instances (E09c) |
| The heterogeneous model is better than the homogeneous one for this target | **[H]** ablation in E13–E14 |
| The gain holds at larger budgets and on the benchmarks | **[H]** G5. Expected to shrink as the budget grows |

## 8. Limits to state in any paper

- Our base ALNS is far from the published level (H-DARP: 6.2% gap against 0.5–1.1% for published C++ methods in similar time). Effects of guidance are measured relative to this base and may differ on a faster solver.
- The largest possible gain with perfect advice is about 1.3% (H-DARP style, equal iterations). A learned model can only reach part of it.
- Teacher runs cost 12 normal runs per training instance.
- On Li & Lim style instances the score is dominated by the number of vehicles, so means are noisy; medians and win / loss counts are reported with them.

## 9. Outcome of the gates (14 November 2026)

**The learned part of the revised method did not pass.** This section is the current state of the methodology.

### 9.1 What was found

| Question | Answer | Evidence |
|---|---|---|
| Can a network trained on strong teachers predict the co-routing of long runs? | No better than the old network or a distance rule | E13: H-DARP style AUC 0.564 (old head 0.563, static rule 0.537). A single run of normal length reaches 0.606; other long runs 0.658 |
| Does the teacher-trained prior help the search? | No | E14: +0.19% (n.s.) with a weak prior; +5.49% with a strong prior (27 of 28 instances worse) |
| Heterogeneous against homogeneous model (RQ2) | No difference | E13: AUC 0.564 and 0.564; E14: both harmful with a strong prior |
| Why does the oracle help when the consensus of three long runs does not? | The useful information is **one specific better solution**. What good solutions have in common is too little to guide the search | E14: one long-run solution as strong prior −1.33%; the average of three long runs with the same prior +1.98%. E09c: long runs of one instance agree only weakly (AUC 0.66) |

### 9.2 What this means for the research questions

- **RQ1.** No tested form of learned guidance beats classical ALNS at equal effort: route prior with pruning (E04–E06), learned choice of the destroy neighbourhood (upper bound, E08a), static co-routing prior from a weak or a strong teacher (E08b, E14), per-instance configuration (E11).
- **RQ2.** The heterogeneous model is never better than the homogeneous one (E04, E13).
- **RQ3.** All solutions are feasible in every experiment, because feasibility is always checked exactly.
- **RQ4.** Guidance helps only when it carries a specific solution that is clearly better than what the search reaches by itself (4× teacher). A static prediction from the instance cannot carry this.

### 9.3 The method that stands

**Classical ALNS + set partitioning over the route pool** (Section 4.5). It is the only tested change that improves the results without a loss anywhere (E10: 16 better / 0 worse on Li & Lim style; 11 better / 0 worse on H-DARP style). It is a known technique, not a new one. Benchmark confirmation (E15, 140 instances, 5 seeds, same iterations): never worse; Li & Lim 100 −0.73 (p = 0.008), Li & Lim 200 −0.87 (48 better / 0 worse, p < 0.001), H-DARP −0.04 (14 better / 0 worse); solver time 0.0–4.9 s. The gain is small but costs almost nothing.

### 9.4 What the project contributes as it stands

1. A controlled negative result for learned route guidance in ALNS on heterogeneous pickup-and-delivery and dial-a-ride problems, with the causes separated (overhead, transfer, label, teacher, pruning).
2. **Teacher-strength effect** (E08c): advice from a teacher that is not stronger than the search is neutral or harmful; advice from a 4× teacher helps.
3. **Specific solution against shared structure** (E09c, E14): the gain of oracle advice comes from one specific solution; the structure shared by good solutions is weak (AUC 0.66) and does not help. This explains why static learned priors fail here.
4. A "limit first" protocol: before training anything, measure with an oracle how much a perfect model could give (E08–E09), and compare every predictor with the cheap rule it would replace (E07, E13). The per-instance configuration test (E11) shows how the same data would mislead without a held-out seed (−0.59% apparent, −0.15% real).

### 9.5 Options that remain (not yet tested)

| Option | Why it could work | Cost | Status |
|---|---|---|---|
| Faster repair loop (more iterations in the same time) | 4× iterations give −1.24% on H-DARP style; this is the largest lever we measured | Engineering | Not started |
| Set partitioning called several times during the search, with restarts from its solution | Published for H-DARP (Malheiros et al. 2021) | Small | Not started |
| Guidance for *recurring* instances (yesterday's solution as advice for a similar instance today) | The oracle result shows that a specific good solution helps; in daily operations such a solution exists | New problem setting; outside the original proposal | Idea only [H] |
| Learned construction of complete solutions | Would give a specific solution | Far beyond a laptop; published constructive models are behind ALNS at this size [S] | Not planned |
