# Literature review

Last update: 13 November 2026. About 90 papers were checked in three threads; 72 are in the reference list at the end.

- **Part 1** compares the papers and says what they mean for our method (tables 3.1–3.3).
- **Part 2** holds the detailed notes per paper (what is learned, training signal, data, budget, result, limits).
- **Part 3** is the reference list, with a mark for how each paper was checked.

Marks used in Part 1: **[S]** = supported by a published paper that we checked; **[O]** = our own measurement; **[H]** = hypothesis that still needs an experiment.

---

# Part 1. Comparison and what it means for us

We checked about 90 papers in three groups: (A) machine learning inside large neighbourhood search for routing, (B) cases where learned guidance did not help and how pruning should be designed, (C) the best classical methods for PDPTW and DARP. The full reference list, with links and a note on how each paper was checked, is in Section 10.

Reading rule for this section: **[S]** = the statement is supported by a published paper that we checked; **[O]** = our own measurement; **[H]** = a hypothesis that still needs an experiment.

## 3.1 Group A: what is learned, how, and with what result

**Table 3.1** – Learning inside LNS for routing problems. "Classical" means a strong non-learning solver (HGS, LKH-3, SISR).

| Paper | Problem and size | What is learned | Training signal | Budget and hardware | Reported result | Limits |
|---|---|---|---|---|---|---|
| Hottung et al. 2025 (NDS) | CVRP, VRPTW; 100–2000 customers | **Destroy** (which customers to remove); insertion is fixed and greedy | Reinforcement learning, reward = cost improvement | 5–240 s; 1 CPU core + 1 GPU | vs HGS: +0.04% at 100 customers, −0.20% at 500, −2.34% at 2000 (CVRP); VRPTW −0.19% to −3.50% | Classical SISR is almost as good at large sizes; HGS wins at 100 customers; needs a GPU and days of training |
| Li, Yan, Wu 2021 (L2D) | CVRP, CVRPTW; 500–3000 | **Which sub-problem** to re-optimise | Supervised regression on the *realised improvement* | 1 CPU at test time | 7–15× faster than LKH-3 to reach the same quality | Learned choice is only 1.5–2× better than random choice; worse than random on a different distribution |
| Feijen et al. 2024 (LENS) | VRPTW; 1000 customers | **Which of 10 candidate neighbourhoods** to destroy | Supervised "did the repair improve?" labels, 5 on-policy rounds | 200–500 iterations | Closes 4–12% of the gap between random choice and a perfect choice | First model (trained on random-policy data) was *worse than random* in the search although it was good offline |
| Ouyang et al. 2025 (L2Seg) | CVRP, VRPTW; 1000–5000 | Which edges will change (the rest is frozen) | Supervised, labels = what the solver really changes in one step | 120–600 s | 2–7× speed-up of the backbone solvers | Larger gains for weaker backbone solvers |
| Herdianto et al. 2025 (node destroyer) | CVRP; 100–1000+ | Which nodes to protect from removal | **Imitation of high-quality solutions** | – | Gap 0.3012% → 0.2942% | Gain of hundredths of a percent; about 34% more run time in one setting |
| Zhou et al. 2023 | Heterogeneous multi-fleet routing with time windows; 20–200 flights | Destroy (which vehicles to free); repair by MIP solver | Imitation of the best of 10 random destroys, on-policy | 1 min – 2 h | Small gains over random destroy (e.g. 14.24% vs 14.66% gap) | Plain OR-Tools wins on small instances |
| Hottung & Tierney 2020 (NLNS) | CVRP; up to 297 | **Repair** | Reinforcement learning | 190–615 s; GPU + 10 cores | Beats LKH-3 on 11 of 17 groups | HGS is better on all 17 groups |
| Falkner & Schmidt-Thieme 2023 (NRR) | CVRP; 500–4000 | Repair (neural construction) + sub-problem scorer | RL + regression on improvement | 60–480 s; GPU | Best at 2000 customers | Loses to LKH-3 at 500 and 1000 |
| Li et al. 2025 (DRHG) | TSP, CVRP; 100–10,000 | Repair | Supervised | 1000 iterations; GPU | Good on TSP | 7.66% worse than LKH-3 on CVRP with 1000 customers |
| Syed et al. 2019 | Ride hailing | Insertion operator inside LNS | Imitation of LNS solutions | – | "Good solutions" (abstract only) | No comparison with classical LNS verified. Closest relative of our design |
| Reijnen et al. 2024 (DR-ALNS) | Orienteering, CVRP; 20–100 | Operator choice + removal size + acceptance | Reinforcement learning | 100–200 iterations | 1–3% better than tuned ALNS | Operator choice alone gives almost nothing (8.28 vs 8.23); very short budgets |
| Yu et al. 2026 | CVRP, VRPTW; < 80 | Operator choice (GCN state) | Actor-critic | – | −0.11% to −0.49% vs ALNS | 1.8–2.5× the run time; not significant on CVRP |
| Molinghen et al. 2026 | **PDPTW (Li & Lim, 100–600)**, TSP | Which move neighbourhood to use | Bandits vs deep RL with a GNN | 15 min; CPU + GPU | A hand-made rule was best; ε-greedy close | "Deep RL only pays off with much longer run times because of inference overhead" |
| Bongiovanni et al. 2022 | **Dynamic electric DARP** | Which destroy–repair pair to apply | Supervised prediction of improvement (> 1.5 million examples) | Online, short | Up to about 9% better than other data-driven metaheuristics | Very short budgets; baselines not fully verified |
| Ma et al. 2022 (N2S) | Pickup and delivery, **single vehicle**, no time windows | Removal + reinsertion of a pair | Reinforcement learning | 1000–3000 steps; 8 GPUs | −0.20% vs LKH-3 with augmentation | 4.44% worse without augmentation; no route choice at all |

What Table 3.1 shows:

1. **[S] Learning *what to remove* works more often than learning *where to re-insert*.** The positive results at scale (NDS, L2D, L2Seg) all keep a simple, exact repair. Learned repair (NLNS, NRR, DRHG) does not beat strong classical solvers below about 2000 customers.
2. **[S] The training signal that works is the *realised improvement* of the real destroy–repair step**, collected on the states that the guided search itself visits (L2D, LENS rounds 2–5, L2Seg, Bongiovanni et al.).
3. **[S] Imitating good solutions gives very small gains** (Herdianto et al.: 0.007 percentage points) or none (LENS round 1, Syed et al.).
4. **[S] Operator selection has little room.** A meta-analysis of 25 ALNS implementations finds that the whole adaptive layer is worth 0.14% on average (Turkeš et al. 2021).
5. **[S] Size matters.** At 100–200 customers, classical solvers are equal or better. Clear wins start at about 500–2000 customers, with short fixed budgets and usually a GPU.
6. **[S] We found no published paper in which a learned repair or a learned route choice beats a tuned ALNS on a multi-vehicle PDPTW or DARP.** Our negative result is therefore not in conflict with the literature.

## 3.2 Group B: when learned guidance does not help

**Table 3.2** – Published negative or critical evidence that matches our findings.

| Paper | Setting | Finding | What it means for us |
|---|---|---|---|
| Santana, Lodi, Vidal 2023 | CVRP, HGS; 10,000 instances with 100 customers, and 100–1000 customers | A GNN heatmap is *not better than plain distance* as relatedness measure. For neighbourhood filtering the learned information "even appeared to be detrimental". GNN inference takes about as long as solving a 100-customer instance | The closest published match of our result (W1, W4, W5) |
| Xia et al. 2024 (ICML) | TSP 500–10,000, heatmap + tree search | A one-parameter softmax of the distance beats most learned heatmaps | A learned prior must be compared with a tuned cheap prior |
| Pan et al. 2024 | TSP, heatmap + tree search | Search tuning matters more than the heatmap; a k-nearest prior matches learned heatmaps | Tune the search separately for every prior |
| Feijen et al. 2024 (LENS) | VRPTW 1000 | Offline hit rate 44% vs 18%, but in the search the first model scored 110% where random = 100% and perfect = 0% | Offline accuracy does not predict search gain (our Table 2.2 vs Table 2.1) |
| Turkeš et al. 2021 | 25 ALNS implementations | Adaptive layer: +0.14% (95% CI 0.07–0.22%) | Guidance layers on a good ALNS have small room |
| Liu et al. 2023 | TSP, systematic evaluation | Learned solvers fall short of classical solvers in nearly all aspects | Losing to a classical component is the expected case |
| Joshi et al. 2022 | TSP | Models do not generalise to larger sizes without changes | Report accuracy per size (our accuracy drops from 0.43 to 0.19 on large H-DARP) |
| Sun et al. 2021 | TSP, problem reduction | Pruning accuracy falls away from the training distribution | Measure the recall of the pruned set on the test sizes |
| Liu et al. 2025 (Apollo-MILP), Han et al. 2023, Yoon et al. 2023 | MIP | Fixing variables from inaccurate predictions gives poor solutions; better: search *around* the prediction, fix only confident predictions, and tune how much is fixed against the final objective | Hard top-$K$ pruning with a 9% miss rate is the wrong way to use a prediction (W4) |
| Bengio, Lodi, Prouvost 2021 | Survey | ML should replace decisions that are "too expensive to compute or mathematically not well defined" | Route choice in repair is neither: it is cheap and exact in ALNS (W1) |
| Ross, Gordon, Bagnell 2011 (DAgger) | Imitation learning | A policy trained on the expert's states fails on the states it produces itself; the fix is to add data from its own states | Our network is trained only on destroyed *elite* solutions (W2) |

Positive designs in the same group, and why they work:

| Paper | Design | Why it works |
|---|---|---|
| Xin et al. 2021 (NeuroLKH) | Network builds the candidate edge set of LKH, trained on (near-)optimal tours, computed **once per instance** | Replaces a *weaker* default (not an exact one); costs no time during search; teacher is much better than the search |
| Hudson et al. 2022 | Predict the **regret** of an edge, use it as a penalty in guided local search | Regret separates "wrong but harmless" from "wrong and costly"; membership cannot |
| Morabit et al. 2021, 2023 | Learned column and arc selection in column generation (up to 30–40% time saved) | A wrong selection only costs iterations, never solution quality |
| Arnold et al. 2021 (PILS), Maia et al. 2024 | Frequent patterns mined **during the run** from its own elite solutions, injected as moves that are accepted only if they improve | Instance-specific, online, and cost-checked; no option is removed |
| Toth & Vigo 2003, Accorsi & Vigo 2021 | Granular (sparsified) neighbourhoods based on **cost** | Pruning uses a reliable cheap measure and adapts to the current solution |

## 3.3 Group C: how strong is our classical baseline?

**Table 3.3** – Published classical methods on our benchmarks, compared with our classical ALNS.

| Method | Benchmark | Budget and hardware | Result |
|---|---|---|---|
| Ropke & Pisinger 2006, ALNS (C++) | Li & Lim 100 | 25,000 iterations (+ up to 25,000 for vehicle minimisation); 66 s on a 1.5 GHz Pentium IV | 402 vehicles (best of 10), 403 (average) over 56 instances; best-known = 402 |
| Ropke & Pisinger 2006 | Li & Lim 200 | same; 264 s | 606 vehicles (best), 608 (average) over 60 instances |
| Curtois et al. 2018, LNS + guided ejection search | Li & Lim 100–1000 | 5 min (100), 15 min (200); one 3.5 GHz core | All best-known matched at 100; 35 matched + 7 new at 200 |
| Sartori & Buriol 2020, + set partitioning | Li & Lim; new set up to 5000 locations | 5 min / 15 min | Holds 6 best-known solutions at 200 |
| **Our classical ALNS (Python + Numba)** | Li & Lim 100 / 200 | 25 s / 50 s; about **3,600 / 1,800 iterations** | 100: +0.03 vehicles per instance. 200: **+0.40 vehicles per instance** (about +24 over the 60 instances), +2.5% distance |
| Braekers et al. 2014, deterministic annealing | H-DARP, our 24 instances (as re-run by Malheiros et al.) | mean 71.6 s | **1.10%** average gap to best-known |
| Malheiros et al. 2021, ILS + set partitioning (C++) | H-DARP, our 24 instances | mean 75.4 s, one thread | **0.50%** average gap |
| **Our classical ALNS (Python + Numba)** | H-DARP, same 24 instances | mean 62 s; about **3,200 iterations** | **6.18%** average gap |
| Gschwind & Drexl 2019, ALNS (for scale) | Standard DARP (Cordeau set) | 50,000 iterations in 54.7 s | 0.88% average gap |

(The H-DARP rows for Braekers et al. and Malheiros et al. were read by us directly in Table 6 of the Malheiros paper.)

What Table 3.3 shows:

1. **[S][O] Our base solver is far from the published level.** On H-DARP, published methods are at 0.5–1.1% in about the same wall-clock time; we are at 6.2%. The main reason is speed: published ALNS runs make 25,000–50,000 iterations, ours makes 2,000–4,000 (Python with a compiled insertion search; the repair loop is not compiled).
2. **[O] In this regime, "more iterations" is worth much more than any guidance we tested.** Four times more iterations improves the average score from 17.1 to 9.8 (Table 2.3). The guidance effect we were looking for is in the range of ±0.5.
3. **[S] Li & Lim 100 is saturated.** All best-known solutions date from 2001–2004 and good methods match all of them. It is a sanity check, not a place where a new method can show a gain.
4. **[S] The best classical methods use components that our ALNS does not have.** The best supported ones are:
   - **Set partitioning over a pool of routes** found during the search (Parragh & Schmid 2013: about +1.7% quality; Gschwind & Drexl 2019: "always reduces the average gaps considerably"; Malheiros et al. 2021; Sartori & Buriol 2020).
   - **Guided ejection search for vehicle minimisation** on PDPTW (Nagata & Kobayashi 2010; Curtois et al. 2018; Sartori & Buriol 2020).
   - **String removal with "blinks"** (Christiaens & Vanden Berghe 2020; adapted to PDPTW by Hiermann & Schiffer 2024).
   - **A constant-time ride-time feasibility test** for DARP (Gschwind & Drexl 2019: 32× faster on average than the eight-step procedure we use).
   - **Record-to-record or threshold acceptance** (Santini et al. 2018).
5. **[S] Testing guidelines** (Accorsi, Lodi, Vigo 2022) ask for strong, efficiently implemented baselines. A learned component that helps a slow baseline may not help a fast one, and the other way round. So a credible test of any guidance needs a faster base solver first.

## 3.4 How the literature explains our result

| Our finding | Published explanation | Papers |
|---|---|---|
| Network (0.43) is less accurate than exact insertion cost (0.70) | Learned priors often add nothing beyond cheap cost information | Santana et al. 2023; Xia et al. 2024; Pan et al. 2024 |
| Good offline accuracy, no gain in the search | Offline accuracy is a surrogate; the search visits other states | Feijen et al. 2024; Ross et al. 2011; Yoon et al. 2023 |
| Reconstruction labels do not help | Positive designs predict improvement or regret, or use many solutions | Li et al. 2021; Hudson et al. 2022; Arnold et al. 2021; Herdianto et al. 2025 |
| Pruning hurts | Hard fixing with imperfect recall removes good moves; working designs stay soft or keep a full fallback | Liu et al. 2025; Han et al. 2023; Morabit et al. 2021 |
| No speed gain | Inference overhead cancels the saving at small sizes | Santana et al. 2023; Molinghen et al. 2026; Yu et al. 2026 |
| No gain at 50–100 requests | Learned LNS wins at ≥ 500 customers; classical wins at ≤ 200 | Hottung et al. 2025; Li et al. 2021; Falkner & Schmidt-Thieme 2023 |
| Oracle advice helps | Advice from a much stronger teacher, used once per instance, can help | Xin et al. 2021; Arnold et al. 2021 |

---

# Part 2. Detailed notes per paper

## Literature thread A: ML inside LNS / ALNS for routing (verified by a search agent on 2026-11-13)

Flags: [full/abs page opened] unless "abstract only" / "not verified". Numbers were read by a fetch summariser: spot-check before quoting in the paper.

### Learned destroy / neighbourhood selection
- NDS: Hottung, Wong-Chung, Tierney, "Neural Deconstruction Search for Vehicle Routing Problems", TMLR 2025, arXiv:2501.03715. CVRP/VRPTW/PCVRP n=100-2000. Learns destroy only (RL), greedy insertion fixed. 5/60/120/240 s per instance, 1 CPU core + 1 A100 GPU. vs HGS/PyVRP-HGS: CVRP +0.04% (100), -0.20% (500), -0.90% (1000), -2.34% (2000); VRPTW -0.19%, -2.17%, -3.14%, -3.50%. Caveats: SISR nearly as good at large n (CVRP2000 -2.27%, VRPTW2000 -3.49%); HGS wins at CVRP100; GPU; 5-15 days training.
- Gao et al., "Learn to Design the Heuristics for VRP", arXiv:2002.08539 (preprint 2020). GAT (EGATE) destroy + insertion order, PPO. Unbatched 1K iterations worse than SISR-1K on CVRP and worse than ALNS-1K on CVRPTW; best numbers need batch minimum.
- Chen et al., "Dynamic Partial Removal", arXiv:2005.09330 (preprint 2020). CVRPTW; learned destroy; only 150 iterations; ~10% better than SISR at that tiny budget.
- LENS: Feijen, Schäfer, Dekker, Pieterse, "Learning-Enhanced Neighborhood Selection for the VRPTW", arXiv:2403.08839 (2024 preprint). GH R1 1000 customers. Random forest picks 1 of 10 candidate neighbourhoods; supervised improvement labels; 5 on-policy rounds. Closes 11.8% (200 it.) / 4.26% (500 it.) of the random-to-oracle gap = about 0.12% / 0.03% distance. ML1 (random-policy data only) no better than random. Offline hit rate 44% vs 18% but tiny end-to-end gain.
- Feijen, Dekker, van Zwam, "Learn to create neighborhoods in real-world VRP", LOD 2024 (abstract only, https://ir.cwi.nl/pub/35900). Real PDPTW with compatibility; RL destroy; ~9% fewer iterations than random in the realistic phase.
- L2D: Li, Yan, Wu, "Learning to Delegate for Large-scale Vehicle Routing", NeurIPS 2021, arXiv:2107.04139. n=500-3000. Regression on realised sub-solver improvement. 7x-15x faster than LKH-3 to reach its 95% quality; learned selector adds only 1.5x-2x over random selection; uniform-trained model worse than random on real-world distribution.
- L2Seg: Ouyang, Li, Ma, Wu, "Learning to Segment for VRPs", arXiv:2507.01037 (2025). n=1k-5k; labels = edges the solver actually changes (one-step lookahead).
- Herdianto, Billot, Lucas, Sevaux, Vigo, "Hybrid Node-Destroyer Model with LNS for CVRP", arXiv:2508.08659 (2025). GNN protects nodes from removal inside HGS-PILS / FILO2; imitation of HGS solutions. Gap 0.3012% -> 0.2942% (X); FILO2 1.086% -> 1.059% (B) with ~34% more runtime.
- Zhou, Wu, Cao, Song, Zhang, Chen, "Learning LNS for Vehicle Routing in Airport Ground Handling", arXiv:2302.13797 (TKDE 2023 per our kb). Heterogeneous multi-fleet; GCN destroy (imitation of best-of-10 random destroys, forward training) + CPLEX repair; small gains over random-vehicle LNS (primal gap 14.24% vs 14.66% at 50 flights; 20.86% vs 24.21% at 100).

### Learned repair
- NLNS: Hottung & Tierney, ECAI 2020, DOI 10.3233/FAIA200124, arXiv:1911.09539. Learned repair (RL). Beats LKH3 on 11/17 groups; UHGS best on all 17; GPU + batch 300.
- Falkner, Thyssens, Schmidt-Thieme, "LNS based on Neural Construction Heuristics", arXiv:2205.00772 (2022). VRPTW; within ~1.1x BKS; no head-to-head.
- NRR: Falkner & Schmidt-Thieme, "Too Big, so Fail?", arXiv:2309.17089 (2023). Loses to LKH3 at n=500/1000; wins at 2000.
- DRHG: Li, Liu, Wang, Zhang, AAAI 2025, DOI 10.1609/aaai.v39i17.34018. Supervised repair; CVRP1K 7.66% worse than LKH3.
- Syed, Akhnoukh, Kaltenhaeuser, Bogenberger, EPIA 2019, DOI 10.1007/978-3-030-30241-2_49 (abstract only). Imitation-trained insertion inside LNS for ride hailing; no numbers verified.

### Operator selection / control
- Turkeš, Sörensen, Hvattum, EJOR 292(2) 2021, DOI 10.1016/j.ejor.2020.10.045: adaptive layer +0.14% (95% CI 0.07-0.22%), 25 implementations.
- Kallestad, Hasibi, Hemmati, Sörensen (DRLH), EJOR 309(1) 2023, DOI 10.1016/j.ejor.2023.01.017 (abstract only).
- DR-ALNS: Reijnen, Zhang, Lau, Bukhsh, ICAPS 2024, DOI 10.1609/icaps.v34i1.31507. 100-200 iterations; operator selection alone ~ no gain (8.28 vs 8.23), full control 8.44.
- Johnn, Darvariu, Handl, Kalcsics, OLA 2023, arXiv:2302.14678. Toy scale (20 customers).
- Yu, Mao, Wu, Puchinger, arXiv:2601.11414 (2026): -0.11% to -0.49% vs ALNS at 1.8-2.5x runtime; CVRP difference not significant.
- Bongiovanni, Kaspi, Cordeau, Geroliminis, TR-E 165:102835, 2022, DOI 10.1016/j.tre.2022.102835. Dynamic e-ADARP; random forests predict improvement of destroy-repair pairs; >1.5M examples; up to ~9% vs data-driven metaheuristic baselines.
- Molinghen, Delecluse, De Landtsheer, Michelini, "RL Methods for Neighborhood Selection in Local Search", ICORES 2026, arXiv:2601.07948. PDPTW Li & Lim (100-600), 15 min: hand-made best-slope-first rule best; deep RL only pays off with much longer runtimes; epsilon-greedy strong.

### PDP-specific neural
- N2S: Ma et al., IJCAI 2022, DOI 10.24963/ijcai.2022/662. Single vehicle, no TW; -0.20% vs LKH3 with augmentation, 4.44% worse without.
- Li, Xin, Cao, Lim, Song, Zhang, IEEE T-ITS, DOI 10.1109/TITS.2021.3056120 (heterogeneous attention PDP; construction; weak baselines).
- Jacobsen et al., arXiv:2601.22052 (2026, abstract only): e-DARP construction; claims 9.5% better than ALNS (not equal time).
- Zhang et al., arXiv:2512.11187 (2025): network seeds multi-start LNS and gives a removal prior.

### MIP LNS
- Song, Lanka, Yue, Dilkina, NeurIPS 2020, arXiv:2004.00422 (imitation with forward training of best-of-5 random decompositions).
- Wu, Song, Cao, Zhang, arXiv:2111.03466 (RL destroy; advantage at short budgets).
- Sonnerat et al., arXiv:2107.10201 (imitation of local branching; random competitive on 2 of 5).
- CL-LNS: Huang, Ferber, Tian, Dilkina, Steiner, ICML 2023, PMLR 202:13869-13890, arXiv:2302.01578 (contrastive positives+negatives beat imitation and RL).

### Synthesis (agent)
Works: learn WHAT TO DESTROY / which sub-problem, with fixed strong repair; train on REALISED IMPROVEMENT of the real destroy-repair step; ON-POLICY data; negatives (contrastive); many GPU rollouts. Small/failed: operator selection; learned repair alone; imitation of elite solutions to restrict search. Sizes: at n<=100-200 classical HGS/SISR equal or better; wins at n>=500-2000 under 1-10 min, usually GPU. No verified paper where learned repair/route assignment beats tuned ALNS on multi-vehicle PDPTW/DARP.

## Literature thread B: when learned guidance does not help; pruning; training signal (verified by a search agent on 2026-11-13)

Flags: FT = full text read (at least in part); ABS = abstract only; NV = detail not verified.

### Critical / negative evidence
- Santana, Lodi, Vidal, "Neural Networks for Local Search and Crossover in Vehicle Routing: A Possible Overkill?", CPAIOR 2023, LNCS pp. 184-199, DOI 10.1007/978-3-031-33271-5_13, arXiv:2210.12075 [FT]. CVRP, HGS; XML set (10,000 instances, 100 customers, 5 s) and X set. Baseline HGS gap 0.030%; distance-guided crossover 0.024%; GNN-guided crossover 0.023-0.024%; GNN granular search ~0.031% (not better than baseline even ignoring inference time). X: 0.368% baseline, 0.302% distance crossover, 0.299% GNN crossover. Learned information "even appeared to be detrimental in the context of the LS". GNN inference ~0.85 s, same order as solving a 100-customer instance.
- Xia, Yang, Liu, Liu, Song, Bian, "Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale TSP", ICML 2024, PMLR 235, arXiv:2406.03503 [FT part]. SoftDist (softmax of negative distance, 1 parameter) beats most learned heatmaps for MCTS guidance; heatmap-MCTS inferior to LKH-3 at equal resources.
- Pan, Wang, Ying, Xue, Yu, "Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers", arXiv:2411.09238 [ABS]. Search configuration matters more than heatmap; k-nearest prior with tuned MCTS matches learned heatmaps.
- Accorsi, Lodi, Vigo, "Guidelines for the computational testing of machine learning approaches to vehicle routing problems", Operations Research Letters 50(2):229-234, 2022, DOI 10.1016/j.orl.2022.01.018, arXiv:2109.13983.
- Liu, Zhang, Tang, Yao, "How Good Is Neural Combinatorial Optimization? A Systematic Evaluation on the TSP", IEEE Computational Intelligence Magazine 18(3):14-28, 2023, DOI 10.1109/MCI.2023.3277768 [ABS]. Learned solvers fall short of traditional solvers in nearly all aspects.
- Joshi, Cappart, Rousseau, Laurent, "Learning the Travelling Salesperson Problem Requires Rethinking Generalization", Constraints 27:70-98, 2022, DOI 10.1007/s10601-022-09327-y, arXiv:2006.07054 [ABS]. Poor size generalisation.
- Turkeš, Sörensen, Hvattum, EJOR 292(2):423-442, 2021, DOI 10.1016/j.ejor.2020.10.045: adaptive layer +0.14% (95% CI 0.07-0.22%).
- Santini, Ropke, Hvattum, "A comparison of acceptance criteria for the ALNS metaheuristic", Journal of Heuristics 24(5):783-815, 2018, DOI 10.1007/s10732-018-9377-x [ABS]. SA, threshold acceptance, record-to-record best; one RRT variant consistently undominated.
- Lucas, Billot, Sevaux, "A comment on 'What makes a VRP solution good?'", C&OR 110:130-134, 2019, DOI 10.1016/j.cor.2019.05.025 [ABS]: most of the ML-extracted knowledge is recoverable with PCA.

### Search-space reduction / pruning
- Toth & Vigo, "The Granular Tabu Search and Its Application to the VRP", INFORMS J. Computing 15(4):333-346, 2003, DOI 10.1287/ijoc.15.4.333.24890 [ABS]. Drastically restricted neighbourhoods (cost-based). NV: threshold widened on stagnation.
- Accorsi & Vigo, FILO, Transportation Science 55(4):832-856, 2021, DOI 10.1287/trsc.2021.1059 [ABS]. Localised, controlled, tailored optimisation. NV: per-vertex sparsification that expands on non-improvement.
- Arnold & Sörensen, "What makes a VRP solution good?", C&OR 106:280-288, 2019, DOI 10.1016/j.cor.2018.02.007 [ABS].
- Arnold & Sörensen, KGLS, C&OR 105:32-46, 2019, DOI 10.1016/j.cor.2019.01.002 [ABS]. Knowledge used as penalties on undesirable edges; no component of unclear contribution.
- Arnold, Santana, Sörensen, Vidal, PILS, Pattern Recognition 116:107957, 2021, DOI 10.1016/j.patcog.2021.107957, arXiv:1912.11462 [ABS]. Frequent patterns from the run's own elite solutions injected as moves, accepted only on improvement; significantly improves HGS.
- Sun, Ernst, Li, Weiner, "Generalization of machine learning for problem reduction: a case study on TSP", OR Spectrum 2021, DOI 10.1007/s00291-020-00604-x [ABS]. Accuracy deteriorates away from the training set.
- Fitzpatrick, Ajwani, Carroll, "Learning to Sparsify TSP Instances", CPAIOR 2021, DOI 10.1007/978-3-030-78230-6_26 [ABS]. Prunes >85% of variables using LP / cutting-plane / MST features.
- Xin, Song, Cao, Zhang, NeuroLKH, NeurIPS 2021, arXiv:2110.07983 [ABS]. Learned candidate set replaces LKH's default; also CVRP, PDP, CVRPTW.
- Fu, Qiu, Zha, AAAI 2021, DOI 10.1609/aaai.v35i8.16916 [ABS].
- Kool, van Hoof, Gromicho, Welling, DPDP, CPAIOR 2022, DOI 10.1007/978-3-031-08011-1_14 [ABS].
- Hudson, Li, Malencia, Prorok, "GNN Guided Local Search for the TSP", ICLR 2022, arXiv:2110.05291 [FT part]. Predicts REGRET of edges, used as GLS penalty guide; 100-node gap 1.534% -> 0.705% (vs learning baselines).
- Morabit, Desaulniers, Lodi, "ML-Based Column Selection for Column Generation", Transportation Science 55(4):815-831, 2021, DOI 10.1287/trsc.2021.1045 [ABS]. Up to 30% time gain; wrong selection costs only iterations.
- Morabit, Desaulniers, Lodi, "ML-Based Arc Selection for Constrained Shortest Path Problems in CG", INFORMS J. Optimization 5(2):191-210, 2023, DOI 10.1287/ijoo.2022.0082 [ABS]. Up to 40% time reduction.
- Lauri, Dutta, Grassia, Ajwani, arXiv:2001.01230 [ABS]. Interpretable pruning models.
- Herdianto, Billot, Lucas, Sevaux, Vigo, "Edge-Selector Model Applied for Local Search Neighborhood for Solving VRPs", arXiv:2508.14071 (2025) [ABS]. Tabular classifier vs GNN, tuned decision threshold.

### Training signal
- Ross, Gordon, Bagnell (DAgger), AISTATS 2011, arXiv:1011.0686 [ABS].
- LENS (Feijen et al., arXiv:2403.08839) [FT]: scale oracle = 0%, random = 100%. First model (random-policy data) 110.04% after 500 iterations = WORSE than random although it validated well offline ("a part of the solution space that was not encountered during the data collection"). After 5 rounds: 95.74% (500 it.), 88.20% (200 it.).
- L2D (Li, Yan, Wu, NeurIPS 2021) [ABS]: regression on improvement; 1.5x-2x over heuristic/random selection.
- Khalil, Le Bodic, Song, Nemhauser, Dilkina, "Learning to Branch in MIP", AAAI 2016, DOI 10.1609/aaai.v30i1.10080 [ABS]: learning-to-rank surrogate of strong branching, on-the-fly, instance-specific.
- Han et al., "A GNN-Guided Predict-and-Search Framework for MILP", ICLR 2023, arXiv:2302.05636 [ABS]: search in a ball around the prediction instead of fixing variables.
- Liu et al., Apollo-MILP, ICLR 2025, arXiv:2503.01129 [ABS]: fixing inaccurate predictions gives low-quality solutions; fix only with uncertainty-based error bound.
- Yoon, Choi, Yun, Lim, "Threshold-aware Learning to Generate Feasible Solutions for MIP", arXiv:2308.00327 [ABS]: classification accuracy vs primal bound discrepancy; coverage should be learned/tuned.
- Bengio, Lodi, Prouvost, EJOR 290(2):405-421, 2021, DOI 10.1016/j.ejor.2020.07.063 [ABS]: ML should replace decisions "too expensive to compute or mathematically not well defined".
- Karimi-Mamaghan et al., EJOR 296(2):393-422, 2022, DOI 10.1016/j.ejor.2021.04.032 [ABS].

### Elite-set guidance without neural nets
- Rochat & Taillard, J. Heuristics 1(1):147-167, 1995, DOI 10.1007/BF02430370 (adaptive memory; mechanism NV).
- Maia, Plastino, Souza, "An improved hybrid genetic search with data mining for the CVRP", Networks 83(4):692-711, 2024, DOI 10.1002/net.22213 [ABS]. Elite patterns for (re-)initialisation; significantly outperforms HGS.
- Sun, Li, Ernst, IEEE TPAMI 43(5):1746-1760, 2021, DOI 10.1109/TPAMI.2019.2954827 [ABS]. Statistics from sampled solutions (frequency in high-quality solutions, correlation with objective) for problem reduction.
- Cook & Seymour, "Tour Merging via Branch-Decomposition", INFORMS J. Computing 15(3):233-248, 2003, DOI 10.1287/ijoc.15.3.233.16078 [ABS].

### Not found
- No paper on conformal / calibrated pruning for routing heuristics.

### Speculative (agent): regret labels for PDP/DARP (TSP evidence only); calibrated top-p pruning in routing; learning-to-rank on cost features inside ALNS (MILP analogies only).

## Literature thread C: state of the art for PDPTW and (MD-H-)DARP (verified by a search agent on 2026-11-13)

Tags: FT = full text read; ABS = abstract/record page; SR = search-result summary only; NV = not verified.

### PDPTW
- Ropke & Pisinger, "An ALNS heuristic for the PDPTW", Transportation Science 40(4):455-472, 2006, DOI 10.1287/trsc.1050.0135 [FT of thesis-chapter version]. 25,000 LNS iterations (+ up to 25,000 in vehicle minimisation; abort if >=5 requests unplanned with no progress in 2,000 iterations). C++, 1.5 GHz Pentium IV. Li & Lim 100: best of 10 runs 402 vehicles / 58,060; average over runs 403 / 58,249; 66 s per run; 54 of 56 at least as good as previous best. Li & Lim 200: best 606 / 180,931; average 608 / 181,707; 264 s per run; 22 of 60 new best.
- Pisinger & Ropke, "A general heuristic for vehicle routing problems", C&OR 34(8):2403-2435, 2007, DOI 10.1016/j.cor.2005.09.012 [SR + thesis]. ALNS-25K / 50K.
- Shaw, CP-98, LNCS 1520:417-431, DOI 10.1007/3-540-49481-2_30 [SR].
- Li & Lim, ICTAI 2001 pp. 160-167; IJAIT 12:173-186 (2003) [SR].
- SINTEF TOP Li & Lim pages (https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/) [FT]. Objective: vehicles then distance. 100-task set: all BKS date from 2001-2004 (saturated). 200-task BKS holders include Li & Lim 13, SAM::OPT 9, Helsgaun 7, Sartori & Buriol 6, Curtois et al. 4, Ropke & Pisinger 4, Christiaens & Vanden Berghe 1; latest entries Nov 2019. Larger sets dominated by unpublished industrial solvers.
- Curtois, Landa-Silva, Qu, Laesanklang, "LNS with adaptive guided ejection search for the PDPTW", EURO J. Transp. Logist. 7(2):151-192, 2018, DOI 10.1007/s13676-017-0115-6 [FT, open access]. Local search (4 neighbourhoods) + non-adaptive LNS (Shaw 0.6 / random 0.4, regret-k) + late-acceptance hill climbing + AGES for route reduction + restarts. Budgets: 5 min (100), 15 min (200, 400), 30 min (600), 60 min (800, 1000); Xeon E5-1620 3.5 GHz single core, C#. 142 new BKS of 354; 100 set all matched; 200 set 35 matched + 7 new.
- Sartori & Buriol, "A study on the PDPTW: Matheuristics and new instances", C&OR 124:105065, 2020, DOI 10.1016/j.cor.2020.105065 [ABS]. ILS(AGES + LNS) + set partitioning over a route pool; 300 new instances (100-5000 locations, real travel times). Time limits 5/15/15/30/60 min (as cited by Hiermann & Schiffer).
- Christiaens & Vanden Berghe, "Slack Induction by String Removals for VRPs", Transportation Science 54(2):417-433, 2020, DOI 10.1287/trsc.2019.0914 [ABS]. One ruin (adjacent string removal), one recreate (greedy insertion with blinks), fleet minimisation; no adaptive layer.
- Nagata & Kobayashi, "Guided ejection search for the PDPTW", EvoCOP 2010, LNCS 6022:202-213, DOI 10.1007/978-3-642-12139-5_18 [SR]. Route minimisation.
- Bent & Van Hentenryck, "A two-stage hybrid algorithm for PDPTW", C&OR 33(4):875-893, 2006, DOI 10.1016/j.cor.2004.08.001 [SR].
- Hiermann & Schiffer, "A decomposition-based approach for large-scale pickup and delivery problems", arXiv:2405.00230 (2024) [FT]. SISR adapted to PDPTW (string removal variants, blinks 0.05, record-to-record acceptance citing Santini et al., AGES, decomposition); Sartori-Buriol set; 5/15/15/30/60 min, i9-9900 4 cores, Rust; 107 new best, 3.1% fewer vehicles than Sartori & Buriol. Argue Li & Lim is not a good test set.
- Vadseth, Andersson, Cordeau, Stålhane, "Mixed Integer Programs to Improve Solutions of VRPs with Intra-route Constraints", J. Heuristics 2026, DOI 10.1007/s10732-026-09583-8 [FT preprint]. Improved 203 of 300 Sartori-Buriol BKS, 0 Li & Lim.

### DARP
- Cordeau & Laporte, Transportation Research Part B 37(6):579-594, 2003 [ABS].
- Parragh & Schmid, "Hybrid column generation and LNS for the DARP", C&OR 40(1):490-497, 2013, DOI 10.1016/j.cor.2012.08.004 [FT]. Non-adaptive LNS (random operator choice); set covering over routes adds ~1.7% quality for ~3 min.
- Gschwind & Drexl, "ALNS with a constant-time feasibility test for the DARP", Transportation Science 53(2):480-491, 2019, DOI 10.1287/trsc.2018.0837 [FT of discussion paper]. 25k/50k/75k iterations; constant-time ride-time feasibility test 5-72x (avg 32x) faster than the eight-step scheme; pr set: pure ALNS 50k iterations 0.50% best / 0.88% avg gap in 54.7 s; with Balas-Simonetti + set covering 0.32% / 0.50% in 209.5 s. "Set covering at the end always reduces the average gaps considerably."
- Braekers, Caris, Janssens, Transportation Research Part B 67:166-186, 2014, DOI 10.1016/j.trb.2014.05.007 [ABS]. MD-H-DARP; deterministic annealing (DA).
- Parragh, Transportation Research Part C 19(5):912-930, 2011, DOI 10.1016/j.trc.2010.06.002 [ABS]. H-DARP, 4 resource types.
- Malheiros, Ramalho, Passeti, Bulhões, Subramanian, "A hybrid algorithm for the multi-depot heterogeneous dial-a-ride problem", C&OR 129:105196, 2021, DOI 10.1016/j.cor.2020.105196. ILS-RVND + set partitioning (ILS-SP), 20 restarts x max(2n, 200) ILS iterations, 10 runs, single thread Intel i5-9600KF 3.7 GHz.
  **Checked by us in the full text (PDF in research_kb/papers), Table 6, the 24 new instances that we use (group I, 9-16 vehicles, 72-192 requests):**
  DA (Braekers et al.): average gap of the mean to BKS 1.10%, mean CPU 71.6 s. ILS-SP with DA's time as limit: 0.57%. ILS-SP regular: 0.50%, mean CPU 75.4 s (17 s at a9-72 to 217 s at a16-192).
  Our classical ALNS on the same 24 instances: 6.18% gap, mean budget 62 s (pure Python).
  SP ablation (dissertation): gap 0.352% -> 0.228% with SP vs 0.316% from doubling restarts.
- Masmoudi et al. 2016/2017 [SR] (results excluded by Malheiros: some below lower bounds).
- Detti et al., Omega 70:1-14, 2017, DOI 10.1016/j.omega.2016.08.008 [SR].
- Mortazavi, Ghasri, Ray, PLoS ONE 19(2):e0292683, 2024 [FT]. Deterministic annealing; pr set 50k iterations in 17 s.
- Surveys: Molenbruch, Braekers, Caris, Annals of OR 259:295-325, 2017, DOI 10.1007/s10479-017-2525-0; Ho et al., Transportation Research Part B, 2018, DOI 10.1016/j.trb.2018.02.001.

### Hybridisation
- Subramanian, Uchoa, Ochi, C&OR 40(10):2519-2531, 2013, DOI 10.1016/j.cor.2013.01.013 [SR]: ILS-RVND-SP.
- Archetti & Speranza, EURO J. Comput. Optim. 2(4):223-246, 2014, DOI 10.1007/s13675-014-0030-7 [ABS]: matheuristics survey.
- Santini, Ropke, Hvattum 2018; Turkeš, Sörensen, Hvattum 2021 (see thread B).

### Components top methods use that a plain Ropke-Pisinger ALNS lacks (agent)
1 guided ejection search (AGES) for vehicle minimisation; 2 set partitioning/covering over a route pool (best-supported hybridisation); 3 constant-time ride-time feasibility test; 4 string removal with blinks (SISR); 5 other acceptance criteria (LAHC, record-to-record, threshold); 6 local search between ruin-recreate steps; 7 Balas-Simonetti intra-route optimisation; 8 DARP-specific removals (zero-split, history); 9 ILS/restart wrapper; 10 memoisation; 11 dropping the adaptive layer; 12 different metaheuristic per stage.

### Setup comparison (agent)
Li & Lim: our 25 s (100) / 50 s (200) per run vs Ropke-Pisinger 66 s / 264 s (2004 hardware, C++, 25k+ iterations), Curtois 300 s / 900 s, Sartori & Buriol 5 / 15 min. Li & Lim 100 is saturated (sanity check only). MD-H-DARP: our budget is the same order as ILS-SP and DA.

---

# Part 3. References

How each paper was checked (13 November 2026): **F** = full text read (at least the relevant part); **A** = abstract or publisher record read; **S** = seen only in search listings, details not confirmed. Numbers quoted in this report come from papers marked F or A. Before a number goes into the final paper, it should be checked once more against the PDF.

## Machine learning inside large neighbourhood search

1. A. Hottung, P. Wong-Chung, K. Tierney, "Neural Deconstruction Search for Vehicle Routing Problems," *Transactions on Machine Learning Research*, 2025. arXiv:2501.03715. **F**
2. A. Hottung, K. Tierney, "Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem," *ECAI 2020*, pp. 443–450. doi:10.3233/FAIA200124. **A**
3. S. Li, Z. Yan, C. Wu, "Learning to Delegate for Large-scale Vehicle Routing," *NeurIPS 2021*. arXiv:2107.04139. **A**
4. W. Feijen, G. Schäfer, K. Dekker, S. Pieterse, "Learning-Enhanced Neighborhood Selection for the Vehicle Routing Problem with Time Windows," arXiv:2403.08839, 2024 (preprint). **F**
5. W. Feijen, K. Dekker, S. van Zwam, "Learn to create neighborhoods in real-world vehicle routing problem," *LOD 2024*, LNCS. https://ir.cwi.nl/pub/35900. **A**
6. W. Ouyang, S. Li, Y. Ma, C. Wu, "Learning to Segment for Vehicle Routing Problems," arXiv:2507.01037, 2025. **A**
7. B. Herdianto, R. Billot, F. Lucas, M. Sevaux, D. Vigo, "Hybrid Node-Destroyer Model with Large Neighborhood Search for Solving the Capacitated Vehicle Routing Problem," arXiv:2508.08659, 2025 (preprint). **A**
8. B. Herdianto, R. Billot, F. Lucas, M. Sevaux, D. Vigo, "Edge-Selector Model Applied for Local Search Neighborhood for Solving Vehicle Routing Problems," arXiv:2508.14071, 2025 (preprint). **A**
9. J. Zhou, Y. Wu, Z. Cao, W. Song, J. Zhang, Z. Chen, "Learning Large Neighborhood Search for Vehicle Routing in Airport Ground Handling," *IEEE TKDE*, 2023. arXiv:2302.13797. **A**
10. L. Gao, M. Chen, Q. Chen, G. Luo, N. Zhu, Z. Liu, "Learn to Design the Heuristics for Vehicle Routing Problem," arXiv:2002.08539, 2020 (preprint). **A**
11. M. Chen, L. Gao, Q. Chen, Z. Liu, "Dynamic Partial Removal: A Neural Network Heuristic for Large Neighborhood Search," arXiv:2005.09330, 2020 (preprint). **A**
12. J. K. Falkner, L. Schmidt-Thieme, "Too Big, so Fail? – Enabling Neural Construction Methods to Solve Large-Scale Routing Problems," arXiv:2309.17089, 2023 (preprint). **A**
13. J. K. Falkner, D. Thyssens, L. Schmidt-Thieme, "Large Neighborhood Search based on Neural Construction Heuristics," arXiv:2205.00772, 2022. **A**
14. K. Li, F. Liu, Z. Wang, Q. Zhang, "Destroy and Repair Using Hyper-Graphs for Routing," *AAAI 2025*, 39(17):18341–18349. doi:10.1609/aaai.v39i17.34018. **A**
15. A. A. Syed, K. Akhnoukh, B. Kaltenhaeuser, K. Bogenberger, "Neural Network Based Large Neighborhood Search Algorithm for Ride Hailing Services," *EPIA 2019*, LNCS 11804, pp. 584–595. doi:10.1007/978-3-030-30241-2_49. **A**
16. R. Reijnen, Y. Zhang, H. C. Lau, Z. Bukhsh, "Online Control of Adaptive Large Neighborhood Search using Deep Reinforcement Learning," *ICAPS 2024*. doi:10.1609/icaps.v34i1.31507. **A**
17. J. Kallestad, R. Hasibi, A. Hemmati, K. Sörensen, "A general deep reinforcement learning hyperheuristic framework for solving combinatorial optimization problems," *EJOR* 309(1):446–468, 2023. doi:10.1016/j.ejor.2023.01.017. **A**
18. S.-N. Johnn, V.-A. Darvariu, J. Handl, J. Kalcsics, "Graph Reinforcement Learning for Operator Selection in the ALNS Metaheuristic," *OLA 2023*. arXiv:2302.14678. **A** (journal version: *Computers & Operations Research* 172, 2024, doi:10.1016/j.cor.2024.106791)
19. S. Yu, Z. Mao, C. Wu, J. Puchinger, "New Adaptive Mechanism for Large Neighborhood Search using Dual Actor-Critic," arXiv:2601.11414, 2026 (preprint). **A**
20. Y. Molinghen, A. Delecluse, R. De Landtsheer, S. Michelini, "Reinforcement Learning Methods for Neighborhood Selection in Local Search," *ICORES 2026*. arXiv:2601.07948. **A**
21. C. Bongiovanni, M. Kaspi, J.-F. Cordeau, N. Geroliminis, "A machine learning-driven two-phase metaheuristic for autonomous ridesharing operations," *Transportation Research Part E* 165:102835, 2022. doi:10.1016/j.tre.2022.102835. **A**
22. Y. Ma, J. Li, Z. Cao, W. Song, H. Guo, Y. Gong, Y. M. Chee, "Efficient Neural Neighborhood Search for Pickup and Delivery Problems," *IJCAI 2022*. doi:10.24963/ijcai.2022/662. **A**
23. J. Li, L. Xin, Z. Cao, A. Lim, W. Song, J. Zhang, "Heterogeneous Attentions for Solving Pickup and Delivery Problem via Deep Reinforcement Learning," *IEEE Trans. Intelligent Transportation Systems*, 2022. doi:10.1109/TITS.2021.3056120. **A**
24. J. Song, R. Lanka, Y. Yue, B. Dilkina, "A General Large Neighborhood Search Framework for Solving Integer Linear Programs," *NeurIPS 2020*. arXiv:2004.00422. **A**
25. Y. Wu, W. Song, Z. Cao, J. Zhang, "Learning Large Neighborhood Search Policy for Integer Programming," arXiv:2111.03466, 2021. **A**
26. N. Sonnerat, P. Wang, I. Ktena, S. Bartunov, V. Nair, "Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs," arXiv:2107.10201, 2021. **A**
27. T. Huang, A. Ferber, Y. Tian, B. Dilkina, B. Steiner, "Searching Large Neighborhoods for Integer Linear Programs with Contrastive Learning," *ICML 2023*, PMLR 202:13869–13890. arXiv:2302.01578. **A**

## Critical evidence, pruning and training signal

28. Í. Santana, A. Lodi, T. Vidal, "Neural Networks for Local Search and Crossover in Vehicle Routing: A Possible Overkill?," *CPAIOR 2023*, LNCS, pp. 184–199. doi:10.1007/978-3-031-33271-5_13. arXiv:2210.12075. **F**
29. Y. Xia, X. Yang, Z. Liu, Z. Liu, L. Song, J. Bian, "Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling Salesman Problems," *ICML 2024*, PMLR 235. arXiv:2406.03503. **F**
30. X. Pan, C. Wang, C. Ying, Y. Xue, T. Yu, "Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers," arXiv:2411.09238, 2024. **A**
31. L. Accorsi, A. Lodi, D. Vigo, "Guidelines for the computational testing of machine learning approaches to vehicle routing problems," *Operations Research Letters* 50(2):229–234, 2022. doi:10.1016/j.orl.2022.01.018. **F**
32. S. Liu, Y. Zhang, K. Tang, X. Yao, "How Good Is Neural Combinatorial Optimization? A Systematic Evaluation on the Traveling Salesman Problem," *IEEE Computational Intelligence Magazine* 18(3):14–28, 2023. doi:10.1109/MCI.2023.3277768. **A**
33. C. K. Joshi, Q. Cappart, L.-M. Rousseau, T. Laurent, "Learning the Travelling Salesperson Problem Requires Rethinking Generalization," *Constraints* 27:70–98, 2022. doi:10.1007/s10601-022-09327-y. **A**
34. R. Turkeš, K. Sörensen, L. M. Hvattum, "Meta-analysis of metaheuristics: Quantifying the effect of adaptiveness in adaptive large neighborhood search," *EJOR* 292(2):423–442, 2021. doi:10.1016/j.ejor.2020.10.045. **A**
35. A. Santini, S. Ropke, L. M. Hvattum, "A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic," *Journal of Heuristics* 24(5):783–815, 2018. doi:10.1007/s10732-018-9377-x. **A**
36. P. Toth, D. Vigo, "The Granular Tabu Search and Its Application to the Vehicle-Routing Problem," *INFORMS Journal on Computing* 15(4):333–346, 2003. doi:10.1287/ijoc.15.4.333.24890. **A**
37. L. Accorsi, D. Vigo, "A Fast and Scalable Heuristic for the Solution of Large-Scale Capacitated Vehicle Routing Problems," *Transportation Science* 55(4):832–856, 2021. doi:10.1287/trsc.2021.1059. **A**
38. F. Arnold, K. Sörensen, "Knowledge-guided local search for the vehicle routing problem," *Computers & Operations Research* 105:32–46, 2019. doi:10.1016/j.cor.2019.01.002. **A**
39. F. Arnold, Í. Santana, K. Sörensen, T. Vidal, "PILS: Exploring high-order neighborhoods by pattern mining and injection," *Pattern Recognition* 116:107957, 2021. doi:10.1016/j.patcog.2021.107957. **A**
40. M. R. H. Maia, A. Plastino, U. S. Souza, "An improved hybrid genetic search with data mining for the CVRP," *Networks* 83(4):692–711, 2024. doi:10.1002/net.22213. **A**
41. Y. Sun, A. Ernst, X. Li, J. Weiner, "Generalization of machine learning for problem reduction: a case study on travelling salesman problems," *OR Spectrum* 43, 2021. doi:10.1007/s00291-020-00604-x. **A**
42. J. Fitzpatrick, D. Ajwani, P. Carroll, "Learning to Sparsify Travelling Salesman Problem Instances," *CPAIOR 2021*, LNCS, pp. 410–426. doi:10.1007/978-3-030-78230-6_26. **A**
43. L. Xin, W. Song, Z. Cao, J. Zhang, "NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem," *NeurIPS 2021*. arXiv:2110.07983. **A**
44. W. Kool, H. van Hoof, J. Gromicho, M. Welling, "Deep Policy Dynamic Programming for Vehicle Routing Problems," *CPAIOR 2022*, LNCS, pp. 190–213. doi:10.1007/978-3-031-08011-1_14. **A**
45. B. Hudson, Q. Li, M. Malencia, A. Prorok, "Graph Neural Network Guided Local Search for the Traveling Salesperson Problem," *ICLR 2022*. arXiv:2110.05291. **F**
46. M. Morabit, G. Desaulniers, A. Lodi, "Machine-Learning–Based Column Selection for Column Generation," *Transportation Science* 55(4):815–831, 2021. doi:10.1287/trsc.2021.1045. **A**
47. M. Morabit, G. Desaulniers, A. Lodi, "Machine-Learning–Based Arc Selection for Constrained Shortest Path Problems in Column Generation," *INFORMS Journal on Optimization* 5(2):191–210, 2023. doi:10.1287/ijoo.2022.0082. **A**
48. S. Ross, G. Gordon, D. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," *AISTATS 2011*. arXiv:1011.0686. **A**
49. E. Khalil, P. Le Bodic, L. Song, G. Nemhauser, B. Dilkina, "Learning to Branch in Mixed Integer Programming," *AAAI 2016*. doi:10.1609/aaai.v30i1.10080. **A**
50. Q. Han, L. Yang, Q. Chen, X. Zhou, D. Zhang, A. Wang, R. Sun, X. Luo, "A GNN-Guided Predict-and-Search Framework for Mixed-Integer Linear Programming," *ICLR 2023*. arXiv:2302.05636. **A**
51. H. Liu, J. Wang, Z. Geng, X. Li, Y. Zong, F. Zhu, J. Hao, F. Wu, "Apollo-MILP: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming," *ICLR 2025*. arXiv:2503.01129. **A**
52. T. Yoon, J. Choi, H. Yun, S. Lim, "Threshold-aware Learning to Generate Feasible Solutions for Mixed Integer Programs," arXiv:2308.00327, 2023. **A**
53. Y. Bengio, A. Lodi, A. Prouvost, "Machine learning for combinatorial optimization: A methodological tour d'horizon," *EJOR* 290(2):405–421, 2021. doi:10.1016/j.ejor.2020.07.063. **A**
54. M. Karimi-Mamaghan, M. Mohammadi, P. Meyer, A. M. Karimi-Mamaghan, E.-G. Talbi, "Machine learning at the service of meta-heuristics for solving combinatorial optimization problems: A state-of-the-art," *EJOR* 296(2):393–422, 2022. doi:10.1016/j.ejor.2021.04.032. **A**
55. Y. Rochat, É. Taillard, "Probabilistic diversification and intensification in local search for vehicle routing," *Journal of Heuristics* 1(1):147–167, 1995. doi:10.1007/BF02430370. **S**

## Classical methods and benchmarks for PDPTW and DARP

56. S. Ropke, D. Pisinger, "An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows," *Transportation Science* 40(4):455–472, 2006. doi:10.1287/trsc.1050.0135. **F** (thesis-chapter version of the same paper)
57. D. Pisinger, S. Ropke, "A general heuristic for vehicle routing problems," *Computers & Operations Research* 34(8):2403–2435, 2007. doi:10.1016/j.cor.2005.09.012. **S**
58. H. Li, A. Lim, "A metaheuristic for the pickup and delivery problem with time windows," *Int. J. Artificial Intelligence Tools* 12:173–186, 2003. **S**
59. SINTEF, "Li & Lim benchmark – best known solutions," https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/ (read November 2026). **F**
60. T. Curtois, D. Landa-Silva, Y. Qu, W. Laesanklang, "Large neighbourhood search with adaptive guided ejection search for the pickup and delivery problem with time windows," *EURO J. Transportation and Logistics* 7(2):151–192, 2018. doi:10.1007/s13676-017-0115-6. **F**
61. C. S. Sartori, L. S. Buriol, "A study on the pickup and delivery problem with time windows: Matheuristics and new instances," *Computers & Operations Research* 124:105065, 2020. doi:10.1016/j.cor.2020.105065. **A**
62. J. Christiaens, G. Vanden Berghe, "Slack Induction by String Removals for Vehicle Routing Problems," *Transportation Science* 54(2):417–433, 2020. doi:10.1287/trsc.2019.0914. **A**
63. Y. Nagata, S. Kobayashi, "Guided ejection search for the pickup and delivery problem with time windows," *EvoCOP 2010*, LNCS 6022, pp. 202–213. doi:10.1007/978-3-642-12139-5_18. **S**
64. G. Hiermann, M. Schiffer, "A decomposition-based approach for large-scale pickup and delivery problems," arXiv:2405.00230, 2024. **F**
65. J.-F. Cordeau, G. Laporte, "A tabu search heuristic for the static multi-vehicle dial-a-ride problem," *Transportation Research Part B* 37(6):579–594, 2003. doi:10.1016/S0191-2615(02)00045-0. **A**
66. S. N. Parragh, V. Schmid, "Hybrid column generation and large neighborhood search for the dial-a-ride problem," *Computers & Operations Research* 40(1):490–497, 2013. doi:10.1016/j.cor.2012.08.004. **F**
67. T. Gschwind, M. Drexl, "Adaptive large neighborhood search with a constant-time feasibility test for the dial-a-ride problem," *Transportation Science* 53(2):480–491, 2019. doi:10.1287/trsc.2018.0837. **F** (discussion-paper version)
68. K. Braekers, A. Caris, G. K. Janssens, "Exact and meta-heuristic approach for a general heterogeneous dial-a-ride problem with multiple depots," *Transportation Research Part B* 67:166–186, 2014. doi:10.1016/j.trb.2014.05.007. **A**
69. S. N. Parragh, "Introducing heterogeneous users and vehicles into models and algorithms for the dial-a-ride problem," *Transportation Research Part C* 19(5):912–930, 2011. doi:10.1016/j.trc.2010.06.002. **A**
70. I. Malheiros, R. Ramalho, B. Passeti, T. Bulhões, A. Subramanian, "A hybrid algorithm for the multi-depot heterogeneous dial-a-ride problem," *Computers & Operations Research* 129:105196, 2021. doi:10.1016/j.cor.2020.105196. **F** (our own copy; Table 6)
71. A. Subramanian, E. Uchoa, L. S. Ochi, "A hybrid algorithm for a class of vehicle routing problems," *Computers & Operations Research* 40(10):2519–2531, 2013. doi:10.1016/j.cor.2013.01.013. **S**
72. C. Archetti, M. G. Speranza, "A survey on matheuristics for routing problems," *EURO J. Computational Optimization* 2(4):223–246, 2014. doi:10.1007/s13675-014-0030-7. **A**

Notes on the search:

- We found **no** published paper in which a learned repair or learned route choice beats a tuned ALNS on a multi-vehicle PDPTW or DARP.
- We found **no** paper on confidence-based ("calibrated") pruning for routing heuristics. The idea in Section 6 is taken from mixed-integer programming papers (50–52) and is marked as a hypothesis.
- We found no paper from 2022–2026 with new best-known solutions for Li & Lim 100/200 or for our H-DARP instances.
