# Benchmarking Parameter-Space vs. Activation-Space Decompositions for Mechanistic Interpretability

> *We benchmark parameter-space (SPD) vs. activation-space decompositions on interpretability, faithfulness, stability, and downstream tasks (steering/probing/ablation), and map how their learned components align across methods.*

## Abstract

Mechanistic interpretability increasingly relies on decomposition methods that break a model into more legible parts. Current work spans at least two largely separate families: activation-space decompositions (e.g., SAEs, Transcoders, CLTs, MOLTs) and parameter-space decompositions (e.g., Stochastic Parameter Decomposition, SPD). Ideally for interpretability, these methods should find components that are faithful (actually correspond to the underlying computation), useful for probing/steering/ablation, and canonical (e.g., not artifacts of hyperparameters). This project will (1) quantitatively benchmark the components found by these different approaches, (2) directly test whether parameter-space components are more canonical units of analysis, and (3) transfer the best training practices from these different methods. The goal is a clearer answer to: What are the differences and similarities between components found by different decomposition methods, and which components are better units of analysis for interpretability work? The findings will be integrated into a forthcoming paper on applying SPD to language models.

## 1. Background

**Problem and safety relevance.** Developing powerful AI systems poses a significant risk if we have no clear understanding of their internal workings. While activation-based dictionary learning variants such as SAEs are the leading method for identifying a model's features, they often exhibit pathological behaviours like feature splitting (where a single concept is represented by multiple features), or feature absorption and composition, which obscure the atomic nature of the representations.

If we had better methods for decomposing model computation, this knowledge could be leveraged for high-impact downstream applications, such as more effective model steering, precise knowledge editing, white-box monitoring, and robust auditing for harmful capabilities. Finally, if we had better units of analysis, we could be more successful in uncovering "unknown unknowns" within AI systems, thereby reducing the risk of unforeseen model failures.

**Related work and gap.** Our proposal builds on earlier work showing issues with activation-based decompositions leading to non-canonical features [1, 2], caused by problems such as feature splitting [3], absorption [4], and composition [5, 6]. Parameter decomposition [7, 8] methods claim they may suffer less from these pathologies, but this has not been empirically shown.

In our project, we aim to test this hypothesis by developing new evaluations, and also evaluate components with existing benchmarks such as SAEBench [9] and the best available automatic interpretability methods [10].

**Path to impact.** Results will inform interpretability researchers which decomposition methods they should prioritize. If SPD finds more canonical units where SAEs do not, this motivates increased investment in parameter-space decomposition methods. The findings of this project will be integrated in an upcoming paper published by mentor's team on using SPD to decompose LLMs.

## 2. Work Done So Far

I have built familiarity with the SPD codebase by decomposing MNIST MLPs, recovering clean structure: one mechanism per digit class in the output layer and interpretable visual features in the input layer. I implemented SAE variants (TopK, JumpReLU, Stochastic), transcoders, and MOLTs as baselines for systematic comparison against SPD. I also explored training decomposed models from scratch rather than post-hoc, finding that the two approaches yield different decompositions.

I have also explored possible connections between SPD and Shard Theory, including an initial attempt to investigate "shards" in convolutional neural networks, but dropped this direction because it seemed that more fundamental questions about decomposition methods remain open. These experiments have sharpened my understanding of the tradeoffs between parameter-space and activation-space methods, which directly informs the evaluation framework proposed above.

## 3. Planned Work

**Weeks 5–6: Match training setups.** Select a single target model and specific MLP layers to serve as the shared substrate kept constant across methods. Train SPD decompositions for the target matrices, and Transcoder, SAE, and MOLT baselines for the same MLP.

*Deliverable:* Open-source repository to train these different methods on the same data/model, enabling apples-to-apples measurement.

**Weeks 7–8: Quantitative evaluation on shared metrics.** Design evaluations and run comparisons to evaluate the components found by the different methods on automated interpretability, faithfulness/loss recovered, steering, sparse probing, feature splitting/absorption, stability across seeds, atomicity, and completeness.

*Deliverable:* Results table and narrative on where SPD is stronger/weaker than activation-based approaches.

**Weeks 9–10: Test the canonical units hypothesis.** Directly test whether SPD components are more canonical units of analysis (in the sense of Leask et al. [1]) than activation-based features.

- Train decompositions at multiple dictionary sizes (varying *C* for SPD, width for SAEs/Transcoders) and measure whether SPD components remain stable while activation-based features split.
- Correlate SPD importance patterns with SAE/Transcoder activations to identify where the two families agree and diverge. Investigate how SPD components interact with SAE components (e.g., whether SAE features in a later layer can be explained by SPD transforms of SAE features in an earlier layer).
- Characterize the many-to-many mapping between methods: quantify how many activation-based features map to a single SPD component and vice versa, and whether SPD components that subsume multiple SAE features correspond to less-split, more coherent concepts.

*Deliverable:* Quantitative evidence for or against parameter-space components as more canonical units of analysis and description of how the different decompositions relate to each other.

**Weeks 11–12: Cross-pollination of decompositions.** Based on the evaluations and insights from weeks 7–10, identify ways to transfer the best training practices between methods. Concrete candidates include: porting SPD's stochastic ablation loss to transcoders to test whether it reduces feature splitting, and constraining MOLTs to sum to the original weights to improve faithfulness. I will also synthesize the benchmark results into comparison figures and draft the relevant sections for the forthcoming SPD paper.

*Deliverable:* If successful, this yields an improved baseline that advances the Pareto frontier of decomposition quality; if not, the negative results still clarify which properties of SPD's training are specific to parameter-space decomposition versus transferable to activation-based methods.

**Extension: Qualitative comparisons of circuits.** For well-studied circuits (e.g., induction heads, subject-verb agreement, IOI), attempt end-to-end circuit analysis using both activation-space and parameter-space decompositions. Compare the kinds of interpretations each method yields, where they fail, and whether they provide complementary views. This is the most ambitious part of the project and benefits from the familiarity with the decompositions built during the main program.

*Deliverable:* Circuit comparison write-up, either as a section of the SPD paper or as a standalone post. Expected duration: 4–6 weeks.

## 4. Risks and Contingency Plans

The primary risk is that SPD has so far only been applied to small models, and it is unclear whether insights from comparing decomposition methods at this scale will transfer to real language models. SPD may also remain too computationally expensive for labs to run on large models, limiting the practical relevance of the findings.

A second risk is that parameter-space and activation-space components turn out to be incommensurable — not because one is better, but because they decompose along fundamentally different axes. In that case, the project pivots from "which is better" to characterizing how the decompositions differ and whether they provide complementary views of the same computation.

A third risk is that the evaluation metrics fail to discriminate between methods. Here, the contingency is to focus on qualitative circuit-level comparisons where differences may be more apparent.

Regarding dual use, if SPD yields substantially better interpretations of model internals, these insights could in principle be used to accelerate capability development, reducing the window available for other alignment work. However, we believe the safety benefits of better interpretability outweigh this risk.

## References

1. Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. "Sparse Autoencoders Do Not Find Canonical Units of Analysis." *arXiv preprint* arXiv:2502.04878, 2025.
2. Gonçalo Paulo and Nora Belrose. "Sparse Autoencoders Trained on the Same Data Learn Different Features." *arXiv preprint* arXiv:2501.16615, 2025.
3. T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, et al. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." *Transformer Circuits Thread*, 2, 2023.
4. David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Bloom. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." *arXiv preprint* arXiv:2409.14507, 2024.
5. E. Anders, C. Neo, J. Hoelscher-Obermaier, and J. N. Howard. "Sparse Autoencoders Find Composed Features in Small Toy Models." LessWrong, 2024.
6. Martin Wattenberg and Fernanda B. Viégas. "Relational Composition in Neural Networks: A Survey and Call to Action." *arXiv preprint* arXiv:2407.14662, 2024.
7. Dan Braun, Lucius Bushnaq, Stefan Heimersheim, Jake Mendel, and Lee Sharkey. "Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition." *arXiv preprint* arXiv:2501.14926, 2025.
8. Lucius Bushnaq, Dan Braun, and Lee Sharkey. "Stochastic Parameter Decomposition." *arXiv preprint*, 2025. https://arxiv.org/abs/2506.20790
9. Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, et al. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability." *arXiv preprint* arXiv:2503.09532, 2025.
10. Gonçalo Paulo, Alex Mallen, Caden Juang, and Nora Belrose. "Automatically Interpreting Millions of Features in Large Language Models." *arXiv preprint* arXiv:2410.13928, 2024.