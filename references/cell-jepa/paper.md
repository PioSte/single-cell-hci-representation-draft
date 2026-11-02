<!-- page 1 of 26 -->

arXiv:2602.02093v1 [cs.CE] 2 Feb 2026

# Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics

**Ali ElSheikh**<sup>∗,1</sup>, **Rui-Xi Wang**<sup>∗,3,6</sup>, **Weimin Wu**<sup>∗,1</sup>, **Yibo Wen**<sup>1</sup>, **Payam Dibaeinia**<sup>2</sup>, **Jennifer Yuntong Zhang**<sup>5</sup>, **Jerry Yao-Chieh Hu**<sup>1</sup>, **Mei Knudson**<sup>2,4</sup>, **Sudarshan Babu**<sup>2</sup>, **Shao-Hua Sun**<sup>6</sup>, **Aly A. Khan**<sup>2,4</sup>, **Han Liu**<sup>†,1</sup>

<sup>1</sup>Northwestern University, <sup>2</sup>Biohub, <sup>3</sup>Massachusetts Institute of Technology, <sup>4</sup>University of Chicago, <sup>5</sup>University of Toronto, <sup>6</sup>National Taiwan University

**Abstract:** Single-cell foundation models learn by reconstructing masked gene expression, implicitly treating technical noise as signal. With dropout rates exceeding 90%, reconstruction objectives encourage models to encode measurement artifacts rather than stable cellular programs. We introduce Cell-JEPA, a joint-embedding predictive architecture that shifts learning from reconstructing sparse counts to predicting in latent space. The key insight is that cell identity is redundantly encoded across genes. We show predicting cell-level embeddings from partial observations forces the model to learn dropout-robust features. On cell-type clustering, Cell-JEPA achieves 0.72 AvgBIO in zero-shot transfer versus 0.53 for scGPT, a 36% relative improvement. On perturbation prediction within a single cell line, Cell-JEPA improves absolute-state reconstruction but not effect-size estimation, suggesting that representation learning and perturbation modeling address complementary aspects of cellular prediction.

**Keywords:** Single Cell; Foundation Model

Single-Cell Transcriptomics

![Image block](images/page001-block08.png)

![Image block](images/page001-block09.png)

Downstream Tasks

![Image block](images/page001-block11.png)

Perturbation Response Prediction

![Image block](images/page001-block13.png)

Figure 1: Overview of the Cell-JEPA Pipeline. Raw biological cells undergo single-cell RNA sequencing to generate high-dimensional, sparse gene expression profiles. The profiles are passed through a joint embedding architecture, where a student encoder receives masked inputs and predicts the stable latent representations produced by a teacher encoder from unmasked data. This predictive alignment learns a robust biological manifold that enables downstream tasks, including cell-type clustering and perturbation response prediction.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>∗</sup>Equal contribution.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>†</sup>Correspondence: Han Liu (hanliu@northwestern.edu).</span></small>

<!-- page 2 of 26 -->

## 1 Introduction

We introduce Cell-JEPA, a self-supervised foundation model for single-cell transcriptomics. Single-cell transcriptomics enables the study of cellular heterogeneity, development, and disease [Ding et al., 2022, Svensson et al., 2018, Tang et al., 2011]. Each cell is represented by an expression profile across tens of thousands of genes, providing a high-dimensional snapshot of cellular state. However, single-cell measurements are sparse and noisy due to limited molecular capture, stochastic sampling, and experimental variability [Luecken and Theis, 2019]. Therefore, the core challenge is to infer stable and biologically meaningful cellular programs from high-dimensional and noisy observations.

Recent progress has introduced large-scale self-supervised foundation models for single-cell data. A common strategy is to learn representations by predicting masked or noisy parts of the transcriptome. For example, scGPT [Cui et al., 2024] performs generative pre-training by predicting expression values for a subset of “unknown” genes given observed “known” genes, using a specialized attention mask and an iterative generation procedure that progressively incorporates high-confidence predictions. Geneformer [Theodoris et al., 2023] adopts a BERT-style masked learning objective on ranked gene tokens, training to recover masked genes from the remaining context within each cell. UCE [Rosen et al., 2023] relies on self-supervision by masking expressed genes and training to predict gene-level expression presence, while aiming to provide a transferable embedding space across diverse tissues and species. In parallel, scVI [Lopez et al., 2018] learns low-dimensional cell states through a probabilistic generative model of counts trained by variational inference. While these approaches differ in modeling assumptions, they share the key property that pre-training is strongly tied to reconstructing information closely aligned with the observed measurement space (e.g., masked counts, masked gene identities, or gene-level presence/absence).

A limitation of this paradigm is that single-cell expression values can be heavily influenced by sequencing depth, capture efficiency, and technical noise. As a result, objectives that emphasize accurate gene-level reconstruction may encourage models to represent measurement-specific variability, instead of prioritizing stable structure that defines cellular identity and regulatory programs. This motivates learning objectives that constrain the representation space, rather than optimizing for fidelity to noisy observed entries.

Recent progress in computer vision suggests that Joint Embedding Predictive Architectures (JEPA) provide a strong self-supervised signal by predicting masked content in a learned latent space, rather than reconstructing raw pixel inputs [Assran et al., 2025, 2023, Lecun, 2022]. By shifting prediction to the embedding space, JEPA encourages consistency at the level of abstract features and reduces reliance on exact input values, which is desirable when observations are noisy and incomplete. This property closely matches the goal of single-cell modeling: recovering robust cell states from sparse and noisy gene expression measurements.

Motivated by these developments, we introduce **Cell-JEPA**, a JEPA-style framework for single-cell rep resentation learning built on top of scGPT. Cell-JEPA augments reconstruction-style pre-training with a latent-space prediction objective that enforces consistency between predicted and target cell representations under partial observation. We hypothesize that combining scGPT’s scalable transcriptome modeling with JEPA-style latent prediction yields embeddings that better capture stable cellular programs and higher-order biological structure. Across downstream tasks, we show that Cell-JEPA learns more robust representations, improving performance on cell-state prediction problems compared to reconstruction-only baselines.

In summary, we have the following three key contributions.

• We propose **Cell-JEPA**, a predictive self-supervised framework for single-cell transcriptomics that

2

<!-- page 3 of 26 -->

![Image block](images/page003-block01.png)

Figure 2: Cell-JEPA Model Architecture and Training Pipeline. Cell-JEPA builds on scGPT with a student–teacher transformer architecture. Gene identities and discretized expression values are embedded and summed to form input token embeddings. The student encoder processes masked inputs, while the teacher encoder processes the corresponding unmasked inputs and is updated via an exponential moving average (EMA). Training optimizes a gene-level reconstruction loss in expression space and a JEPA loss that predicts the teacher’s cell embedding from the masked student view.

complements reconstruction-based pre-training with latent-space prediction. It enables stable and meaningful cellular representation learning from high-dimensional, sparse, and noisy gene expression.

• We curate a large-scale human kidney scRNA-seq corpus comprising 800,000 cells spanning multiple studies, donors, and experimental conditions. Based on this, we pre-train and release our single-cell foundation model under this new pre-training paradigm.

• We evaluate Cell-JEPA in three downstream settings: (i) supervised finetuning for cell-type clustering, (ii) zero-shot transfer for cell-type clustering, and (iii) perturbation-response prediction. Across all experiments, Cell-JEPA outperforms the baseline scGPT model. In particular, Cell-JEPA achieves a relative improvement of 35% on the primary evaluation metric in the zero-shot setting.

**Organization.** We describe the pre-training and finetuning methods in Section 2, present experimental results in Section 3, and review related work in Section B.

## 2 Method

In this section, we first describe the data preprocessing used for Cell-JEPA pre-training in Section 2.1. We then present the tokenization procedure and model architecture in Section 2.2. Next, we detail the pre-training objective in Section 2.3. Finally, we describe the downstream finetuning losses in Section 2.4.

**Dataset Formatting.** Each scRNA-seq dataset is represented as a sparse cell-by-gene count matrix. Following prior work [Cui et al., 2024], we exclude genes with zero counts within each cell and store the

3

<!-- page 4 of 26 -->

remaining non-zero entries as aligned lists of gene identifiers and expression values. We encode each cell as a sparse sequence $\{ ( y _ { k } , v _ { k } ) \} _ { k = 1 } ^ { L }$ , where $y _ { k }$ denotes a gene identity and $v _ { k } \in \mathbb { R } _ { + }$ is its corresponding expression value.

## 2.1 Pre-training Dataset Processing

We follow the preprocessing strategy of scGPT [Cui et al., 2024] to (i) discretize expression values via per-cell quantile binning and (ii) subsample genes to control sequence length across cells.

**Quantile-based Binning.** Following scGPT [Cui et al., 2024], we discretize non-zero gene expression values using per-cell quantile binning to improve robustness to scale differences across cells. For each cell, we compute quantile bin edges over its non-zero expression values and map each value to a bin index in $\{ 1 , \ldots , B \}$ , with $B = 5 0$ by default. These bin indices serve as the value inputs $v _ { k }$ , preserving the relative ordering of expression levels within a cell while reducing dependence on raw count magnitudes.

**Gene Subsampling.** Cells express variable numbers of genes, resulting in sequences with different lengths. Following scGPT [Cui et al., 2024], we cap the number $\mathbf { o f }$ expressed genes per cell at $L _ { \mathrm { m a x } }$ (set to $L _ { \mathrm { m a x } } { = } 6 0 0 )$ . For cells exceeding this limit, we avoid selecting a fixed subset of genes (e.g., the first $L _ { \mathrm { m a x } } )$ , which could introduce bias. Instead, we uniformly subsample $L _ { \mathrm { m a x } }$ gene tokens at random from the expressed gene set, ensuring that each expressed gene has equal probability of being included. This stochastic subsampling also serves as a form of biological dropout: across training epochs, the model observes different partial views of the same cell and must learn from incomplete information. As a result, the learned representations become more robust to sparsity and expression variability.

## 2.2 Tokenization and Model Architecture

**Overview.** Cell-JEPA builds directly on scGPT [Cui et al., 2024]. scGPT is a bidirectional Transformer encoder trained with a gene expression prediction objective under masked inputs (see Section C). We adopt the same tokenization, embedding design, and encoder backbone as scGPT, and introduce a Joint Embedding Predictive Architecture (JEPA) by instantiating two copies of the scGPT encoder: a student network optimized by gradient descent and a slowly-evolving teacher network updated via exponential moving average (EMA). This student–teacher formulation enables us to add a representation-level JEPA objective alongside the original scGPT reconstruction-style loss. We detail the tokenization procedure, the teacher–student model design, and the encoder architecture as follows.

## 2.2.1 Tokenization and Input Embeddings

**Vocabulary and Special Tokens.** Following scGPT [Cui et al., 2024], we construct a unified gene vocabulary and augment it with special tokens including $<   \mathtt { c l } s   >$ and &lt;pad&gt;. For each cell, we prepend a &lt;cls&gt; token, and pad sequences within a minibatch to length $L _ { \mathrm { m a x } } + 1$ using &lt;pad&gt;.

**Gene and Value Embeddings.** Given a gene identity $y _ { i }$ and its (preprocessed) expression value $v _ { i } ,$ , we embed the gene token and value using the same design as scGPT:

$$
\mathbf {y} _ {i} = f _ {\mathrm{gene}} (y _ {i}), \quad \mathbf {v} _ {i} = f _ {\mathrm{val}} (v _ {i}),
$$

4

<!-- page 5 of 26 -->

where $f _ { \mathrm { g e n e } }$ is a learnable embedding lookup table and $f _ { \mathrm { v a l } }$ is a small MLP mapping scalar inputs to the model hidden dimension. The input representation for gene i is the sum:

$$
\mathbf {z} _ {i} = \mathbf {y} _ {i} + \mathbf {v} _ {i}.\tag{2.1}
$$

The &lt;cls&gt; token embedding is obtained from the same gene embedding table:

$$
\mathbf {z} _ {\mathrm{cls}} = f _ {\text {gene}} (\langle \mathrm{cls} \rangle).
$$

**Masking.** During pre-training, we apply masking to expression values only, following scGPT [Cui et al., 2024]. For a masked gene i, we replace $v _ { i }$ with a sentinel value $v _ { \mathrm { m a s k } }$ (set to −1), producing a masked value embedding:

$$
\mathbf {v} _ {\mathrm{mask}} = f _ {\mathrm{val}} (v _ {\mathrm{mask}}),
$$

and a masked token representation:

$$
\tilde {\mathbf {z}} _ {i} = \mathbf {y} _ {i} + \mathbf {v} _ {\text {mask}}.
$$

Let $\mathbf { Z } = ( \mathbf { z } _ { \mathrm { c l s } } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } )$ denote the unmasked token sequence and $\tilde { \mathbf { Z } } = ( \mathbf { z } _ { \operatorname { c l s } } , \tilde { \mathbf { z } } _ { 1 } , \ldots , \tilde { \mathbf { z } } _ { n } )$ the corresponding masked sequence.

## 2.2.2 Student-teacher Encoders

**Encoders.** Both the student encoder $g _ { \mathrm { S } }$ and teacher encoder $g _ { \mathrm { T } }$ share the same architecture as scGPT [Cui et al., 2024] (i.e., a bidirectional Transformer encoder stack). The student processes Z, while the teacher processes $\tilde { \mathbf { Z } }$ to produce stable target representations:

$$
\hat {\mathbf {H}} = g _ {\mathrm{S}} (\tilde {\mathbf {Z}}), \quad \mathbf {H} = g _ {\mathrm{T}} (\mathbf {Z}),
$$

We use the &lt;cls&gt; outputs as cell-level embeddings:

$$
\hat {\mathbf {e}} = \hat {\mathbf {H}} _ {0}, \quad \mathbf {e} = \mathbf {H} _ {0}.\tag{2.2}
$$

**EMA Teacher Update.** The teacher parameters are updated as an exponential moving average of the student parameters (as commonly used in JEPA-style methods):

$$
\theta_ {\mathrm{T}} \leftarrow m \theta_ {\mathrm{T}} + (1 - m) \theta_ {\mathrm{S}},
$$

where $m \in [ 0 , 1 )$ is the momentum coefficient, and $\theta _ { \mathrm { S } } , \theta _ { \mathrm { T } }$ are the parameters of the student and teacher encoders, respectively. This yields a slowly-evolving target encoder that stabilizes the representation prediction objective.

## 2.2.3 Transformer Block and Attention Masking

Both encoders use the same transformer block structure as scGPT [Cui et al., 2024]. Each block consists of self-attention layers, multilayer perceptrons, nonlinear transformations, and layer normalization. We describe the self-attention layer in detail below for clarity.

5

<!-- page 6 of 26 -->

**Self-attention.** For a hidden sequence H $\in \mathbb { R } ^ { ( L + 1 ) \times D }$ , where $D$ is the embedding dimension. We use $\mathbf { A } _ { \mathrm { m a s k } }$ to denote a structured attention mask to prevent information leakage between masked targets. Let $\mathcal { U } _ { \mathrm { m a s k } }$ denote the indices of masked genes at the current step, and then the corresponding entries in $\mathbf { A } _ { \mathrm { m a s k } }$ are:

$$
a _ {i, j} = \left\{ \begin{array}{l l} 0, & \text {if} j \notin \mathcal {U} _ {\text {mask}}, \\ 0, & \text {if} i = j \text {and} j \in \mathcal {U} _ {\text {mask}}, \\ - \infty , & \text {otherwise}. \end{array} \right.
$$

This design ensures that each masked token can attend to all unmasked tokens (and itself), but not to other masked tokens. We then compute the attention module as:

$$
\operatorname{Attn} (\mathbf {H}) = \operatorname{Softmax} \left(\frac {\mathbf {Q K} ^ {\top}}{\sqrt {d}} + \mathbf {A} _ {\text {mask}}\right) \mathbf {V},
$$

where $\mathbf { Q } = \mathbf { H } \mathbf { W } _ { Q } , \mathbf { K } = \mathbf { H } \mathbf { W } _ { K } , \mathbf { V } = \mathbf { H } \mathbf { W } _ { V }$ and $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { D \times d }$ are the projection matrices, and d denotes the attention head dimension.

## 2.3 Pre-training Loss Calculation

Cell-JEPA augments the original scGPT pre-training objective [Cui et al., 2024] with a JEPA-style representation prediction loss. Concretely, we optimize the student encoder using (i) a gene-level reconstruction loss identical in spirit to scGPT and (ii) a cell-level JEPA objective that predicts the teacher’s latent embedding from the masked student view.

**JEPA Objective.** Let ê and e denote the student and teacher &lt;cls&gt; embeddings from (2.2). We introduce a predictor head $p ( \cdot )$ and apply it to the student embedding:

$$
\tilde {\mathbf {e}} = p (\hat {\mathbf {e}}).
$$

We then minimize a cosine-distance loss between the predicted student embedding and the teacher target embedding:

$$
\mathcal {L} _ {\mathrm{JEPA}} = 1 - \frac {\tilde {\mathbf {e}} ^ {\top} \operatorname{sg} (\mathbf {e})}{\| \tilde {\mathbf {e}} \| _ {2} \| \operatorname{sg} (\mathbf {e}) \| _ {2}},\tag{2.3}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient (the teacher embedding is treated as a fixed target). Unlike reconstruction losses defined in expression space, $\mathcal { L } _ { \mathrm { J E P A } }$ operates purely in representation space and encourages the student to match the teacher’s cell-level semantics under masking.

**Gene-level Reconstruction Loss.** In addition to $\mathcal { L } _ { \mathrm { J E P A } }$ , we retain the scGPT-style gene expression pre-diction objective [Cui et al., 2024]. Unlike $\mathbf { s c G P T s }$ iterative generation procedure, we predict all masked genes in a single forward pass, following a BERT-style masked prediction scheme.

Given the student token outputs $\hat { \mathbf { H } } = g _ { \mathrm { S } } ( \tilde { \mathbf { Z } } )$ , we predict expression values for each token using a value head $r ( \cdot )$ (MLP applied row-wise):

$$
\hat {\mathbf {v}} = r (\hat {\mathbf {H}}),
$$

6

<!-- page 7 of 26 -->

where $\hat { \mathbf { v } } _ { i }$ denotes the reconstructed value for gene token i. We compute mean squared error on the masked gene set $\mathcal { U } _ { \mathrm { m a s k } }$

$$
\mathcal {L} _ {\mathrm{rec}} = \frac {1}{| \mathcal {U} _ {\mathrm{mask}} |} \sum_ {i \in \mathcal {U} _ {\mathrm{mask}}} (\hat {\mathbf {v}} _ {i} - v _ {i}) ^ {2}.\tag{2.4}
$$

This term preserves scGPT’s training signal by explicitly encouraging accurate prediction of masked expression values from the observed context.

**Combined Pre-training Objective.** The final pre-training objective is a weighted combination of reconstruction and JEPA losses:

$$
\mathcal {L} _ {\text {pre - train}} = w _ {\text {rec}} \mathcal {L} _ {\text {rec}} + w _ {\text {JEPA}} \mathcal {L} _ {\text {JEPA}},
$$

where $w _ { \mathrm { r e c } }$ and $w _ { \mathrm { J E P A } }$ control the relative contribution of each term. Intuitively, $\mathcal { L } _ { \mathrm { r e c } }$ anchors learning to gene-level expression signals as in scGPT, while $\mathcal { L } _ { \mathrm { J E P A } }$ enforces consistency of cell-level representations in latent space. Optimizing both jointly yields embeddings that retain fine-grained information for expression recovery while improving robustness and semantic structure through representation-level prediction.

## 2.4 Downstream Finetuning

During downstream finetuning, we optimize Cell-JEPA using four objectives: (i) gene expression prediction (GEP); (ii) gene expression prediction for cell modeling (GEPC); (iii) elastic cell similarity (ECS); and (iv) JEPA prediction loss. We adopt GEP, GEPC, and ECS from scGPT [Cui et al., 2024], and extend the finetuning objective by additionally retaining $\mathcal { L } _ { \mathrm { J E P A } }$

## 2.4.1 GEP and GEPC Loss

**Gene Expression Prediction (GEP).** GEP is equivalent to the reconstruction loss $\mathcal { L } _ { r e c }$ in (2.4) used during pre-training. Using the student predictions $\hat { \mathbf { v } } _ { i } ,$ , we minimize mean squared error over the masked gene set $\mathcal { U } _ { \mathrm { m a s k } }$ :

$$
\mathcal {L} _ {\mathrm{GEP}} = \frac {1}{| \mathcal {U} _ {\mathrm{mask}} |} \sum_ {i \in \mathcal {U} _ {\mathrm{mask}}} \left(\hat {\mathbf {v}} _ {i} - v _ {i}\right) ^ {2}.
$$

**Gene Expression Prediction for Cell Modeling (GEPC).** GEPC is a complementary objective from scGPT to explicitly tie gene-level prediction to the global cell representation [Cui et al., 2024]. For each gene token i, we compute a query vector from its gene embedding $\mathbf { y } _ { i }$ and predict expression by an inner product with the cell embedding ê:

$$
\tilde {\mathbf {v}} _ {i} = f (\mathbf {y} _ {i}) ^ {T} \mathbf {W} \hat {\mathbf {e}},
$$

where $f ( \cdot )$ is an MLP and W is a learnable weight matrix. The GEPC loss is the MSE over masked genes:

$$
\mathcal {L} _ {\mathrm{GEPC}} = \frac {1}{| \mathcal {U} _ {\mathrm{mask}} |} \sum_ {i \in \mathcal {U} _ {\mathrm{mask}}} (\tilde {\mathbf {v}} _ {i} - v _ {i}) ^ {2}.
$$

7

<!-- page 8 of 26 -->

Together, GEP and GEPC provide complementary supervision: GEP enforces accurate masked-value recovery, while GEPC encourages the global cell embedding to be predictive of gene-level expression.

## 2.4.2 ECS Loss

**Elastic Cell Similarity (ECS).** ECS regularizes the cell embedding space by encouraging representations of biologically similar cells to be closer. While scGPT proposes an ECS regularizer based on pairwise similarity thresholds [Cui et al., 2024], we modify this component into the following forms given the intuition described in Section C.5. Consider a minibatch of $N$ cells with embeddings $\{ ( \mathbf { e } _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\mathbf { e } _ { i }$ represents the embedding of cell i (as defined in (2.2)) and $c _ { i }$ is its corresponding cell-type label. We define the cosine similarity between cells i and $j$ as:

$$
s _ {i j} = \frac {\cos (\mathbf {e} ^ {i} , \mathbf {e} ^ {j})}{\tau},
$$

where $\tau$ is a temperature hyperparameter.

We treat each cell in the minibatch as an anchor in turn. For a given anchor cell $i ,$ we identify cells in the minibatch that share the same cell type as:

$$
p _ {i j} = \left\{ \begin{array}{l l} \frac {1}{| \mathcal {P} (i) |}, & \text {if} c _ {j} = c _ {i}, \\ 0, & \text {otherwise.} \end{array} \right.
$$

The contrastive loss for anchor cell i is then given by:

$$
\mathcal {L} _ {\mathrm{ECS}, i} = - \sum_ {j = 1} ^ {N} p _ {i j} \log \frac {\exp (s _ {i j})}{\sum_ {k = 1} ^ {N} \exp (s _ {i k})}.
$$

We calculate the elastic cell similarity loss by averaging over all anchors in the minibatch:

$$
\mathcal {L} _ {\mathrm{ECS}} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathcal {L} _ {\mathrm{ECS}, i}.
$$

This objective acts as a continuous regularizer to enforce structured similarity in the learned cell embedding space.

## 2.4.3 JEPA and Overall Loss

We retain the JEPA loss term $\mathcal { L } _ { \mathrm { J E P A } }$ in (2.3) during finetuning. This allows the model to adapt embeddingspace mappings to distributional differences within the dataset.

The overall finetuning objective $\mathcal { L } _ { \mathrm { f i n e t u n e } }$ is defined as a weighted sum of the four losses as follows:

$$
\mathcal {L} _ {\text {finetune}} = w _ {\mathrm{GEP}} \mathcal {L} _ {\mathrm{GEP}} + w _ {\mathrm{GEPC}} \mathcal {L} _ {\mathrm{GEPC}} + w _ {\mathrm{ECS}} \mathcal {L} _ {\mathrm{ECS}} + w _ {\mathrm{JEPA}} \mathcal {L} _ {\mathrm{JEPA}}.
$$

For the scGPT baseline, we use the same finetuning objectives excluding $\mathcal { L } _ { \mathrm { l } }$ <sub>EPA</sub>:

$$
\mathcal {L} _ {\mathrm{finetune}} ^ {\mathrm{scGPT}} = w _ {\mathrm{GEP}} \mathcal {L} _ {\mathrm{GEP}} + w _ {\mathrm{GEPC}} \mathcal {L} _ {\mathrm{GEPC}} + w _ {\mathrm{ECS}} \mathcal {L} _ {\mathrm{ECS}}.
$$

8

<!-- page 9 of 26 -->

## 2.5 Perturbation Prediction

**Perturbation Objective.** We further finetune the pre-trained scGPT and Cell-JEPA models to predict gene perturbation effects. To incorporate perturbation information, we embed the perturbation label $p _ { i }$ of gene i using a standard ID embedding layer $f _ { \mathrm { p e r t u r b } } ( \cdot )$ in prediction model:

$$
\mathbf {p} _ {i} = f _ {\text {perturb}} (p _ {i}).
$$

We modify the gene token representation in (2.1) by incorporating the perturbation embedding:

$$
\tilde {\mathbf {z}} _ {i} ^ {\mathrm{pert}} = \mathbf {y} _ {i} + \mathbf {p} _ {i} + \mathbf {v} _ {i}.
$$

We pass the perturbed gene embeddings to the student encoder to generate predicted perturbed gene values. Let $\tilde { \mathbf { Z } } _ { \mathrm { p e r t } } = ( \mathbf { z } _ { \mathrm { c l s } } , \tilde { \mathbf { z } } _ { 1 } ^ { \mathrm { p e r t } } , . . . , \tilde { \mathbf { z } } _ { n } ^ { \mathrm { p e r t } } )$ , The prediction process is:

$$
\hat {\mathbf {H}} _ {\mathrm{pert}} = g _ {\mathrm{S}} (\tilde {\mathbf {Z}} _ {\mathrm{pert}}), \quad \hat {\mathbf {v}} _ {i} ^ {\mathrm{pert}} = r _ {\mathrm{pert}} (\hat {\mathbf {H}} _ {\mathrm{pert}}),
$$

where $r _ { \mathrm { p e r t } }$ is an MLP value head. We define the perturbation reconstruction loss as:

$$
\mathcal {L} _ {\mathrm{pert-pred}} = \frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {\mathbf {v}} _ {i} ^ {\mathrm{pert}} - v _ {i} ^ {\mathrm{pert}}) ^ {2},
$$

where $v _ { i } ^ { \mathrm { p e r t } }$ is the ground truth perturbed expression value for gene $i .$

**JEPA and Overall Loss.** Unlike the pre-training and finetuning settings, we do not apply masking to the gene value inputs. We define the student perturbed cell embedding as:

$$
\hat {\mathbf {e}} _ {\mathrm{pert}} = \hat {\mathbf {H}} _ {0} ^ {\mathrm{pert}}.
$$

We construct the target perturbed gene embedding as:

$$
\mathbf {z} _ {i} ^ {\mathrm{pert}} = \mathbf {y} _ {i} + \mathbf {p} _ {i} + \mathbf {v} _ {i} ^ {\mathrm{pert}},
$$

where $\mathbf { v } _ { i } ^ { \mathrm { p e r t } }$ is the perturbed gene expression embedding for gene i. Let $\mathbf { Z } _ { \mathrm { p e r t } } = ( \mathbf { z } _ { \mathrm { c l s } } , \mathbf { z } _ { 1 } ^ { \mathrm { p e r t } } , . . . , \mathbf { z } _ { n } ^ { \mathrm { p e r t } } )$ . The target encoder produces:

$$
\mathbf {H} _ {\mathrm{pert}} = g _ {\mathrm{T}} (\mathbf {Z} _ {\mathrm{pert}}), \quad \mathbf {e} _ {\mathrm{pert}} = \mathbf {H} _ {0} ^ {\mathrm{pert}}.
$$

We introduce a predictor head $p _ { \mathrm { p e r t } } ( \cdot )$ (an MLP) and apply it to the student perturbed embedding:

$$
\tilde {\mathbf {e}} _ {\mathrm{pert}} = p _ {\mathrm{pert}} (\hat {\mathbf {e}} _ {\mathrm{pert}}).
$$

We compute the perturbed JEPA loss analogously to (2.3):

$$
\mathcal {L} _ {\mathrm{JEPA}} ^ {\mathrm{pert}} = 1 - \frac {\tilde {\mathbf {e}} _ {\mathrm{pert}} ^ {\top} \mathrm{sg} (\mathbf {e} _ {\mathrm{pert}})}{\| \tilde {\mathbf {e}} _ {\mathrm{pert}} \| _ {2} \| \mathrm{sg} (\mathbf {e} _ {\mathrm{pert}}) \| _ {2}}.
$$

We also adopt the ECS formulation from Section 2.4.2 to define a perturbed ECS loss, where the similarity between two cells i and $j$ is computed as the cosine similarity between their perturbed cell embeddings

9

<!-- page 10 of 26 -->

$\hat { \mathbf { e } } _ { i } ^ { \mathrm { p e r t } }$ and $\hat { \mathbf { e } } _ { j } ^ { \mathrm { p e r t } }$

The overall perturbation objective is a weighted sum of all perturbed loss terms:

$$
\mathcal {L} _ {\mathrm{pert}} = w _ {\mathrm{pert-rec}} \mathcal {L} _ {\mathrm{pert-rec}} + w _ {\mathrm{JEPA}} ^ {\mathrm{pert}} \mathcal {L} _ {\mathrm{JEPA}} ^ {\mathrm{pert}} + w _ {\mathrm{ECS}} \mathcal {L} _ {\mathrm{ECS}}.
$$

For the scGPT baseline, we use the same objective but exclude the JEPA term:

$$
\mathcal {L} _ {\mathrm{pert}} ^ {\mathrm{scGPT}} = w _ {\mathrm{pert-rec}} \mathcal {L} _ {\mathrm{pert-rec}} + w _ {\mathrm{ECS}} \mathcal {L} _ {\mathrm{ECS}}.
$$

## 3 Experimental Studies

In this section, we first provide the foundation model pre-training recipe in Section 3.1. We then evaluate the pre-trained models in three downstream settings: (i) supervised finetuning for cell-type clustering in Section 3.2, (ii) zero-shot transfer for cell-type clustering in Section 3.3, and (iii) perturbation-response prediction Section 3.4.

## 3.1 Pre-training Recipe

To validate the benefits of the Cell-JEPA framework, we pre-train two foundation models using the same pre-training dataset and training recipe: Cell-JEPA and scGPT [Cui et al., 2024]. The only difference between the two models lies in the pre-training objective, as shown in Section 2.3.

**Pre-training Dataset.** For foundation model pre-training, we curate a large-scale human kidney scRNAseq corpus from the CELLxGENE Census [Program et al., 2024] (version 2023-05-15). We filter datasets by Organism (Homo sapiens) and Tissue (kidney), and restrict to RNA modality. This yields a heterogeneous collection of kidney cells spanning multiple studies, donors, and experimental conditions, with all constituent datasets listed in Section E.1. In total, the pre-training corpus contains **800K** cells.

**Pre-training Recipe.** We train Cell-JEPA using the pre-training objective described in Section 2.3. For the scGPT baseline, we follow the training objective described in Section C and Cui et al. [2024]. Implementation details and hyperparameter settings for training the Cell-JEPA model are provided in Section E.1.

## 3.2 Task 1: Finetuning on PBMC

**Task Description.** We extract cell embeddings from both Cell-JEPA and scGPT and evaluate representation quality primarily through quantitative clustering metrics, complemented by qualitative UMAP visualizations. We report biologically motivated evaluation metrics following Cui et al. [2024], Luecken et al. [2022].

**Dataset.** We evaluate on a peripheral blood mononuclear cell (PBMC) dataset, PBMC-10k, consisting of two scRNA-seq batches obtained from a healthy donor (nominally 4k and 8k cells) [10x Genomics, 2017a,b]. The data are preprocessed following the scvi-tools PBMC pipeline [Gayoso et al., 2022]. The processed dataset contains 3,346 highly variable genes, with 7,982 cells in the first batch and 4,008 cells in the second batch [Gayoso et al., 2022]. Cell-type annotations cover major immune populations, including B cells, CD4+ and CD8+ T cells, NK cells, monocytes, dendritic cells, and others.

10

<!-- page 11 of 26 -->

![Image block](images/page011-block01.png)

Figure 3: Comparison of Cell Embeddings from scGPT and Cell-JEPA. UMAP visualizations of cell embeddings on the PBMC-10K. (a) scGPT after finetuning (Task 1), (b) Cell-JEPA after finetuning (Task 1), (c) scGPT under zero-shot evaluation (Task 2), and (d) Cell-JEPA under zero-shot evaluation (Task 2).

**Training/Validation Split.** We adopt the data splitting strategy from [Cui et al., 2024]. We randomly split the PBMC 10K dataset into training and validation sets with a 9:1 ratio, using a fixed random seed of 42 to ensure reproducibility. UMAP visualizations were generated using the full PBMC 10K dataset.

**Finetuning Recipe.** We finetune our pre-trained Cell-JEPA and scGPT checkpoints on each dataset using the finetuning objectives described in Section 2.4. The implementation detail and hyperparameter we use for finetuning the Cell-JEPA model can be found in Section E.2.

**Evaluation Metrics.** We consider four embedding-quality metrics: average biological conservation (Avg-BIO), normalized mutual information $( \mathrm { N M I _ { c e l l } } )$ , average silhouette width $( \mathrm { A S W _ { c e l l } } )$ , and adjusted Rand index $( \mathrm { A R I _ { c e l l } } )$ . We provide detailed definitions in Section D.1.

**Results.** We generate single-cell embeddings for PBMC-10k and visualize them using UMAP, showing that the learned latent spaces of both models exhibit qualitative separation of annotated cell types. Figures 3a and 3b. Compared with the baseline scGPT model, Cell-JEPA achieves improved performance across all four clustering metrics (Table 1). These results suggest that augmenting scGPT with the JEPA objective facilitates learning more informative representations from single-cell gene expression data.

Table 1: Performance of Task 1: Finetuning on PBMC. AvgBIO score, $\mathrm { N M I _ { c e l l } ,   A S W _ { c e l l } , }$ , and $\mathrm { A R I _ { c e l l } }$ metrics for Cell-JEPA and scGPT after finetuning on PBMC-10K dataset. After task-specific finetuning, Cell-JEPA outperforms scGPT across all four metrics.

Table 2: Performance of Task 2: Zero-shot Evaluation on PBMC. AvgBIO score and $\mathrm { N M I _ { c e l l } ,   A S W _ { c e l l } }$ , and $\mathrm { A R I _ { c e l l } }$ metrics for pre-trained Cell-JEPA and scGPT model under zero-shot evaluation on the PBMC-10K dataset. Cell-JEPA outperforms scGPT.

| Model | AvgBIO | NMI<sub>cell</sub> | ASW<sub>cell</sub> | ARI<sub>cell</sub> |
| --- | --- | --- | --- | --- |
| scGPT | 0.7531 | 0.7652 | 0.7100 | 0.7842 |
| Cell-JEPA | 0.7830 | 0.7761 | 0.7256 | 0.8472 |

| Model | AvgBIO | NMI<sub>cell</sub> | ASW<sub>cell</sub> | ARI<sub>cell</sub> |
| --- | --- | --- | --- | --- |
| scGPT | 0.5288 | 0.5585 | 0.5329 | 0.4951 |
| Cell-JEPA | 0.7194 | 0.7690 | 0.5811 | 0.8081 |

## 3.3 Task 2: Zero-shot on PBMC

Next, we evaluate zero-shot representation quality without any task-specific finetuning. We perform inference using the pre-trained checkpoints and evaluate embeddings using the same metrics as in Section 3.2.

11

<!-- page 12 of 26 -->

**Results.** We generate single-cell embeddings for PBMC-10K via zero-shot inference and visualize them using UMAP (Figures 3c and 3d), which suggests stronger zero-shot separation of annotated cell types for Cell-JEPA than for scGPT. Moreover, across quantitative metrics, Cell-JEPA consistently outperforms scGPT in the zero-shot setting; for instance, it improves AvgBIO by 36% relative to scGPT (Table 2). Overall, these results indicate that the JEPA objective improves transferability of learned representations, enabling strong performance without downstream finetuning.

## 3.4 Task 3: Perturbation Prediction

**Task Description.** We explore whether improved representations benefit perturbation-response prediction within a fixed biological context. Given a cell’s baseline expression and a genetic perturbation, the goal is to predict post-perturbation expression. Both evaluation datasets (Norman, Adamson) use the K562 cell line; we therefore assess within-context generalization to held-out perturbations rather than cross-context transfer.

**Dataset.** We evaluate perturbation-response prediction on two Perturb-seq CRISPR screening datasets profiled in the K562 leukemia cell line: Adamson [Adamson et al., 2016] and Norman [Norman et al., 2019]. Adamson is a CRISPRi Perturb-seq dataset containing 86 single-gene perturbations (GEARS-processed), together with matched control cells. Norman is a CRISPRa Perturb-seq dataset containing 105 single-gene perturbations and 131 two-gene perturbations (GEARS-processed), with hundreds of cells per perturbation on average. All datasets are processed using the GEARS pipeline [Roohani et al., 2023].

**Training/Validation/Testing Split.** We follow the GEARS simulation split protocol, which partitions data at the perturbation level rather than the cell level. Under this setting, perturbations in the test set are held out during training. In Norman, held-out two-gene perturbations are further grouped into combo\_seen2, combo\_seen1, and combo\_seen0, depending on whether both, one, or neither constituent gene was observed as a single-gene perturbation during training [Roohani et al., 2023].

**Finetuning Recipe.** We finetune scGPT and our Cell-JEPA model using the official scGPT perturbation finetuning pipeline, which leverages GEARS for dataset preprocessing, split construction, and evaluation. In this setup, a Transformer-based predictor (scGPT, also adapted for Cell-JEPA) is trained to model post-perturbation gene expression. Each finetuning data point consists of (i) an unperturbed (control) gene expression profile, (ii) binary perturbation indicators specifying the perturbed gene(s), and (iii) the groundtruth post-perturbation expression profile. Both models are trained for 15 epochs with learning rate $1 0 ^ { - 4 }$ and batch size 64. The training takes 2 hours on a single A100 GPU. We provide the full training configuration details in Section E.3.

**Evaluation Metrics.** Following GEARS [Roohani et al., 2023], we evaluate perturbation prediction using Pearson correlation between predicted and observed post-perturbation gene expression vectors. We report pearson over all genes, pearson\_de restricted to differentially expressed genes, and top20\_de\_non\_dropout restricted to top-ranked DE genes after filtering dropout genes. Full details are provided in Section D.2.

**Results.** Table 3 summarizes performance on GEARS unseen\_single perturbations for Adamson and Norman. Cell-JEPA consistently improves both absolute state prediction and DE-focused recovery relative to the scGPT baseline, with the largest gains observed on Norman. These results suggest that augmenting

12

<!-- page 13 of 26 -->

Table 3: Performance of Task 3: Perturbation Pre-diction on Unseen Single-gene Perturbations. Compared with scGPT, Cell-JEPA improves absolute post-perturbation state prediction and DE-focused recovery on both the Norman [Norman et al., 2019] and Adamson [Adamson et al., 2016] datasets. pear denotes pearson, pear\_de denotes pearson\_de, and top20 denotes top20\_de\_non\_dropout.

Table 4: Delta-based Metrics on Unseen Singlegene Perturbations. Delta correlations remain low for both models, and Cell-JEPA shows reduced performance on both datasets, with the largest decrease observed on Adamson [Adamson et al., 2016]. $\Delta \mathtt { \_ p e a r }$ denotes pearson\_delta, $\Delta \_ \mathtt { p e a r \_ d e }$ denotes pearson\_delta\_de, and $\Delta \_ \mathtt { t o p 2 0 }$ denotes pearson\_delta\_top20\_de\_non\_dropout.

| Dataset | Model | pear | pear_de | top20 |
| --- | --- | --- | --- | --- |
| Norman | scGPT | 0.631 | 0.276 | 0.278 |
| Norman | Cell-JEPA | 0.787 | 0.592 | 0.565 |
| Adamson | scGPT | 0.905 | 0.699 | 0.677 |
| Adamson | Cell-JEPA | 0.937 | 0.741 | 0.720 |

| Dataset | Model | ∆_pear | ∆_pear_de | ∆_top20 |
| --- | --- | --- | --- | --- |
| Norman | scGPT | 0.103 | 0.105 | 0.169 |
| Norman | JEPA | 0.059 | 0.041 | 0.115 |
| Adamson | scGPT | 0.173 | 0.101 | 0.120 |
| Adamson | JEPA | 0.120 | -0.176 | -0.152 |

scGPT finetuning with a latent-space JEPA objective yields representations that better generalize to unseen genetic perturbations.

We also report delta-based metrics (pearson\_delta, pearson\_de\_delta, delta\_top20\_de\_non\_dropout), which evaluate correlation on expression changes relative to the control state (post-perturbation minus control), and thus emphasize effect-size accuracy rather than absolute post-perturbation state prediction. As shown in Table 4, both scGPT and Cell-JEPA achieve low delta correlations on unseen single-gene perturbations, indicating that accurate change-from-control estimation remains challenging in this regime. While Cell-JEPA improves absolute-state metrics (Table 3), it does not consistently improve delta-based metrics and exhibits the largest decrease on Adamson.

A plausible explanation is that the JEPA objective encourages learning features invariant across masked views, which benefits recognizing cellular states but may suppress sensitivity to perturbation-induced deviations. This suggests that representation learning (what state is this cell in?) and perturbation modeling (how does a cell change?) address complementary aspects of cellular prediction and may require distinct approaches. Overall, we view these results as highlighting a trade-off between absolute-state fidelity and effect-size accuracy, and suggest that scaling to larger perturbation datasets and/or explicitly training with delta-aware objectives may be necessary to fully realize the benefits of latent-space predictive learning for perturbation generalization.

## 4 Conclusion

We introduced Cell-JEPA, a joint-embedding predictive framework that complements reconstruction-based pre-training with latent-space prediction. By predicting cell-level embeddings rather than reconstructing noisy counts, Cell-JEPA learns representations robust to the dropout artifacts that dominate single-cell measurements.

Our primary finding is a 36% relative improvement in zero-shot cell-type clustering, demonstrating that latent prediction yields representations that transfer without task-specific finetuning. On perturbation prediction within a single cell line, we observe improved absolute-state reconstruction but not effect-size estimation. This suggests that learning robust cell representations and modeling perturbation effects are complementary challenges that may require distinct approaches.

13

<!-- page 14 of 26 -->

These results establish latent-space prediction as a promising direction for single-cell representation learning, while highlighting that perturbation modeling remains an open problem likely requiring explicit context modeling or greater contextual diversity in training data. We leave limitations in Section A and related work in Section B.

## Acknowledgments

We thank the Quest high-performance computing facility at Northwestern University for computational resources and staff support; Quest is jointly supported by the Office of the Provost, the Office for Research, and Northwestern University Information Technology.

H.L. was supported by NIH R01LM1372201, NSF AST-2421845, Simons Foundation MPS-AI-00010513, AbbVie, Dolby, and a Chan Zuckerberg Biohub Chicago Spoke Award. A.A.K. was supported by NIH DP2AI177884, a Chan Zuckerberg Investigator Award, and a Common Mechanisms of Autoimmunity Insight Award. M.K. was supported by NIH T32GM139782.

The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding agencies.

## References

10x Genomics. 4k pbmcs from a healthy donor (single cell gene expression dataset by cell ranger v2.1.0). 10x Genomics public dataset, 2017a.

10x Genomics. 8k pbmcs from a healthy donor (single cell gene expression dataset by cell ranger v2.1.0). 10x Genomics public dataset, 2017b.

Britt Adamson, Thomas M. Norman, Marco Jost, Min Y. Cho, James K. Nuñez, Yuwen Chen, Jacqueline E. Villalta, Luke A. Gilbert, Max A. Horlbeck, Marco Y. Hein, Ryan A. Pak, Andrew N. Gray, Carol A. Gross, Atray Dixit, Oren Parnas, Aviv Regev, and Jonathan S. Weissman. A multiplexed single-cell crispr screening platform enables systematic dissection of the unfolded protein response. Cell, 167(7): 1867–1882.e21, Dec 2016.

Constantin Ahlmann-Eltze, Wolfgang Huber, and Simon Anders. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. Nature Methods, 22(8):1657–1661, 2025.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture, 2023.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025.

14

<!-- page 15 of 26 -->

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Junyue Cao, Diana R. O’Day, Hannah A. Pliner, Paul D. Kingsley, Mei Deng, Riza M. Daza, Michael A. Zager, Kimberly A. Aldinger, Ronnie Blecher-Gonen, Fan Zhang, Malte Spielmann, James Palis, Dan Doherty, Frank J. Steemers, Ian A. Glass, Cole Trapnell, and Jay Shendure. A human cell atlas of fetal gene expression. Science, 370(6518), November 2020.

The Tabula Sapiens Consortium\*, Robert C. Jones, Jim Karkanias, Mark A. Krasnow, Angela Oliveira Pisco, Stephen R. Quake, Julia Salzman, Nir Yosef, Bryan Bulthaup, Phillip Brown, William Harper, Marisa Hemenez, Ravikumar Ponnusamy, Ahmad Salehi, Bhavani A. Sanagavarapu, Eileen Spallino, Ksenia A. Aaron, Waldo Concepcion, James M. Gardner, Burnett Kelly, Nikole Neidlinger, Zifa Wang, Sheela Crasta, Saroja Kolluru, Maurizio Morri, Serena Y. Tan, Kyle J. Travaglini, Chenling Xu, Marcela Alcántara-Hernández, Nicole Almanzar, Jane Antony, Benjamin Beyersdorf, Deviana Burhan, Kruti Calcuttawala, Matthew M. Carter, Charles K. F. Chan, Charles A. Chang, Stephen Chang, Alex Colville, Rebecca N. Culver, Ivana Cvijović, Gaetano D’Amato, Camille Ezran, Francisco X. Galdos, Astrid Gillich, William R. Goodyer, Yan Hang, Alyssa Hayashi, Sahar Houshdaran, Xianxi Huang, Juan C. Irwin, SoRi Jang, Julia Vallve Juanico, Aaron M. Kershner, Soochi Kim, Bernhard Kiss, William Kong, Maya E. Kumar, Angera H. Kuo, Rebecca Leylek, Baoxiang Li, Gabriel B. Loeb, Wan-Jin Lu, Sruthi Mantri, Maxim Markovic, Patrick L. McAlpine, Antoine de Morree, Karim Mrouj, Shravani Mukherjee, Tyler Muser, Patrick Neuhöfer, Thi D. Nguyen, Kimberly Perez, Ragini Phansalkar, Nazan Puluca, Zhen Qi, Poorvi Rao, Hayley Raquer-McKay, Nicholas Schaum, Bronwyn Scott, Bobak Seddighzadeh, Joe Segal, Sushmita Sen, Shaheen Sikandar, Sean P. Spencer, Lea C. Steffes, Varun R. Subramaniam, Aditi Swarup, Michael Swift, Will Van Treuren, Emily Trimm, Stefan Veizades, Sivakamasundari Vijayakumar, Kim Chi Vo, Sevahn K. Vorperian, Wanxin Wang, Hannah N. W. Weinstein, Juliane Winkler, Timothy T. H. Wu, Jamie Xie, Andrea R. Yung, Yue Zhang, Angela M. Detweiler, Honey Mekonen, Norma F. Neff, Rene V. Sit, Michelle Tan, Jia Yan, Gregory R. Bean, Vivek Charu, Erna Forgó, Brock A. Martin, Michael G. Ozawa, Oscar Silva, Angus Toland, Venkata N. P. Vemuri, Shaked Afik, Kyle Awayan, Olga Borisovna Botvinnik, Ashley Byrne, Michelle Chen, Roozbeh Dehghannasiri, Adam Gayoso, Alejandro A. Granados, Qiqing Li, Gita Mahmoudabadi, Aaron McGeever, Julia Eve Olivieri, Madeline Park, Neha Ravikumar, Geoff Stanley, Weilun Tan, Alexander J. Tarashansky, Rohan Vanheusden, Peter Wang, Sheng Wang, Galen Xing, Les Dethlefsen, Po-Yi Ho, Shixuan Liu, Jonathan S. Maltzman, Ross J. Metzger, Koki Sasagawa, Rahul Sinha, Hanbing Song, Bruce Wang, Steven E. Artandi, Philip A. Beachy, Michael F. Clarke, Linda C. Giudice, Franklin W. Huang, Kerwyn Casey Huang, Juliana Idoyaga, Seung K. Kim, Christin S. Kuo, Patricia Nguyen, Thomas A. Rando, Kristy Red-Horse, Jeremy Reiter, David A. Relman, Justin L. Sonnenburg, Albert Wu, Sean M. Wu, and Tony Wyss-Coray. The tabula sapiens: A multiple-organ, single-cell transcriptomic atlas of humans. Science, 376(6594), May 2022.

Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan, and Bo Wang. scgpt: toward building a foundation model for single-cell multi-omics using generative ai. Nature Methods, 21 (8):1470–1480, Aug 2024.

Jun Ding, Nadav Sharon, and Ziv Bar-Joseph. Temporal modelling using single-cell transcriptomics. Nature Reviews Genetics, 23(6):355–368, Jun 2022.

Adam Gayoso, Romain Lopez, Galen Xing, Pierre Boyeau, Valeh Valiollah Pour Amiri, Justin Hong, Kather-

15

<!-- page 16 of 26 -->

ine Wu, Michael Jayasuriya, Edouard Mehlman, Maxime Langevin, Yining Liu, Jules Samaran, Gabriel Misrachi, Achille Nazaret, Oscar Clivio, Chenling Xu, Tal Ashuach, Mariano Gabitto, Mohammad Lotfollahi, Valentine Svensson, Eduardo da Veiga Beltrame, Vitalii Kleshchevnikov, Carlos Talavera-López, Lior Pachter, Fabian J. Theis, Aaron Streets, Michael I. Jordan, Jeffrey Regier, and Nir Yosef. A python library for probabilistic analysis of single-cell omics data. Nature Biotechnology, 40(2):163–166, Feb 2022.

Xiaoping Han, Ziming Zhou, Lijiang Fei, Huiyu Sun, Renying Wang, Yao Chen, Haide Chen, Jingjing Wang, Huanna Tang, Wenhao Ge, Yincong Zhou, Fang Ye, Mengmeng Jiang, Junqing Wu, Yanyu Xiao, Xiaoning Jia, Tingyue Zhang, Xiaojie Ma, Qi Zhang, Xueli Bai, Shujing Lai, Chengxuan Yu, Lijun Zhu, Rui Lin, Yuchi Gao, Min Wang, Yiqing Wu, Jianming Zhang, Renya Zhan, Saiyong Zhu, Hailan Hu, Changchun Wang, Ming Chen, He Huang, Tingbo Liang, Jianghua Chen, Weilin Wang, Dan Zhang, and Guoji Guo. Construction of a human cell landscape at single-cell level. Nature, 581(7808):303–309, March 2020.

Minsheng Hao, Jing Gong, Xin Zeng, Chiming Liu, Yucheng Guo, Xingyi Cheng, Taifeng Wang, Jianzhu Ma, Xuegong Zhang, and Le Song. Large-scale foundation model on single-cell transcriptomics. Nature Methods, 21(8):1481–1491, June 2024.

Hai Huang, Yann LeCun, and Randall Balestriero. Llm-jepa: Large language models meet joint embedding predictive architectures, 2025.

Matthias Kretzler, Edgar Otto, Christopher O’Connor, Markus Bitzer, and Rajasree Menon. Hca kidney seed network: University of michigan, 2025.

David Lähnemann, Johannes Köster, Ewa Szczurek, Davis J McCarthy, Stephanie C Hicks, Mark D Robinson, Catalina A Vallejos, Kieran R Campbell, Niko Beerenwinkel, Ahmed Mahfouz, et al. Eleven grand challenges in single-cell data science. Genome biology, 21(1):31, 2020.

Blue B. Lake, Rajasree Menon, Seth Winfree, Qiwen Hu, Ricardo Melo Ferreira, Kian Kalhor, Daria Barwinska, Edgar A. Otto, Michael Ferkowicz, Dinh Diep, Nongluk Plongthongkum, Amanda Knoten, Sarah Urata, Abhijit S. Naik, Sean Eddy, Bo Zhang, Yan Wu, Diane Salamon, James C. Williams, Xin Wang, Karol S. Balderrama, Paul Hoover, Evan Murray, Anitha Vijayan, Fei Chen, Sushrut S. Waikar, Sylvia Rosas, Francis P. Wilson, Paul M. Palevsky, Krzysztof Kiryluk, John R. Sedor, Robert D. Toto, Chirag Parikh, Eric H. Kim, Evan Z. Macosko, Peter V. Kharchenko, Joseph P. Gaut, Jeffrey B. Hodgin, Michael T. Eadon, Pierre C. Dagher, Tarek M. El-Achkar, Kun Zhang, Matthias Kretzler, and Sanjay Jain. An atlas of healthy and injured cell states and niches in the human kidney. July 2021.

Yann Lecun. A Path Towards Autonomous Machine Intelligence. 2022.

Elon Litman, Tyler Myers, Vinayak Agarwal, Ekansh Mittal, Orion Li, Ashwin Gopinath, and Timothy Kassis. Genejepa: A predictive world model of the transcriptome. bioRxiv, 2025.

Romain Lopez, Jeffrey Regier, Michael B. Cole, Michael I. Jordan, and Nir Yosef. Deep generative modeling for single-cell transcriptomics. Nature Methods, 15(12):1053–1058, November 2018.

Malte D Luecken and Fabian J Theis. Current best practices in single-cell RNA-seq analysis: a tutorial. Mol. Syst. Biol., 15(6):e8746, June 2019.

16

<!-- page 17 of 26 -->

Malte D. Luecken, M. Büttner, K. Chaichoompu, A. Danese, M. Interlandi, M. F. Mueller, D. C. Strobl, L. Zappia, M. Dugas, M. Colomé-Tatché, and Fabian J. Theis. Benchmarking atlas-level data integration in single-cell genomics. Nature Methods, 19(1):41–50, Jan 2022.

Yoshiharu Muto, Parker C. Wilson, Nicolas Ledru, Haojia Wu, Henrik Dimke, Sushrut S. Waikar, and Benjamin D. Humphreys. Single cell transcriptional and chromatin accessibility profiling redefine cellular heterogeneity in the adult human kidney. Nature Communications, 12(1), April 2021.

Thomas M. Norman, Max A. Horlbeck, Joseph M. Replogle, Alex Y. Ge, Albert Xu, Marco Jost, Luke A. Gilbert, and Jonathan S. Weissman. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. Science, 365(6455):786–793, 2019.

James D Pearce, Sara E Simmonds, Gita Mahmoudabadi, Lakshmi Krishnan, Giovanni Palla, Ana-Maria Istrate, Alexander Tarashansky, Benjamin Nelson, Omar Valenzuela, Donghui Li, Stephen R Quake, and Theofanis Karaletsos. A cross-species generative cell atlas across 1.5 billion years of evolution: The transcriptformer single-cell model. bioRxiv, 2025.

CZI Cell Science Program, Shibla Abdulla, Brian Aevermann, Pedro Assis, Seve Badajoz, Sidney M Bell, Emanuele Bezzi, Batuhan Cakir, Jim Chaffer, Signe Chambers, J Michael Cherry, Tiffany Chi, Jennifer Chien, Leah Dorman, Pablo Garcia-Nieto, Nayib Gloria, Mim Hastie, Daniel Hegeman, Jason Hilton, Timmy Huang, Amanda Infeld, Ana-Maria Istrate, Ivana Jelic, Kuni Katsuya, Yang Joon Kim, Karen Liang, Mike Lin, Maximilian Lombardo, Bailey Marshall, Bruce Martin, Fran McDade, Colin Megill, Nikhil Patel, Alexander Predeus, Brian Raymor, Behnam Robatmili, Dave Rogers, Erica Rutherford, Dana Sadgat, Andrew Shin, Corinn Small, Trent Smith, Prathap Sridharan, Alexander Tarashansky, Norbert Tavares, Harley Thomas, Andrew Tolopko, Meghan Urisko, Joyce Yan, Garabet Yeretssian, Jennifer Zamanian, Arathi Mani, Jonah Cool, and Ambrose Carr. Cz cellxgene discover: a single-cell data platform for scalable exploration, analysis and modeling of aggregated data. Nucleic Acids Research, 53(D1):D886–D900, 11 2024.

Yusuf Roohani, Kexin Huang, and Jure Leskovec. Predicting transcriptional outcomes of novel multigene perturbations with gears. Nature Biotechnology, 2023.

Yanay Rosen, Yusuf Roohani, Ayush Agrawal, Leon Samotorcan, Tabula Sapiens Consortium, Stephen R. Quake, and Jure Leskovec. Universal cell embeddings: A foundation model for cell biology. November 2023.

Benjamin J. Stewart, John R. Ferdinand, Matthew D. Young, Thomas J. Mitchell, Kevin W. Loudon, Alexandra M. Riding, Nathan Richoz, Gordon L. Frazer, Joy U. L. Staniforth, Felipe A. Vieira Braga, Rachel A. Botting, Dorin-Mirel Popescu, Roser Vento-Tormo, Emily Stephenson, Alex Cagan, Sarah J. Farndon, Krzysztof Polanski, Mirjana Efremova, Kile Green, Martin Del Castillo Velasco-Herrera, Charlotte Guzzo, Grace Collord, Lira Mamanova, Tevita Aho, James N. Armitage, Antony C. P. Riddick, Imran Mushtaq, Stephen Farrell, Dyanne Rampling, James Nicholson, Andrew Filby, Johanna Burge, Steven Lisgo, Susan Lindsay, Marc Bajenoff, Anne Y. Warren, Grant D. Stewart, Neil Sebire, Nicholas Coleman, Muzlifah Haniffa, Sarah A. Teichmann, Sam Behjati, and Menna R. Clatworthy. Spatiotemporal immune zonation of the human kidney. Science, 365(6460):1461–1466, September 2019.

Chenqu Suo, Emma Dann, Issac Goh, Laura Jardine, Vitalii Kleshchevnikov, Jong-Eun Park, Rachel A. Botting, Emily Stephenson, Justin Engelbert, Zewen Kelvin Tuong, Krzysztof Polanski, Nadav Yayon,

17

<!-- page 18 of 26 -->

Chuan Xu, Ondrej Suchanek, Rasa Elmentaite, Cecilia Domínguez Conde, Peng He, Sophie Pritchard, Mohi Miah, Corina Moldovan, Alexander S. Steemers, Pavel Mazin, Martin Prete, Dave Horsfall, John C. Marioni, Menna R. Clatworthy, Muzlifah Haniffa, and Sarah A. Teichmann. Mapping the developing human immune system across organs. Science, 376(6597), June 2022.

Valentine Svensson, Roser Vento-Tormo, and Sarah A. Teichmann. Exponential scaling of single-cell rna-seq in the past decade. Nature Protocols, 13(4):599–604, Apr 2018.

Fuchou Tang, Kaiqin Lao, and M. Azim Surani. Development and applications of single-cell transcriptome analysis. Nature Methods, 8(4):S6–S11, Apr 2011.

Christina V. Theodoris, Ling Xiao, Anant Chopra, Mark D. Chaffin, Zeina R. Al Sayed, Matthew C. Hill, Helene Mantineo, Elizabeth M. Brydon, Zexian Zeng, X. Shirley Liu, and Patrick T. Ellinor. Transfer learning enables predictions in network biology. Nature, 618(7965):616–624, May 2023.

Ramon Viñas Torné, Maciej Wiatrak, Zoe Piran, Shuyang Fan, Liangze Jiang, Sarah A Teichmann, Mor Nitzan, and Maria Brbić. Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation. Nature Biotechnology, pages 1–10, 2025.

Parker C. Wilson, Yoshiharu Muto, Haojia Wu, Anil Karihaloo, Sushrut S. Waikar, and Benjamin D. Humphreys. Multimodal single cell sequencing implicates chromatin accessibility and genetic background in diabetic kidney disease progression. Nature Communications, 13(1), September 2022.

Weimin Wu, Xuefeng Song, Yibo Wen, Qinjie Lin, Zhihan Zhou, Jerry Yao-Chieh Hu, Zhong Wang, and Han Liu. Genome-factory: An integrated library for tuning, deploying, and interpreting genomic models. arXiv preprint arXiv:2509.12266, 2025.

Fan Yang, Wenchuan Wang, Fang Wang, Yuan Fang, Duyu Tang, Junzhou Huang, Hui Lu, and Jianhua Yao. scbert as a large-scale pretrained deep language model for cell type annotation of single-cell rna-seq data. Nature Machine Intelligence, 4(10):852–866, September 2022.

Xiaodong Yang, Guole Liu, Guihai Feng, Dechao Bu, Pengfei Wang, Jie Jiang, Shubai Chen, Qinmeng Yang, Hefan Miao, Yiyang Zhang, Zhenpeng Man, Zhongming Liang, Zichen Wang, Yaning Li, Zheng Li, Yana Liu, Yao Tian, Wenhao Liu, Cong Li, Ao Li, Jingxi Dong, Zhilong Hu, Chen Fang, Lina Cui, Zixu Deng, Haiping Jiang, Wentao Cui, Jiahao Zhang, Zhaohui Yang, Handong Li, Xingjian He, Liqun Zhong, Jiaheng Zhou, Zijian Wang, Qingqing Long, Ping Xu, Xin Li, Hongmei Wang, Baoyang Hu, Wei Li, Fei Gao, Jingtao Guo, Leqian Yu, Qi Gu, Weiwei Zhai, Zhengting Zou, Guihai Feng, Wenhao Liu, Yao Tian, Chen Fang, Jingxi Dong, Yana Liu, Jingqi Yu, Wenhui Wu, Xinxin Lin, Cong Li, Yu Zou, Yongshun Ren, Fan Li, Yixiao Zhao, Yike Xin, Longfei Han, Shuyang Jiang, Kai Ma, Qicheng Chen, Haoyuan Wang, Huanhuan Wu, Chaofan He, Yilong Hu, Shuyu Guo, Yiyun Li, Yuanchun Zhou, Yangang Wang, Xuezhi Wang, Pengfei Wang, Fei Li, Zhen Meng, Zaitian Wang, Ping Xu, Wentao Cui, Zhilong Hu, Huimin He, Shan Zong, Jiajia Wang, Yan Chen, Chunyang Zhang, Chengrui Wang, Ran Zhang, Meng Xiao, Yining Wang, Yiqiang Chen, Yi Zhao, Xiaodong Yang, Dechao Bu, Xin Qin, Jiaxin Qin, Zhaohui Yang, Chenhao Li, Zhufeng Xu, Zeyuan Zhang, Xiaoning Qi, Shubai Chen, Wuliang Huang, Yaning Li, Ge Yang, Jing Liu, Guole Liu, Liqun Zhong, Yaoru Luo, Jiaheng Zhou, Zichen Wang, Qinxuan Luo, Ziwen Liu, Ao Li, Teng Wang, Yiming Huang, Handong Li, Yong Wang, Shihua Zhang, Jiahao Zhang, Yiyang Zhang, Shirui Li, Zhongming Liang, Zhenpeng Man, Kangning Dong, Qunlun Shen, Hongmei Wang, Zhen Meng, Xuezhi Wang, Yangang Wang, Yong Wang, Shihua Zhang, Jingtao Guo, Yi Zhao,

18

<!-- page 19 of 26 -->

Yuanchun Zhou, Fei Li, Jing Liu, Yiqiang Chen, Ge Yang, and Xin Li. Genecompass: deciphering universal gene regulatory mechanisms with a knowledge-informed cross-species foundation model. Cell Research, 34(12):830–845, October 2024.

Matthew D. Young, Thomas J. Mitchell, Felipe A. Vieira Braga, Maxine G. B. Tran, Benjamin J. Stewart, John R. Ferdinand, Grace Collord, Rachel A. Botting, Dorin-Mirel Popescu, Kevin W. Loudon, Roser Vento-Tormo, Emily Stephenson, Alex Cagan, Sarah J. Farndon, Martin Del Castillo Velasco-Herrera, Charlotte Guzzo, Nathan Richoz, Lira Mamanova, Tevita Aho, James N. Armitage, Antony C. P. Riddick, Imran Mushtaq, Stephen Farrell, Dyanne Rampling, James Nicholson, Andrew Filby, Johanna Burge, Steven Lisgo, Patrick H. Maxwell, Susan Lindsay, Anne Y. Warren, Grant D. Stewart, Neil Sebire, Nicholas Coleman, Muzlifah Haniffa, Sarah A. Teichmann, Menna Clatworthy, and Sam Behjati. Singlecell transcriptomes from human kidneys reveal the cellular identity of renal tumors. Science, 361(6402): 594–599, August 2018.

Matthew D. Young, Thomas J. Mitchell, Lars Custers, Thanasis Margaritis, Francisco Morales-Rodriguez, Kwasi Kwakwa, Eleonora Khabirova, Gerda Kildisiute, Thomas R. W. Oliver, Ronald R. de Krijger, Marry M. van den Heuvel-Eibrink, Federico Comitani, Alice Piapi, Eva Bugallo-Blanco, Christine Thevanesan, Christina Burke, Elena Prigmore, Kirsty Ambridge, Kenny Roberts, Felipe A. Vieira Braga, Tim H. H. Coorens, Ignacio Del Valle, Anna Wilbrey-Clark, Lira Mamanova, Grant D. Stewart, Vincent J. Gnanapragasam, Dyanne Rampling, Neil Sebire, Nicholas Coleman, Liz Hook, Anne Warren, Muzlifah Haniffa, Marcel Kool, Stefan M. Pfister, John C. Achermann, Xiaoling He, Roger A. Barker, Adam Shlien, Omer A. Bayraktar, Sarah A. Teichmann, Frank C. Holstege, Kerstin B. Meyer, Jarno Drost, Karin Straathof, and Sam Behjati. Single cell derived mrna signals across human kidney tumors. Nature Communications, 12(1), June 2021.

Yuping Zhang, Sathiya P. Narayanan, Rahul Mannan, Gregory Raskind, Xiaoming Wang, Pankaj Vats, Fengyun Su, Noshad Hosseini, Xuhong Cao, Chandan Kumar-Sinha, Stephanie J. Ellison, Thomas J. Giordano, Todd M. Morgan, Sethuramasundaram Pitchiaya, Ajjai Alva, Rohit Mehra, Marcin Cieslik, Saravana M. Dhanasekaran, and Arul M. Chinnaiyan. Single-cell analyses of renal cell cancers reveal insights into tumor microenvironment, cell of origin, and therapy response. Proceedings of the National Academy of Sciences, 118(24), June 2021.

Zhihan Zhou, Weimin Wu, Harrison Ho, Jiayi Wang, Lizhen Shi, Ramana V Davuluri, Zhong Wang, and Han Liu. Dnabert-s: Learning species-aware dna embedding with genome foundation models. arXiv preprint arXiv:2402.08777, 10, 2024.

Zhihan Zhou, Robert Riley, Satria Kautsar, Weimin Wu, Rob Egan, Steven Hofmeyr, Shira Goldhaber-Gordon, Mutian Yu, Harrison Ho, Fengchen Liu, et al. Genomeocean: An efficient genome foundation model trained on large-scale metagenomic assemblies. bioRxiv, pages 2025–01, 2025.

19

<!-- page 20 of 26 -->

## Appendix

- **A Limitations 20**
- **B Related Work 21**
- **C scGPT 22**
- C.1 Tokenization and Input Representation 22
- C.2 Autoregressive Gene Generation via Masked Attention 22
- C.3 Pretraining Objective 23
- C.4 Iterative Autoregressive Inference 23
- C.5 Issues with the ECS Loss in scGPT 23
- **D Evaluation Metrics 24**
- D.1 Average Biological Conservation (AvgBIO) 24
- D.2 Evaluation Metrics for Perturbation Prediction 24
- **E Training Recipe 26**
- E.1 Pretraining Recipe 26
- E.2 PBMC-10K Finetuning Recipe 26
- E.3 Perturbation Finetuning Recipe 26

## A Limitations

Despite the improvements demonstrated by Cell-JEPA, five limitations remain.

• **Limited Pre-training Dataset.** Due to computational constraints, we pre-train our models on a kidney-specific dataset rather than the full human single-cell atlas used by Cui et al. [2024], which contains approximately 33 million cells. From a biological perspective, Cell-JEPA is therefore exposed to a limited diversity of cell types, and its ability to generalize to other tissues, rare cell populations, and disease states has not been systematically evaluated. To ensure a fair comparison with the scGPT baseline under this constraint, we adopt identical hyperparameters and mini-batch sizes, which preclude task-specific hyperparameter optimization for Cell-JEPA.

• **Model Scale.** We constrain the model size to 28 million parameters (single student or teacher model), which may be insufficient to fully leverage substantially larger and more heterogeneous pre-training datasets. Scaling Cell-JEPA to larger model capacities and conducting large-scale foundation model pre-training are left to future work.

• **Limited Evaluation Scope across Model Families.** Our experimental evaluation applies the JEPA training framework exclusively to scGPT and compares performance with and without JEPA under this fixed backbone. We do not extend the JEPA objective to other single-cell foundation model architectures, and assessing its generality across alternative training pipelines remains an important direction for future exploration.

• **Delta-based Evaluation Metrics.** While our method improves absolute post-perturbation state prediction, performance on delta-based evaluation metrics remains limited. This suggests persistent challenges in accurately modeling effect-size changes relative to control conditions.

• **Residual Technical and Biological Confounders.** Despite the robustness of latent-space JEPA objectives, learned embeddings may still reflect residual batch effects, donor-specific variability, and differences in sequencing depth, which could influence downstream biological interpretation.

20

<!-- page 21 of 26 -->

• **Perturbation evaluation scope.** Our perturbation experiments use two datasets from a single cell line (K562), assessing generalization to held-out perturbations within a fixed biological context. We do not evaluate cross-context transfer (e.g., across cell types, donors, or activation states), which recent work suggests is the primary challenge for perturbation prediction [Ahlmann-Eltze et al., 2025]. We also do not compare against simple baselines such as mean or additive models, which have proven competitive in systematic evaluations [Viñas Torné et al., 2025]. Our perturbation results should therefore be interpreted as preliminary exploration rather than a solution to the perturbation prediction problem.

## B Related Work

In this section, we review prior work on the development of single-cell foundation models and the application of joint embedding predictive architectures in other domains.

**Single Cell Foundation Models.** Large foundation models have demonstrated strong potential for modeling complex data across diverse domains, including language and biology [Zhou et al., 2025, Wu et al., 2025, Zhou et al., 2024, Brown et al., 2020]. The adaptation of large foundation models to transcrip tomic data has reshaped representation learning in single-cell biology. Early deep probabilistic approaches such as scVI [Lopez et al., 2018] introduced variational inference with encoder–decoder architectures to model scRNA-seq counts, learning latent representations that help account for batch effects and library-size variation. Building on this foundation, Transformer-based methods learn context-aware representations by treating each cell as a collection of gene tokens (often arranged into a pseudo-sequence) and applying self-supervised objectives. Within this paradigm, masked modeling approaches such as scBERT [Yang et al., 2022] and Geneformer [Theodoris et al., 2023] pre-train Transformer encoders to recover masked genes from cellular context, while generative methods such as scGPT [Cui et al., 2024] and TranscriptFormer [Pearce et al., 2025] learn to model gene expression by predicting missing values under structured conditioning schemes. Recent work has focused on scaling model capacity and incorporating cross-species knowledge. For instance, scFoundation [Hao et al., 2024] leverages xTrimoGene to scale pre-training to over 50 million human cells [Hao et al., 2024], while GeneCompass [Yang et al., 2024] introduces a knowledge-informed cross-species framework trained on over 120 million transcriptomes [Yang et al., 2024]. Similarly, Universal Cell Embeddings (UCE) [Rosen et al., 2023] utilizes protein language models to construct a unified latent space for cells that generalizes across tissues and species.

A remaining challenge is that many current pre-training objectives are closely tied to gene-level reconstruction in the observed measurement space—predicting masked counts, masked gene identities, or related entry-wise targets. Because scRNA-seq measurements are sparse and strongly affected by technical variation (e.g., capture efficiency and sampling noise), optimizing such objectives can encourage representations that preserve nuisance variation in addition to biology [Lähnemann et al., 2020]. This motivates complementary objectives that enforce invariances directly in representation space. In contrast, our approach learns single-cell representations through latent-space prediction, aiming to reduce sensitivity to superficial measurement noise while better capturing stable, high-level cellular programs.

**Joint Embedding Predictive Architecture.** Joint Embedding Architectures (JEA) provide a unifying framework for self-supervised representation learning methods. In JEAs, the model encodes and projects multiple views of the same data point into a shared latent space, and their training objectives encourage consistency across views. A central challenge in this setting is representation collapse, where the encoder

21

<!-- page 22 of 26 -->

converges to producing identical representations regardless of the input. Joint Embedding Predictive Architectures (**JEPA**) [Lecun, 2022] address limitations of direct alignment objectives by shifting the learning signal from representation matching to latent-space prediction. Instead of enforcing similarity between embeddings of different views, JEPA trains a predictor to map the representation from a context view to the representation of a target view. In common instantiations, a separate target encoder produces the target representation and is updated using an exponential moving average of the context encoder weights, while the prediction loss is computed entirely in representation space rather than input space. This predictive formulation eliminates the need for negative samples and avoids explicit reconstruction of observed tokens or values, which can otherwise bias learning toward low-level details. JEPA has shown success across multiple domains, including I-JEPA [Assran et al., 2023] for images, V-JEPA [Assran et al., 2025] for videos, and LLM-JEPA [Huang et al., 2025] for language modeling. It demonstrates strong performance and improved scalability. By focusing on predicting embeddings rather than reconstructing raw measurements, JEPA promotes learning meaningful and compositional features.

In our setting, gene expression values reflect complex regulatory programs and cell-state–specific signals. Effective modeling requires representations to capture both gene-level variation and cell-level semantics. GeneJEPA [Litman et al., 2025] is a recent JEPA-style foundation model for transcriptomics that applies latent-space prediction to masked gene subsets, supporting strong transfer across downstream tasks. While conceptually aligned with our goal of moving beyond pure reconstruction-based pre-training, our approach differs in a key way: Cell-JEPA combines a JEPA representation-level objective with an explicit gene-level reconstruction loss. We find that retaining a reconstruction anchor helps preserve fine-grained geneexpression information and stabilizes optimization, while JEPA-style prediction encourages invariances and higher-level structure in the learned representations. This hybrid objective enables Cell-JEPA to learn cell embeddings that are both robust to measurement noise and informative for downstream cellular prediction tasks.

## C scGPT

Unlike language, gene expression vectors are inherently unordered (genes do not form a natural sequence), so scGPT [Cui et al., 2024] introduces a specialized attention mask and iterative generation scheme that is autoregressive over sets of genes, rather than over a fixed gene order.

## C.1 Tokenization and Input Representation

Each cell is represented as a set of gene tokens with associated expression values. The transformer input typically consists of: (i) a dedicated &lt;cls&gt; token that yields a global cell embedding, (ii) a subset of known genes with observed expression values, and (iii) a subset of unknown genes whose expression values are masked and must be predicted. Gene identities are embedded using a learnable gene embedding table, while expression values are embedded through a value embedding module (and optionally additional condition tokens such as batch or perturbation labels).

## C.2 Autoregressive Gene Generation via Masked Attention

To enable generation without imposing an arbitrary gene ordering, scGPT uses a structured attention mask that enforces a known → unknown dependency. When predicting an unknown gene, its token is allowed to attend to all known genes (and itself), but it cannot attend to other unknown genes. Let $\mathcal { U } _ { \mathrm { u n k } }$ denote indices

22

<!-- page 23 of 26 -->

of unknown genes in the current step. The attention mask is defined as:

$$
a _ {i, j} = \left\{ \begin{array}{l l} 0, & \text {if} j \notin \mathcal {U} _ {\mathrm{unk}}, \\ 0, & \text {if} i = j \text {and} j \in \mathcal {U} _ {\mathrm{unk}}, \\ - \infty , & \text {if} i \neq j \text {and} j \in \mathcal {U} _ {\mathrm{unk}}. \end{array} \right.\tag{C.1}
$$

This prevents information leakage between simultaneously masked targets while allowing each unknown gene to condition on the full known context.

## C.3 Pretraining Objective

Given final hidden states $\{ h _ { j } \}$ from the transformer, scGPT predicts expression values of unknown genes via an MLP head and optimizes mean squared error over masked positions:

$$
\mathcal {L} _ {\mathrm{GEP}} = \frac {1}{| \mathcal {U} _ {\mathrm{unk}} |} \sum_ {j \in \mathcal {U} _ {\mathrm{unk}}} (\mathrm{MLP} (h _ {j}) - v _ {j}) ^ {2},
$$

where $v _ { j }$ is the ground-truth expression of gene $j .$

In addition to gene-prompt prediction (predict masked genes from observed genes), scGPT also describes a cell-prompt generation mode where the global cell embedding (from &lt;cls&gt;) is used as a prompt to generate remaining gene expression values [Cui et al., 2024]. In practice, these modes can be applied sequentially and their losses combined during pretraining.

## C.4 Iterative Autoregressive Inference

At inference time, scGPT performs generation in K iterative rounds, yielding an autoregressive procedure over gene subsets rather than a fixed gene order. At each iteration $t ,$ the model computes predictions for the remaining unknown genes using the mask in (C.1). It then selects a subset of the most confident predictions and promotes them to the known set for the next iteration. Concretely, scGPT uses a confidence-based expansion strategy where approximately a $1 / K$ fraction of genes with highest confidence are added each round until all genes are generated [Cui et al., 2024]. This creates an autoregressive refinement process: high-confidence genes are generated earlier and subsequently serve as context for harder-to-predict genes in later rounds.

## C.5 Issues with the ECS Loss in scGPT

The original scGPT framework [Cui et al., 2024] adopts an alternative formulation of the ECS loss:

$$
\mathrm{ECS} _ {\mathrm{scGPT}} = \left(\cos \left(\mathbf {e} ^ {i}, \mathbf {e} ^ {j}\right) - \beta\right) ^ {2},
$$

where $\mathbf { e } ^ { i }$ and $\mathbf { e } ^ { j }$ denote the cell embeddings of cells i and $j ,$ respectively, and $\beta$ is a fixed similarity threshold set to 0.3. While this formulation explicitly enforces pairwise cosine similarities toward a predefined target, it can inadvertently distort biologically meaningful relationships by artificially pushing distinct cellular states closer together or farther apart based solely on a manually chosen hyperparameter. In contrast, our method avoids reliance on such fixed thresholds, leading to a more robust preservation of intrinsic biological structure in the learned representation space.

23

<!-- page 24 of 26 -->

## D Evaluation Metrics

We provide the detailed evaluation metrics calculation methods here.

## D.1 Average Biological Conservation (AvgBIO)

We adopt the AvgBIO score from [Cui et al., 2024] to evaluate the quality of single-cell embeddings. The AvgBIO score is the average of three cell-type clustering metrics developed by [Luecken et al., 2022], including normalized mutual information $( \mathrm { N M I _ { c e l l } } )$ , adjusted rand index $( \mathrm { A R I _ { c e l l } } )$ , and average silhouette width $( \mathrm { A S W _ { c e l l } } )$ . In particular, the AvgBIO score is defined as follows:

$$
\text {AvgBIO} = \frac {\mathrm{NMI} _ {\text {cell}} + \mathrm{ARI} _ {\text {cell}} + \mathrm{ASW} _ {\text {cell}}}{3}
$$

**Normalized Mutual Information** $\mathrm { N M I _ { c e l l } }$ To quantify the agreement between ground-truth cell type annotations and clustering results derived from integrated cell embeddings, we compute the normalized mutual information (NMI) score. Louvain clustering is performed over a range of resolutions from 0.1 to 2.0 in increments of 0.1, and the maximum NMI score across resolutions is reported. The resulting metric, referred to as $\mathrm { N M I _ { c e l l } }$ , measures the shared information between the predicted cluster assignments and true cell type labels. $\mathrm { N M I _ { c e l l } }$ ranges from 0 to 1, with higher values indicating stronger correspondence between clusters and annotated cell types.

**Adjusted Rand Index** $\mathrm { A R I _ { c e l l } }$ . In addition to NMI, we employ the adjusted rand index (ARI) to assess the consistency between annotated cell type labels and the Louvain clusters optimized for NMI. Unlike the standard rand index, ARI corrects for agreements that occur by chance, providing a more robust measure of clustering accuracy. The ARI score for cell types, denoted as $\mathrm { A R I _ { c e l l } }$ , ranges from 0 to 1, where 0 corresponds to random label agreement and 1 indicates a perfect match between predicted clusters and ground-truth annotations.

Average Silhouette Width $\mathrm { A S W _ { c e l l } } .$ To evaluate the quality of cell-type separation in the integrated embedding space, we compute the average silhouette width (ASW). The silhouette width of each cell measures the relative similarity between its within-cluster distance and the distance to the nearest neighboring cluster, thereby capturing both cluster compactness and separation. By averaging silhouette widths across all cells, we obtain the ASW score for cell types, denoted as $\mathrm { A S W _ { c e l l } }$ . This score ranges from −1 to 1, where higher values indicate well-separated and cohesive clusters, while values near or below zero suggest overlapping clusters or potential misclassification.

## D.2 Evaluation Metrics for Perturbation Prediction

Following GEARS [Roohani et al., 2023], we evaluate perturbation-response prediction by comparing predicted and observed post-perturbation gene expression profiles using Pearson correlation. For each perturbation $p$ in the evaluation set, let $\boldsymbol { x } ^ { ( p ) } \in \mathbb { R } ^ { G }$ denote the observed post-perturbation gene expression vector over $G$ genes. And let $\hat { x } ^ { ( p ) } \in \mathbb { R } ^ { G }$ denote the corresponding model prediction. The GEARS evaluation suite reports several Pearson correlation metrics computed on different gene subsets. Pearson correlation is computed over: all genes (global state matches), differentially expressed (DE) genes (perturbation-specific effects), and the top 20 DE genes with dropout-prone genes removed.

24

<!-- page 25 of 26 -->

For any two vectors $a , b \in \mathbb { R } ^ { d }$ , Pearson correlation is:

$$
\rho (a, b) = \frac {\sum_ {i = 1} ^ {d} (a _ {i} - \bar {a}) (b _ {i} - \bar {b})}{\sqrt {\sum_ {i = 1} ^ {d} (a _ {i} - \bar {a}) ^ {2}} \sqrt {\sum_ {i = 1} ^ {d} (b _ {i} - \bar {b}) ^ {2}}},
$$

where $\bar { a }$ and $\overline { b }$ are the component-wise means. Unless otherwise stated, the reported metrics are aggregated across perturbations in the corresponding GEARS test subset (i.e. unseen\_single) by averaging $\rho ( \hat { x } ^ { ( p ) } , x ^ { ( p ) } )$ over $p .$

**All-gene Correlation** pearson. pearson computes $\rho ( \hat { x } ^ { ( p ) } , x ^ { ( p ) } )$ using all genes:

$$
\text {pearson} = \mathbb {E} _ {p} \big [ \rho (\hat {x} ^ {(p)}, x ^ {(p)}) \big ].
$$

This metric emphasizes agreement with the absolute post-perturbation transcriptional state.

**DE-restricted Correlation** pearson\_de. For each perturbation $p ,$ a set of differentially expressed (DE) genes $\mathcal { D } _ { p }$ was identified by comparing the perturbed and control conditions [Roohani et al., 2023]. pearson\_de computes Pearson correlation restricted to this DE set:

$$
\text {pearson\_de} = \mathbb {E} _ {p} \big [ \rho (\hat {x} _ {\mathcal {D} _ {p}} ^ {(p)}, x _ {\mathcal {D} _ {p}} ^ {(p)}) \big ],
$$

where $x _ { \mathcal { D } _ { p } }$ denotes the subvector of x indexed by $\mathcal { D } _ { p }$ . This metric focuses evaluation on genes that exhibit a transcriptional response to the perturbation.

**Top DE Genes with Non-dropout Filtering** top20\_de\_non\_dropout. GEARS provides a filtered subset of top-ranked DE genes with dropout-prone genes removed. Let $\mathcal { T } _ { p } \subseteq \mathcal { D } _ { p }$ denote the resulting set (typically $| \mathcal { T } _ { p } | \; = \; 2 0 )$ provided by the GEARS preprocessing [Roohani et al., 2023]. top20\_de\_non\_dropout computes correlation on this set:

$$
\text {top20\_de\_non\_dropout} = \mathbb {E} _ {p} \big [ \rho (\hat {x} _ {\mathcal {T} _ {p}} ^ {(p)}, x _ {\mathcal {T} _ {p}} ^ {(p)}) \big ].
$$

**Delta Variants (Change-from-control Correlation).** GEARS also reports delta-based metrics that evaluate predicted changes relative to control. Let $x ^ { \mathrm { ( c t r l ) } }   \in   \mathbb { R } ^ { G }$ denote the control expression reference (as provided by the GEARS pipeline [Roohani et al., 2023]). We define the observed and predicted deltas for perturbation $p$ as:

$$
\Delta x ^ {(p)} = x ^ {(p)} - x ^ {(\mathrm{ctrl})}, \quad \Delta \hat {x} ^ {(p)} = \hat {x} ^ {(p)} - x ^ {(\mathrm{ctrl})}.
$$

The delta metrics compute the same correlations as above, but on $\Delta \hat { x } ^ { ( p ) }$ versus $\Delta x ^ { ( p ) }$

$$
\begin{array}{c} \text {pearson\_delta} = \mathbb {E} _ {p} \big [ \rho (\Delta \hat {x} ^ {(p)}, \Delta x ^ {(p)}) \big ], \\ \text {pearson\_de\_delta} = \mathbb {E} _ {p} \big [ \rho (\Delta \hat {x} _ {\mathcal {D} _ {p}} ^ {(p)}, \Delta x _ {\mathcal {D} _ {p}} ^ {(p)}) \big ], \\ \text {delta\_top20\_de\_non\_dropout} = \mathbb {E} _ {p} \big [ \rho (\Delta \hat {x} _ {\mathcal {T} _ {p}} ^ {(p)}, \Delta x _ {\mathcal {T} _ {p}} ^ {(p)}) \big ]. \end{array}
$$

These metrics emphasize delta (change-from-control) estimation and are sensitive to effect-size accuracy relative to the control baseline.

25

<!-- page 26 of 26 -->

## E Training Recipe

We provide the detailed hyperparameter settings for pre-training and finetuning here.

## E.1 Pretraining Recipe

The pretrained model uses an embedding dimension of 512 and comprises 12 stacked transformer blocks, each with 8 attention heads and a hidden dimension of 512. The embedding alignment MLP consists of two fully connected layers with a GELU activation between them. We optimize the model using AdamW with a mini-batch size of 32, a token-masking ratio of 0.15, and an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . We apply a weight decay of $2 \times 1 0 ^ { - 4 }$ and decay the learning rate by a factor of 0.9 after each epoch. We apply dropout with a probability of 0.2 between each layer in the transformer head and the embedding alignment MLP. Training is conducted on the Kidney dataset for a total of 4 epochs. To account for differences in numerical scale, we assign a weight of 1000 to the JEPA loss term and a weight of 1 to the reconstruction loss term. The full training process takes approximately one day on a single A5000 GPU.

We would like to acknowledge and thank the following authors whose datasets are used in the training of our foundation models and preparation of this paper: [Kretzler et al., 2025, Consortium\* et al., 2022, Wilson et al., 2022, Suo et al., 2022, Young et al., 2021, Lake et al., 2021, Zhang et al., 2021, Muto et al., 2021, Han et al., 2020, Cao et al., 2020, Stewart et al., 2019, Young et al., 2018].

## E.2 PBMC-10K Finetuning Recipe

We finetune the model from a pretrained checkpoint trained on the Kidney dataset, as described in Section E.1. We train the model on the PBMC-10K dataset using a masking ratio of 40%. We inherit the JEPA loss weight from pretraining and set the weights of all other loss terms to 1. We optimize the model using the Adam optimizer with a mini-batch size of 64 and an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , decaying the learning rate by a factor of 0.9 after each epoch. We train the model for a total of 30 epochs, and the finetuning process takes approximately 2 hours on a single NVIDIA RTX A5000 GPU.

## E.3 Perturbation Finetuning Recipe

We finetune the model for perturbation prediction using the Adamson and Norman datasets. The model is initialized from a checkpoint pretrained on the Kidney dataset. For the JEPA-enhanced model, we initialize the projector and predictor heads with random weights (or inferred from a checkpoint if available) and train them alongside the student encoder. We optimize using Adam with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 64. The learning rate is decayed by a factor of 0.9 every epoch. Unlike the pretraining phase, we do not use masking on the input genes for perturbation prediction (mask ratio set to 0.0). For the JEPA configuration, the student model is trained to predict the representations of a frozen teacher model (a copy of the student). The JEPA loss weight is set to 1.0 (as to prevent representation drift due to out-of-distribution cell lines), and the reconstruction loss weight is set to 1.0. The contrastive ECS loss is activated and weighted at 0.8. Training is conducted for 15 epochs, which takes approximately 2 hours on a single NVIDIA RTX A100 GPU. The model performance is evaluated using the GEARS evaluation suite (Section D.2) on a held-out test set.

26