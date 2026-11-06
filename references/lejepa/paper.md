<!-- page 1 of 50 -->

arXiv:2511.08544v3 [cs.LG] 14 Nov 2025

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

Randall Balestriero<sup>1,2</sup>,\* Yann LeCun<sup>3,2</sup>,\*

1 Brown University 3 New York University (NYU) 2 Meta-FAIR \* Equal contribution

Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in **LeJEPA**, a lean, scalable, and theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution that JEPAs’ embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel objective–**Sketched Isotropic Gaussian Regularization** (SIGReg)–to constrain embeddings to reach that ideal distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and practical benefits: (i) single trade-off hyperparameter, (ii) linear time and memory complexity, (iii) stability across hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop-gradient, no teacher–student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring only ≈50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will reestablish self-supervised pre-training as a core pillar of AI research ([GitHub repo](https://github.com/rbalestr-lab/lejepa)).

![Chart block](images/page001-block06.png)

ViT-g/14, ImageNet-1K, LeJEPA

![Chart block](images/page001-block08.png)

![Image block](images/page001-block09.png)

![Image block](images/page001-block10.png)

<table><tr><td rowspan="2">Method</td><td colspan="2">Full FT</td><td colspan="2">Frozen</td></tr><tr><td>1-sh</td><td>Full</td><td>1-sh</td><td>Full</td></tr><tr><td colspan="5">LeJEPA (in-domain)</td></tr><tr><td>ConvNeXt-V2 Nano</td><td>29.42</td><td>82.72</td><td>28.74</td><td>76.52</td></tr><tr><td>ResNet-34</td><td>24.27</td><td>83.28</td><td>31.08</td><td>78.17</td></tr><tr><td colspan="5">Frontier (transfer)</td></tr><tr><td>DINOv2 ViT-S/16</td><td>21.05</td><td>78.34</td><td>27.68</td><td>67.62</td></tr><tr><td>DINOv3 ViT-S/16</td><td>24.71</td><td>81.60</td><td>30.17</td><td>71.38</td></tr></table>

Figure 1. LeJEPA overview. Top-left: Training loss exhibits strong correlation with downstream linear probe performance on ImageNet-1k (ViT-base), providing the first practical loss for model selection without supervised probing. Top-right: Training stability without heuristics even on 1.8B ViT-g models, stable training loss. Bottom-left: PCA features from ImageNet-1k pretrained LeJEPA ViT-Large demonstrate clear semantic relationships. Bottom-right: Galaxy10 in-domain results showcasing LeJEPA’s in-domain pretraining consistently outperforms state-of-the-art frontier foundation models transfer learning (DINOv2/v3 trained on natural images) across data regimes from 1-shot to full supervision. This demonstrates that domain-specific SSL beats generic transfer learning, even against massive-scale frontier models, when the framework scales effortlessly to any domain, model, and data scale.

<!-- page 2 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

LeJEPA:

## Introduction

Learning manipulable representations of the world and its dynamics is a long-standing question in AI, with roots dating back centuries ago [Von Helmholtz, 1867, Tolman, 1948, Gregory, 1980, Sutton, 1991, Friston, 2010]. Across domains, $\mathbf { e . g . }$ , image recognition, robotics, physics, space exploration, the unifying question is how to learn an organized and actionable high-dimensional embedding space from observations? Using Deep Networks–parameterized non-linear operators $f _ { \theta ^ { - } }$ –to map observations to embeddings is a standard first piece of that puzzle [LeCun et al., 2015, Goodfellow et al., 2016]. The second, less standardized, piece of that puzzle is how to train $f _ { \theta } .$ Joint-Embedding Predictive Architectures (JEPAs) suggest training $f _ { \theta }$ by maximizing predictive agreement between the embeddings of semantically related views [Bromley et al., 1993, LeCun, 2022, Balestriero et al., 2023]. Views can come in two forms: transformations or corruptions. They can involve masking, cropping, blurring, temporal or spatial translations, geometric or photometric transformations, viewpoint changes, views from different sensor modalities, etc. The supervised forms involve human-produced components such as image-caption pairs, text-code pairs, etc [Tian et al., 2020]. In any case, views are expected to share some degree of semantic relationship to allow the prediction task to align $f _ { \theta } { ' s }$ embeddings towards the underlying knowledge present in the data.

Alas, JEPA’s prediction task admits failure modes, such as representation collapse, where $f _ { \theta }$ maps all inputs to nearly identical embeddings (complete collapse) or to a lowdimensional subspace (dimensional collapse) [Jing et al., 2021][Jing et al., 2021, Cosentino et al., 2022, Balestriero and LeCun, 2022]. To mitigate such shortcut solutions, state-of-the-art recipes rely on heuristics–stop-gradient [Chen et al., 2020a], asymmetric view generation [Wang et al., 2022], teacher–student networks with carefully tuned EMA schedules [Caron et al., 2021, Tian et al., 2021], explicit normalization and whitening layers [Ermolov et al., 2021, Chen et al., 2021]–and a delicate balance of hyperparameters. As a result, today’s JEPA training is brittle and most research has shifted toward scaling data [Vo et al., 2024], models [Fan et al., 2025] and even post-training Rodas et al. [2025] while leaving the theoretical foundations of JEPAs largely unexplored.

Our study proposes to break that cycle by questioning some of the fundamental design principles underpinning JEPAs. That introspection will start by asking what are the necessary conditions that JEPAs should abide by? Those minimal conditions will then act as axioms for us to design a novel and lean JEPA. We identify two axioms: (i) solving the prediction task while (ii) enforcing an isotropic Gaussian distribution of the embeddings

(Section 3). While (i) follows standard practice [Balestriero and LeCun, 2022], we introduce in Section 4 a novel distribution matching objective–Sketched Isotropic Gaussian Regularization (SIGReg)–to enforce (ii). The use of SIGReg not only removes the need for the numerous heuristics previously employed to prevent representation collapse, but SIGReg also exhibits favorable scaling properties as its memory and computational complexity is linear in dimension and sample size. Crucially, SIGReg’s isotropic Gaussian enforcement solves the collapsed shortcut solution and provably minimizes the model’s expected risk over the space of downstream tasks to be encountered post-training. The resulting JEPA solution–coined Latent-Euclidean JEPA (LeJEPA)–is introduced in Section 5. Beyond theoretical optimality, LeJEPA offers numerous benefits such as (i) provable statistical guarantees, (ii) removal of heuristics such as teacher-student networks, (iii) linear memory and computational complexity, and most importantly (iv) a unified design with a single trade-off parameter that works out of the box across datasets, architectures and scales (see Section 6). We summarize our contributions below.

**Contribution 1: We prove the optimal embedding distribution for foundation models.** We establish that the isotropic Gaussian uniquely minimizes downstream prediction risk across broad task families. In Section 3, we derive this result rigorously for both linear (Section 3.1) and nonlinear probes (Section 3.2), providing the first principled answer to what distribution $f _ { \theta } ^ { \prime }$ s embeddings should follow. This theoretical result transforms JEPA design from heuristic exploration to targeted optimization. **Contribution 2: We introduce SIGReg, a distribution matching objective that uniquely combines provable correctness with computational efficiency at scale.** We present Sketched Isotropic Gaussian Regularization (SIGReg), a novel objective that enforces distributional alignment via random projections and characteristic-function matching (Section 4 and Figure 2). SIGReg provides statistical guarantees (Sections 4.1 and 4.2) while achieving linear complexity and bounded gradients—a combination that existing distribution matching methods do not offer. Critically, its projection-based construction defeats the curse of dimensionality (Section 4.3), making it both theoretically sound and practically efficient for high-dimensional embeddings.

**Contribution 3: We design LeJEPA, a statistically optimal JEPA that eliminates collapse by construction.** By combining JEPA’s predictive objective with SIGReg targeting the isotropic Gaussian, we introduce LeJEPA—Latent-Euclidean JEPA (Section 5). LeJEPA requires only a single hyperparameter, eliminates representational collapse without stop-gradients or teacher-student architectures, and transfers across architectures and datasets without hyperparameter tuning. This demonstrates that principled

2

<!-- page 3 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Image block](images/page003-block03.png)

![Chart block](images/page003-block04.png)

Figure 2. Sketched Isotropic Gaussian Regularization (SIGReg): Given some arbitrary input data with density $p _ { x }$ with support that may or may not lie on a manifold (left), a Deep network (DN) encoder ( 𝑓<sub>𝜽</sub>) produces embeddings $z = f _ { \pmb { \theta } } ( \pmb { x } )$ with some distribution $z \sim p _ { z }$ (middle). Our proposed Backward Cramér-Wold Statistics (Section 4) objective pushes $p _ { z }$ to match a target distribution $p _ { t }$ by projecting the embeddings along 1𝑑 directions (middle, arrows) and enforcing that the univariate densities (right, colored lines) match the distribution of $p _ { t } ,$ projected along the same directions. Any popular statistical test (provided in Section 4.2) can assess the goodness-of-fit–in practice we argue for characteristic function tests (Section 4.2). By using SIGReg with $p _ { t }$ isotropic Gaussian (right, black lines), we introduce a lean and provably optimal (Section 3) JEPA, coined LeJEPA, free of numerous heuristics and able to produce competitive performances (Sections 5 and 6).

theory directly yields practical simplicity.

**Contribution 4: We validate LeJEPA at scale across diverse architectures and establish in-domain pretraining as viable.** Our experiments (Section 6) span ViTs, ConvNeXts, ResNets, MaxViTs, and Swin Transformers at scales approaching 1 billion parameters, where LeJEPA matches or exceeds state-of-the-art methods while maintaining training simplicity and robustness. Critically, on domain-specific datasets (Galaxy10, Food101), LeJEPA outperforms DINOv2-based transfer learning when pre-trained directly on target data. This challenges the transfer learning paradigm and demonstrates that principled SSL can unlock effective in-domain pretraining—previously considered impractical for small datasets.

## 2 Background and Notations

We start by introducing some of the notations we will be using throughout our manuscript (Section 2.1), followed by a review of JEPAs (Section 2.2), and existing literature studying their design (Section 2.3).

## 2.1 Notations and Definitions

**Data.** We are in possession of a dataset of shape $\left( N , V , D \right) \in$ $\mathbb { N } ^ { * 3 }$ where 𝑁 is the number of samples, 𝑉 is the number of views, and 𝐷 is the dimension. One entry of this dataset is accessed via $x _ { n , v , d }$ . Those dimensions are often interpreted as follows: (N) is the number of independent samples, $\mathbf { e . g . }$ , different images or different videos, (V) is the number of views, $\mathbf { e . g . }$ , data-augmentations for images, frames for videos, and (D) is the dimension of each $x _ { n , v } ,$ e.g., number of RGB pixels for images. In many cases the ordering over 𝑉 is given by time–but in some cases, e.g., data-augmentation of an image, ordering becomes

irrelevant. Our study does not require any particular choice to organize one’s dataset into a $( N , V , D )$ tensor– and none of our theory and implementation assumes a particular design decision for that tensor. However, we will rely on the following two properties, (independence) the samples $x _ { n } , x _ { n ^ { \prime } }$ have been obtained independently from each other $\forall n \neq n ^ { \prime }$ , and (identically distributed) the sampling process was identical among $x _ { n } , \forall n$

**Deep Networks.** Today’s AI solutions rely on Deep (Neural) Networks (DNs), which are compositions of a large number of parameterized linear and nonlinear operators. We denote the DN’s mapping as $f _ { \theta } : \mathbb { R } ^ { D } \rightarrow \mathbb { R } ^ { \vec { K } }$ with 𝐾 the dimension of the embedding space. The internals of $f _ { \theta }$ are designed by the researcher to incorporate as much prior knowledge about the data as possible. The details of $f _ { \theta }$ are irrelevant to our study–as we will see the proposed LeJEPA works out-of-the-box on any $f _ { \theta } .$ In any case, all the learnable parameters are gathered in the vector $\pmb { \theta } \in \mathbb { R } ^ { P }$ , with 𝑃 counting the total number of parameters. A central challenge in AI research is to design the right architecture and training objective so that 𝜽 can be learned from gradient descent to ultimately produce a useful system, or foundation model, $f _ { \theta } .$

**JEPAs.** A foundation model is any system, e.g., a DN, able to solve numerous downstream tasks without requiring any change in its internal parameters 𝜽. This is in sharp contrast with a supervised model that only considers its training task. JEPAs have formally been introduced by LeCun [2022] as a vehicle to produce foundation models. The core building blocks of JEPAs rely on numerous wellestablished techniques such as siamese networks [Bromley et $\mathrm { a l . } ,$ 1993] and predictive coding [Helmholtz et al., 1867, Bruner and Postman, 1949]. While the exact blueprint of

3

<!-- page 4 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

## Definition 1: JEPA

$\mathrm { J E P A } ( \pmb { x } ) \iff \mathrm { E n c } \left( \pmb { x } _ { n , t + 1 , . } \right)$ is predictable from Enc $\left( x _ { n , t , . } \right) , \forall n , t$ and Enc $\left( x _ { . . . , . . } \right)$ is not degenerate.

![Image block](images/page004-block05.png)

JEPAs varies greatly between use-cases, they all rely on two core principles: (i) being able to predict the embedding of a view $x _ { n , v }$ from the embedding of another view $x _ { n , v ^ { \prime } } , v ^ { \prime } \neq v ,$ all while (ii) ensuring that the embeddings do not become degenerate. Concretely, once a JEPA is designed and trained, it should be able to solve numerous downstream tasks in zero or few shots. The JEPA objective function, along with some examples for $x ,$ i s provided in Equation (1). The predictability criterion can be done by directly comparing the embeddings of the partial views $E n c ( \pmb { x } _ { n , v , . } )$ and $E n c ( \pmb { x } _ { n , v ^ { \prime } , . } )$ with a metric, $\mathbf { e . g . } , \ell _ { p }$ . In some cases, an additional DN coined Pred, is employed to compare $P r e d ( E n c ( \pmb { x } _ { n , v , . } ) )$ against $E n c ( \pmb { x } _ { n , v ^ { \prime } , \cdot } )$ )–which is only justified when there exists an asymmetry between the information content of the different views, e.g., by conditioning the predictions on observed actions from robotics data [Khazatsky et al., 2024].

## 2.2 The Need for Reliable Pretraining

The JEPA’s prediction task is designed based on a priori knowledge of the data. Its design is often quite natural since it is relatively intuitive to form 𝒙 so that its views share the relevant information content one hope to capture. On the other hand, the design of the “anti-collapse” criterion is much closer to a game of Whac-A-Mole. Today’s designs rely on many different under-specified safeguards which are carefully combined in the hope that degenerate shortcut solutions are avoided during training. Such mechanisms include (i) feature whitening [Ermolov et al., 2021, Bardes et al., 2021], (ii) negative samples [Chen et al., 2020a, He et al., 2020], and (iii) asymmetric views and teacher-student networks with stop-gradient [Caron et al., 2021, Assran et al., 2023]. Those mechanisms all suffer from at least two of the following limitations: (i)

under-specification, $\mathbf { i . e . } ,$ the criteria can be minimized while embeddings are in a degenerate configuration, (ii) quadratic time and memory complexity with mini-batch size and/or embedding dimension, (iii) sensitivity to data distribution, hyperparameters, architecture, and (iv) lack of theoretical understanding and guarantees.

## 2.3 The Need for Actionable Theory

For decades, the two major solutions for AI were supervised learning [LeCun et al., 2015] and learning by reconstruction [Rumelhart et al., 1986]–sometimes combined together, e.g., for semi-supervised learning [Kingma et al., 2014]. In supervised learning, the labels both ensure that semantically similar samples are close to each other in embedding space while preventing complete representation collapse. In particular, it is possible to measure the amount of collapse in supervised learning as a function of the number of classes [Papyan et al., 2020]. The reconstruction objective is similarly well suited to prevent representation collapse as the original input must be recovered from the embeddings, i.e., the embeddings must be as informative about the input as possible–up to some optional denoising tasks that users can setup as part of the training [Vincent et al., 2010].

Because supervised and reconstruction-based learning have been widely studied for decades, there exists a large body of work to explain and inform practical designs–as well as studying their limitations in producing foundation models [Balestriero and LeCun, 2024, Van Assel et al., 2025]. This is not the case for the more recent JEPAs where empirical advances quickly outpace anyone hoping to delve into their inner workings. This dynamic led the community to focus on post-hoc theoretical justification of already found solutions [Liu et al., 2021, Shwartz Ziv

4

<!-- page 5 of 50 -->

LeJEPA: Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

and LeCun, 2024, Shwartz-Ziv et al., 2022, Zhang et al., 2023]. In most cases, those studies involve the Mutual Information (MI) [Shannon, 1948, Cover, 1999] whose different bounds recover established methods [Gutmann and Hyvärinen, 2010, Ma and Collins, 2018, Oord et al., 2018, Poole et al., 2019, Hjelm et al., 2018, McAllester and Stratos, 2020]. Because existing studies focus on explaining and interpreting already developed $\mathrm { J E P A s } ,$ too little principled guidance and innovation has been brought forward. Instead, most of the recent empirical advances take the form of collecting larger dataset, scaling up pre-existing training recipes [Goyal et al., 2019, Chen et al., 2020b, Oquab et $\mathrm { a l . } ,$ 2023, Fan et al., 2025], and deriving novel data curation processes [Vo et al., 2024, Kerdreux et al., 2025].

In contrast, our goal in the following Sections 3 to 5 will be to derive a novel JEPA solution from first principles, i.e., whose design relies on proved necessary conditions for optimality, and with a pretraining recipe that can finally reconcile exploratory research, scalability, and state-of-the-art performances.

## 3 Latent Euclidean: Embeddings Should be Isotropic Gaussian

We address a fundamental question: which distribution should Enc(𝒙) follow to minimize empirical risk on any downstream task? We prove that the isotropic Gaussian is the unique optimal distribution for both linear (Section 3.1) and nonlinear probing (Section 3.2), with geometric intuition provided in Section 3.3. This theoretical result establishes the necessary design principle for our JEPA; Section 4 then provides the practical implementation to achieve it.

## 3.1 Linear Probing

We begin by identifying the optimal distribution for $f _ { \theta } { ' s }$ embeddings by analyzing linear probes–one of the most popular methods for frozen encoder evaluation. Specifically, we ask: which distribution for $f _ { \theta } ( x )$ would be most favorable for solving arbitrary downstream tasks, $i . e . , f _ { } { o r }$ any realization of targets $y ^ { 2 }$

Denote as $\stackrel { \circ } { \boldsymbol { Z } \in \mathbb { R } ^ { \check { N } \times K } }$ the matrix of 𝑁 embeddings, each 𝐾-dimensional, from $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { n } )$ . The unknown corresponding labels are denoted as $\dot { \boldsymbol { y } } \in \mathbb { R } ^ { N }$ . Without loss of generality, we consider univariate targets; the following analysis extends to multivariate targets. The linear probe minimizes the following least square problem [Bishop and Nasrabadi, 2006]

$$
\hat{\beta} = \operatorname *{arg  min}_{\substack{\boldsymbol {\beta}\in \mathbb{R}^{K}}}\| \boldsymbol {y} - \boldsymbol {Z}\boldsymbol {\beta}\|_{2}^{2} + \lambda \| \boldsymbol {\beta}\|_{2}^{2},\tag{OLS}
$$

where $\hat { \beta }$ is the optimal probe parameters, and $\lambda \geq 0$ is an hyperparameter controlling the Tikhonov regularizer strength [Bishop, 1995, Golub et al., 1999]. Despite

not knowing $y ,$ it is possible to describe the bias and variance of the estimator $\hat { \beta }$ as a function of the distribution of $Z .$ Consider two embeddings with identical column spans $Z _ { \mathrm { a n i s o } } , Z _ { \mathrm { i s o } } . Z _ { \mathrm { a n i s o } } ,$ s covariance matrix eigenvalues are given by $\{ \lambda _ { k } \} _ { k = 1 } ^ { K }$ with at least two distinct values, while $\mathbf { Z _ { \mathrm { i s o } } } ^ { \prime } \mathbf { s }$ covariance matrix eigenvalues are all equal to $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lambda _ { k } } \end{array}$ . Hence, the two candidate embeddings $Z _ { \mathrm { a n i s o } } , Z _ { \mathrm { i s o } } ^ { \mathrm { ~   ~ } }$ capture the same intrinsic features and have same energy, but different geometries.

## Lemma 1: Anisotropy amplifies bias

Whenever $\lambda _ { K } > \lambda _ { 1 } ,$ there always exists a downstream task $( y )$ for which $Z _ { \mathrm { a n i s o } }$ produces a higher bias estimator than $Z _ { \mathrm { i s o } }$ for $\lambda > 0 .$ (Proof in Section B.1.)

**Lemma 2: Anisotropy amplifies variance** With $\lambda = 0 ,$ , the total variance of $\mid \hat { \beta }$ (OLS) is minimized for $Z _ { \mathrm { i s o } }$ with $\mathrm{tr}(\mathrm{Var}(\hat{\boldsymbol{\beta}}_{\mathrm{aniso}})) > \mathrm{tr}(\mathrm{Var}(\hat{\boldsymbol{\beta}}_{\mathrm{iso}}))$ . (Proof in Section B.2.)

From the above lemmas. 1 and 2 we obtain that the distribution of features must be isotropic. We now move to nonlinear probing where the standard Gaussian will emerge as the unique optimum.

## 3.2 Nonlinear Probing

To allow for more flexible evaluation of the pretrained encoder $f _ { \theta } ,$ it has become increasingly common to work with a nonlinear probe. We analyze two widely-used nonlinear methods: radius-based k-NN [Taunk et al., 2019, Sun and Huang, 2010, Zhang et al., 2017, Abu Alfeilat et al., 2019] for its simplicity and kernel methods [Nadaraya, 1964, Watson, 1964] for their theoretical tractability.

As in Section 3.1, we ask ourselves which distribution of embeddings would be preferable for a foundation model. We first define our prediction function. The training data consists of the 𝑁 embeddings along with their training labels $\{ ( z _ { n } , \pmb { y } _ { n } ) \} _ { n = 1 } ^ { N } .$ . The prediction, using radius-based k-NN for a query vector $q$ is formed as

$$
\left| \widehat {y} (q) := \frac {1}{| \mathcal {N} _ {r _ {0}} (q) |} \sum_ {n \in \mathcal {N} _ {r _ {0}} (q)} y _ {n}, \right|\tag{kNN}
$$

where $\mathcal { N } _ { r _ { 0 } } ( \boldsymbol { q } ) = \{ n : \| \boldsymbol { z } _ { n } - \boldsymbol { q } \| \leq r _ { 0 } \}$ . The specific choice of radius $r _ { 0 }$ controls how many neighbors predictions are averaged to form the query’s prediction. The kernel’s prediction at a query $\stackrel { \circ } { \pmb { q } \in \mathbb { R } ^ { \tilde { K } } }$ is given by

$$
\widehat {\boldsymbol {y}} (\boldsymbol {q}) \triangleq \frac {\sum_ {n = 1} ^ {N} K _ {h} (\boldsymbol {q} - \boldsymbol {z} _ {n}) \boldsymbol {y} _ {n}}{\sum_ {n = 1} ^ {N} K _ {h} (\boldsymbol {q} - \boldsymbol {z} _ {n})}.\tag{Kernel}
$$

We search over all distributions of $Z$ subject to a fixed total variance constraint, e.g., $\mathrm{Tr}(\mathrm{Cov}(\boldsymbol{Z})) = \kappa_1 \mathrm{or} \|\mathrm{Cov}(\boldsymbol{Z})\|_F =$ $\kappa _ { 2 }$ . The specific value of 𝜅 does not affect the optimal dis-

5

<!-- page 6 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Chart block](images/page006-block03.png)

Figure 3. Illustration of lemma. 2 showcasing how anisotropic (right) embeddings lead to higher variance estimator compared to isotropic embeddings (left). We sample 100 training points for the 2-class classification task and fit a logistic regression–repeating the process over numerous training set sample. Each sampling results in a decision boundary (purple).

tribution shape. Following the same type of derivations as done in the linear regime–with the exception of some additional regularity conditions–we are able to precisely identify the isotropic Gaussian as the unique optimum to minimize bias as formalized below.

## Theorem 1: isotropic Gaussian Optimality

The integrated square bias (ISB) over query points is given by

$$
\mathrm{ISB} _ {k \text {-NN}} = \frac {r _ {0} ^ {4}}{(K + 2) ^ {2}} \tau_ {g} ^ {2} J (p) + O (r _ {0} ^ {4}),\tag{k-NN}
$$

$$
\mathrm{ISB} _ {\text {kernel}} \leq \left(\frac {h ^ {2} \mu_ {2} (K)}{2}\right) ^ {2} \left(2 B ^ {2} + 8 L ^ {2} J (p)\right) + o \left(h ^ {4}\right),\tag{kernel}
$$

and among distributions with a scalar-based covariance constraint, the isotropic Gaussian is the unique minimizer of the integrated square bias. (Proof in Sections B.4 and B.7.)

Numerous additional details and discussions on the regularity assumptions we employed are provided in Section A. Together, these results establish the isotropic Gaus sian distribution as the optimal design to minimize the worst-case risk of a foundation model across downstream tasks.

## 3.3 Geometric and Practical Insights

We now empirically validate that the isotropic Gaussian is optimal when no information about downstream tasks is available. We focus on linear probing (Section 3.1), where all considered distributions have the same total variance.

When employing a linear probe, an anisotropic distribution increases both bias (with Tikhonov regularization) and variance. Examining bias first (lemma. 1), we present in Figure 18 visualizations for both continuous regression and discrete classification tasks. We observe that the cosine similarity between estimated and ground-truth

parameters equals 1 only for isotropic distributions, degrading for anisotropic cases regardless of sample size or regularization strength. Regarding variance (lemma. 2), we show in Figure 3 that learned parameters vary significantly more across training sets when the covariance is anisotropic (right) compared to isotropic (left)—even when using logistic regression instead of OLS. Figure 17 further illustrates this effect, showing the distribution of learned $\beta$ parameters across different training samples for both cases. The anisotropic distribution clearly produces higher-variance estimators.

These theoretical and empirical results establish our design principle for LeJEPA: embeddings 𝑓<sub>𝜽</sub>(𝒙) should follow an isotropic Gaussian distribution to minimize worst-case risk across downstream tasks encountered post-training. Section 4 introduces a novel regularizer to achieve this distribution.

## 4 SIGReg: Reliable Isotropic Gaussian Regularization in High-Dimension

Having established the isotropic Gaussian as the optimal embedding distribution (Section 3), we now introduce Sketched Isotropic Gaussian Regularization (SIGReg)–a distribution matching objective that is simultaneously (i) differentiable, (ii) scalable, (iii) provable, and (iv) interpretable. SIGReg builds on three key innovations. First, we formulate distribution matching as a statistical test under the null hypothesis $P _ { \boldsymbol { \theta } } = Q$ (Section 4.1). Second, we identify a test that guarantees bounded gradients and curvature while maintaining linear complexity and efficient multi-GPU scaling (Section 4.2). Third, SIGReg bypasses the curse of dimensionality, eliminating collapsed shortcut solutions entirely (Section 4.3).

## 4.1 Hypothesis Testing as a Judge

Asking for $f _ { \theta } ( x ) ^ { \prime }$ s distribution $P _ { \theta }$ to match a target distribution 𝑄 is typically done by creating various measures of distance or divergence, and estimating them in highdimension. We propose a different starting point grounded in statistics. Consider the hypothesis testing framework [Fisher, 1928, Neyman and Pearson, 1933] given by

$$
H _ {0}: P _ {\theta} = Q \quad \text {vs.} \quad H _ {1}: P _ {\theta} \neq Q,\tag{2}
$$

with $H _ { 0 }$ being referred to as the null hypothesis. That is, we are asking in Equation (2) if there is enough empirical evidence to reject the null. To answer that question, one $( \mathrm { i } )$ employs a test-statistic, i.e., a single scalar value summarizing the evidence from the empirical samples, (ii) determines a critical value $\tau _ { \alpha }$ for the test-statistic based on the probability 𝛼 of Type I error, i.e., of mistakenly rejecting a true null hypothesis, (iii) compares the test-statistic to

6

<!-- page 7 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

the critical value $\tau _ { \alpha } ;$ if the test-statistic exceeds $\tau _ { \alpha } ,$ reject the null hypothesis. If the null is not rejected, we can only claim that there is not sufficient empirical evidence against $P _ { \boldsymbol { \theta } } = Q$

As it stands, Equation (2) remains impractical in large dimension as existing tests have at least quadratic complexity with the number of samples considered (more details in Section F). We thus propose to derive a sketching strategy by decomposing Equation (2) into simpler univariate tests. Denoting the push-forward distributions $P _ { \boldsymbol { \theta } } ^ { ( a ) } \triangleq ( \boldsymbol { a } ^ { \top } ) _ { \# } P _ { \boldsymbol { \theta } }$ and $Q ^ { ( a ) } \triangleq ( a ^ { \top } ) _ { \# } Q$ , we can define the following directional univariate test

$$
H _ {0} (\boldsymbol {a}): P _ {\boldsymbol {\theta}} ^ {(a)} = Q ^ {(a)} \text {vs.} H _ {1} (\boldsymbol {a}): P _ {\boldsymbol {\theta}} ^ {(a)} \neq Q ^ {(a)},\tag{3}
$$

for a given directional unit-norm vector $\boldsymbol { a } \in \mathcal { S } ^ { K - 1 }$ . The corresponding directional test-statistic of Equation (3) is computed as $\stackrel { \sim } { T } ( \{ \boldsymbol { a } ^ { \top } f _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { n } ) \} _ { n = 1 } ^ { N } )$ . Examples of tests 𝑇 will be provided in the later Section 4.2. Repeating that process over a set of 𝑀 directions $\mathbb{A} \triangleq \{ \boldsymbol{a}_{1}, \cdots, \boldsymbol{a}_{M} \}$ and aggregating the individual values lead to the following global test-statistic

$$
T _ {\mathbb {A}} (\{f _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {n}) \} _ {n = 1} ^ {N}) \triangleq \max _ {\boldsymbol {a} \in \mathbb {A}} T (\{\boldsymbol {a} ^ {\top} f _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {n}) \} _ {n = 1} ^ {N}).\tag{4}
$$

We now provide a formal statement asserting the consistency of Equation (4) to test the original multivariate null hypothesis from Equation (2). Our result leverages the well-known union-intersection principle [Roy, 1953], and a slightly modified Cramér-Wold theorem. We denote by 𝑑 = equality in distribution.

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Lemma 3: Hyperspherical Cramér-Wold
Let $X, Y$ be $\mathbb{R}^{d}$-valued random vectors, then
$\langle u, X \rangle \stackrel{d}{=} \langle u, Y \rangle, \forall u \in \mathbb{S}^{d-1} \iff X \stackrel{d}{=} Y.$
Convergence in distribution also holds. (Proof in Section B.8.)
</div>

## Theorem 2: Sufficiency of directional tests

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Equation (4) is a valid statistical test for Equation (3) as
$P = Q \implies \operatorname* {l i m s u p} _{n \to \infty} \Pr \left( T_{\mathbb{A}} (\{f_{\boldsymbol{\theta}}(\boldsymbol{x}_{n})\}_{n=1}^{N}) \geq \tau_{\alpha} \right) \leq \alpha, (\textbf{level})$
$P \neq Q \implies \operatorname* {l i m s u p} _{n \to \infty} \Pr \left( T_{\mathbb{A}} (\{f_{\boldsymbol{\theta}}(\boldsymbol{x}_{n})\}_{n=1}^{N}) \geq \tau_{\alpha} \right) = 1, (\textbf{power})$
(Proof in Section B.9.)
</div>

The assumptions required in the proof of thm. 2 hold for classical consistent univariate tests 𝑇 such as the ones presented in the following Section 4.2.

![Image block](images/page007-block13.png)

Figure 4. Examples of distributions living on the surface of the sphere with varying Sobolev smoothness coefficients 𝛼. As per thm. 5, the greater 𝛼 is, the more global will be the impact of SIGReg for a given number of directions 𝑀. Practically, this represents the distribution of the encoder’s output. Because the target density (isotropic Gaussian) is smooth, the 𝛼 coeffcients of the embedding will quickly grow hereby making SIGReg (def. 2) immune to the curse of dimensionality.

## 4.2 SIGReg: Sketching the Epps-Pulley Test is Stable and Scalable

Our proposed regularizer–coined Sketched Isotropic Gaussian Regularization (SIGReg)–follows directly from thm. 2 using any statistical test 𝑇 targeted towards the isotropic Gaussian, illustrated in Figures 2 and 5, and formalized below.

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Definition 2: SIGReg (PyTorch code in algorithm 1)
SIGReg sketches a statistical test $T$ towards isotropic Gaussian
$\mathrm{SIGReg}_T(\mathbb{A}, \{f_\boldsymbol{\theta}(\boldsymbol{x}_n)\}_{n=1}^N) \triangleq \frac{1}{|\mathbb{A}|} \sum_{\boldsymbol{a} \in \mathbb{A}} T(\{\boldsymbol{a}^\top f_\boldsymbol{\theta}(\boldsymbol{x}_n)\}_{n=1}^N),$
(SIGReg)
where we recommend the Epps-Pulley test (Section 4.2.3) for $T$.
</div>

We replace the maximum over $a   \in   \mathbb { A }$ in thm. 2 by an average in (SIGReg) to avoid sparse gradient over the directions in A. We now delve on the choice of $T$ for which we compare well-known candidate tests in the field of statistics that are categorized into (i) moment based (Section 4.2.1), (ii) CDF based (Section 4.2.2), and (iii) CF based (Section 4.2.3) statistics–ultimately justifying our choice of the Epps-Pulley statistic.

## 4.2.1 Moments are Unstable and Insufficient

The first family of statistics we consider are moment-based. Taking the standard Gaussian as an instanciation for the moments, we can define the Jarque-Bera [Jarque and Bera, 1980] test that compares the third and fourth moments,

7

<!-- page 8 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Chart block](images/page008-block03.png)

x<sub>1</sub> (blue) — x<sub>2</sub> (red)

![Chart block](images/page008-block05.png)

![Chart block](images/page008-block06.png)

hx, a<sub>i</sub>i

![Chart block](images/page008-block08.png)

Figure 5. Constructed data density with $\mathsf { { ^ { \prime \prime } X ^ { \prime \prime } } }$ distribution whose marginals are standard Gaussian and whose covariance is identity (left densities). Applying $M = 1 0$ projections on the half circle directions produces 10 univariate distributions that can be compared against a standard Gaussian (left) using any preferred statistic from Section 4.2. The appropriate direction is able to capture the degenerate distribution of the data hereby creating a spike in the statistic value.

i.e., skewness and kurtosis, as

$$
\mathrm{JB} (\boldsymbol {u}) \triangleq \frac {N}{6} \left(\widehat {\operatorname{skew}} (\boldsymbol {u}) ^ {2} + \left(\frac {\widehat {\operatorname{kurt}} (\boldsymbol {u}) - 3}{2}\right) ^ {2}\right), \quad \text {(Jarque - Bera)}
$$

where $\widehat { \mathrm { s k e w } }$ is the skewness computed from the data as $\frac { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl ( x _ { i }   -   \hat { \mu } \bigr ) ^ { 3 } } { \hat { \sigma } ^ { 3 } }$ and kurt d is the kurtosis $\frac { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl ( x _ { i }   -   \hat { \mu } \bigr ) ^ { 4 } } { \hat { \sigma } ^ { 4 } }$ . Typically, the $( \mathrm { f a r q u e - B e r a } )$ test is used to see if a density follows a Gaussian distribution of any mean and variance–hence it only looks at moments 3 and 4. In our case we aim for a standard Gaussian test and thus add the usual statistics on the first two moments, leading to the extended test

$$
\begin{array}{c} \operatorname{EJB} (\boldsymbol {u}) \triangleq \frac {N \hat {\mu} (\boldsymbol {u}) ^ {2}}{\hat {\sigma} (\boldsymbol {u}) ^ {2}} + \frac {(N - 1) (\hat {\sigma} (\boldsymbol {u}) ^ {2} - 1) ^ {2}}{2} + \operatorname{JB} (\boldsymbol {u}). \\ \text {(Extended Jarque - Bera)} \end{array}
$$

The (Extended Jarque-Bera) acts as a moment matching problem over the first four moments. Such moment matching methods have proven powerful not only for statistical tests but also as mean to learn parametric and nonparametric models of data.

**The Stability and Identifiability Conundrum.** We now explain why moment-based tests–albeit powerful–will not be suited for LeJEPA. The $k ^ { t h }$ of a distribution 𝑃 is denoted as $m _ { k } ( P )$ The first observation is that wellbehaved distributions abiding the Carleman’s condition $\begin{array} { r } { \sum _ { k = 1 } ^ { \infty } m _ { 2 k } ( Q ) ^ { - 1 / ( 2 k ) } = \infty } \end{array}$ [Carleman, $1 9 2 6 ] ,$ such as the Gaussian, or for distributions with finite interval [Hausdorff, 1923] are uniquely determined by their moments. However, using a finite number of moments creates the following non-identifiability issue which well-known in statistics and often used as a motivation to use all moments [Lehmann and Romano, 2005].

## Theorem 3: Insufficiency of K Moments

Minimizing the following objective with $c _ { k } > 0 , \forall k$

$$
\sum_ {k = 1} ^ {K} c _ {k} \left(m _ {k} \left(P _ {\pmb {\theta}} ^ {(a)}\right) - m _ {k} \left(Q ^ {(a)}\right)\right) ^ {2},
$$

$$
P _ {\boldsymbol {\theta}} ^ {(a)} = Q ^ {(a)}
$$

Hence thm. 3 prescribes us with the guideline to employ as large 𝐾 as possible to remove collapsed shortcut solution by making sure our distribution matching is accurate. Yet, doing so leads to unstable gradient-based training due to the gradient norm scaling as $O ( k )$ , and the variance of Monte Carlo gradient estimates growing as $O ( k ^ { 2 } m _ { 2 ( k - 1 ) } )$ for the 𝑘-th moment since $\left\| \nabla _ { \theta } m _ { k } ( P _ { \theta } ^ { ( a ) } ) \right\| =$ $\| \mathbb { E } \big [ k ( \boldsymbol { a } ^ { \top } f _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) ) ^ { k - 1 } \boldsymbol { a } ^ { \top } J _ { f _ { \boldsymbol { \theta } } } ( \boldsymbol { x } ) \big ] \| ,$ , with $J _ { f _ { \theta } } ( \pmb { x } ) \in \mathbb { R } ^ { K \times P }$ the Jacobian matrix–hereby creating an impractical situation where training stability and identifiability can not be achieved simultaneously.

## 4.2.2 Cumulative Density Functions are Impractical

The second family of tests acts upon the CDF. Because those tests require sorting, let’s denote the $k ^ { \mathrm { t h } }$ order-statistics of 𝑁 samples by $x _ { k : N }$ . Two highly standard tests are quadratic Empirical Density Function statistics with different weighting known as Cramér-von Mises [Cramér, 1928, Von Mises, 1981] and Anderson Darling [Anderson and Darling, 1952], and given by

$$
\left| \begin{array}{ccc} T _ {w} = N \int_ {- \infty} ^ {\infty} (F _ {N} (x) - F (x)) ^ {2}   w (x) d F (x) & \\ w (x) = 1, & \text {(Cramér - von Mises)} \\ w (x) = [ F (x) (1 - F (x)) ] ^ {- 1}, & \text {(Anderson - Darling)} \end{array} \right|
$$

where 𝑤(𝑥) is a weighting function. Adding the $U ^ { 2 }$ statistics on top of Equation (Cramér-von Mises) recovers the

8

<!-- page 9 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Chart block](images/page009-block03.png)

Figure 6. $N = 1 0 0$ samples are drawn from a 1024-dimensional standard Gaussian, and the first 2 coordinates are altered to produce the $\mathsf { { ^ { \prime \prime } } X ^ { \prime \prime } }$ distribution from Figure 5 (left-most column). For each statistic (all other columns), we perform gradient descent on the samples to minimize their value, at each iteration step with sample 𝑀 = 10 random directions to evaluate SIGReg (recall def. 2). We obtain that albeit this is a high-dimensional distribution with limited number of samples, SIGReg is able to capture the degenerate subspace and adapt the data accordingly to match an isotropic Gaussian distribution. Additional figures with varying dimensions and number of 1d projections are provided in Figure 16.

Watson test [Watson, 1961]

$$
U ^ {2} = T _ {w} - N \left(\bar {F} - \frac {1}{2}\right) ^ {2}.\tag{Watson}
$$

We do not consider the Kolmogorov-Smirnov test [Kolmogorov, 1933] as it employs the $\ell _ { \infty } .$ -norm instead of the ℓ<sub>2</sub>-norm hereby producing sparse gradients. Another common test is the Shapiro-Wilk test [Shapiro and Wilk, 1965] which we found to be unstable in practice–details are provided in Section E.

**Lack of Scalability and Differentiability.** CDF-based tests require sorting that have been highly optimized, e.g., with the 𝒪(𝑁 log(𝑁)) Quicksort algorithm [Hoare, 1962] but that nonetheless breaks the embarrassingly parallel nature of SGD–especially on multi-GPU [Tanasic et al., 2013, Maltenberger et al., 2022] due to synchronization requirements. Moreover, these tests involve non-differentiable operations (sorting and order statistics), making them unsuitable for gradient-based optimization without relaxations [Cuturi et al., 2019, Grover et al., 2019, Petersen et al., 2022]. While there exists intricate sketching solutions [Dunning and Ertl, 2019, Masson et al., 2019, Dunning, 2021], each of those solutions introduce numerous additional hyper-parameters–going against our first motivation for LeJEPA.

## 4.2.3 Characteristic Functions are Stable, Scalable and Identifiable

The third family of tests is concerned with Empirical Characteristic Functions (ECF) which are the Fourier transform of the density function. The Epps–Pulley test [Epps and Pulley, 1983] is one of the most popular test and simply compares in weighted $\ell _ { 2 } .$ -norm the ECF of the data against a target CF

$$
E P = N \int_ {- \infty} ^ {\infty} \left| \hat {\phi} _ {X} (t) - \phi (t) \right| ^ {2} w (t) d t. \quad \text {(Epps - Pulley)}
$$

The first crucial observation is that the ECF being defined as $\begin{array} { r } { \hat { \phi } _ { X } ( t ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } e ^ { i t X _ { j } } } \end{array}$ is naturally differentiable and easily computed in distributed settings via efficient all\_reduce operations, as the ECF is a simple average of complex exponentials. The weight function is typically Gaussian, such as 𝑤 $v ( t ) = e ^ { - t ^ { 2 } / \sigma ^ { 2 } }$ with 𝜎 commonly set to 1.

Other tests, $\mathbf { e . g . }$ , based on the Entropy [Székely and Rizzo, 2005] are not considered here as they require numerous additional design choices for the univariate Entropy estimation [Silverman, 2018, Beirlant et al., 1997], e.g., using kernels [Joe, 1989], or M-estimators [Miller, 2003].

**Epps-Pulley has bounded loss, gradient and curvature.** We now consider the remaining two families of tests: moment-based and CF-based. First, recall that moments are polynomial in the data and with extreme growth rate

9

<!-- page 10 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

LeJEPA:

**Algorithm 1.** SIGReg with Epps-Pulley statistic with DDP support and 𝒪(𝑁) time and memory complexity. x is a (N, K) tensor, num\_slices is |A| in def. 2, ‘global\_step‘ is used for sync. sampling across GPUs and can be omited for single-GPU training. An optimized implementation with caching is also provided in our official codebase, computation times provided in Table 6.

```python
def SIGReg(x, global_step, num_slices=256):
    # slice sampling -- synced across devices --
    dev = dict(device=x.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, generator=g, **dev)
    A /= A.norm(p=2, dim=0)
    # -- Epps-Pulley stat. see Sec. 4.3 for alt. --
    # integration points
    t = torch.linspace(-5, 5, 17, **dev)
    # theoretical CF for N(0, 1) and Gauss. window
    exp_f = torch.exp(-0.5 * t**2)
    # empirical CF -- gathered across devices --
    x_t = (x @ A).unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size
    T = torch.trapz(err, t, dim=1) * N
    return T
```

for higher moment–assuming they even exist. Even for well-behaved distributions, raising values to a power of 𝑘 can quickly lead to exploding gradients. This comes in sharp contrast with the ECF which is always bounded and with bounded gradients for any input distribution for the projected samples $z _ { i } = \boldsymbol { a } ^ { \top } f _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { n } ) ,   n = 1 , \cdots , N$

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Theorem 4: Stability of Epps-Pulley Test
(Epps–Pulley) satisfies for samples $z_1, \ldots, z_N$
$\left|\frac{\partial EP(\mathbf{a})}{\partial z_i}\right| \leq \frac{4\sigma^2}{N}, \quad \left|\frac{\partial^2 EP(\mathbf{a})}{\partial z_i^2}\right| \leq \frac{C\sqrt{\pi}\sigma^3}{2N},$
with constant $C$, and bandwidth $\sigma$. (Proof in Section B.12.)
</div>

By the chain rule, thm. 4 directly gives $\| \nabla _ { \theta } E P ( { \pmb a } ) \| \leq$ $\begin{array} { r } { \frac { 4 \sigma ^ { 2 } } { N } \sum _ { i = 1 } ^ { r } \| \pmb { a } ^ { \top } \nabla _ { \theta } f _ { \theta } ( \mathbf { x } _ { i } ) \| } \end{array}$ , providing stable gradients. The limitations of moment-based and CDF-based tests coupled with thm. 4 justifies our choice of the (Epps–Pulley): (i) DDP-friendly and scalable, (ii) uniformly bounded gradients and curvature regardless of input distribution, and (iii) hyper-parameter free implementation. Lastly, we highlight that our implementation has a linear memory and computational complexity of 𝒪(𝑁), with 𝑁 the minibatch size. The implementation of SIGReg using that statistical test is provided in algorithm 1, along with computation times of the forward-backward pass in Table 6.

As a last step before introducing LeJEPA, we ought to

![Chart block](images/page010-block09.png)

Figure 7. Expected directional statistic at the end of training (y-axis) for varying 𝑀 (number of directions used at each training step, x-axis). The 𝑀 directions are either resampled (green) or kept fixed (blue) at each training step. While for fixed directions we benefit from thm. 5 bound where increasing 𝑀 reduces the overall expected loss, being able to resample at every step provides significant coverage boost for free.

study the requirements on the number of directions (|A|) for (2) to be effective in high-dimension.

## 4.3 How SIGReg Beats the Curse of Dimensionality

This last section seeks to characterize how many slices in A one must sample for (SIGReg) to be an effective statistical test. That design is crucial if we hope for LeJEPA to successfully converge towards isotropic Gaussian embeddings.

## Smoothness Beats the Curse of Dimensionality

Our first argument arguing for a favorable scaling of |A| with the embedding dimension 𝐾 relies on the smoothness of $P _ { \theta }$ as measured by its Sobolev regularity 𝛼 [Adams and Fournier, 2003]. We formalize below a bound on the directional test from Equation (3) over all possible directions 𝒂 when the test statistic is minimized over $| \mathbb { A } |   =   M$ directions. While we provide bounds on the expected discrepancy over random directions 𝒂 when the EP test is satisfied (equals zero) on a finite set of directions, the provided proof includes the case of moment-based and CDF-based tests as well.

10

<!-- page 11 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

LeJEPA:

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Theorem 5: Unified Error Bounds
Let $p_{\theta} \in H^{\alpha}(\mathbb{R}^{K})$, $a \sim \mathcal{U}(\mathcal{S}^{K-1})$, and (Epps–Pulley)= 0, i.e., $P_{\theta}^{(\mathbf{a})} = Q^{(\mathbf{a})}, \forall a \in \mathbb{A}$, then
$\mathbb{E}_{a} \left[ \int_{\mathbb{R}} \left| \varphi_{a}(t) - \varphi_{\mathcal{N}}(t) \right|^{2} dt \right] \leq C(K, \alpha) |\mathbb{A}|^{-2\alpha/(K-1)}$
$\times \int_{0}^{\infty} \left\| \varphi.(r) - \varphi_{\mathcal{N}}(r) \right\|_{H^{\alpha}(\mathcal{S}^{K-1})}^{2} dr,$
(Proof in Section B.10.)
</div>

As $| \mathbb { A } | \to \infty ,$ , the bound decays as $| \mathbb { A } | ^ { - 2 \alpha / ( K - 1 ) }$ , showing that $| \mathbb { A } |   =   O ( K )$ directions suffice for 𝜖-approximation when 𝛼 is large. Some examples of embedding densities with varying 𝛼 are provided in Figure 4. The following statement characterizes how the 𝑀 directions actually constrain the entire space as a function of 𝛼. The constant $\begin{array} { r } { C ( K , \alpha ) = \frac { 2 ^ { 2 \alpha } \pi ^ { ( K - 1 ) / 2 } \Gamma \left( \alpha + \frac { K - 1 } { 2 } \right) } { ( K - 1 ) \Gamma ( \alpha ) \Gamma \left( \frac { K - 1 } { 2 } \right) } } \end{array}$ is visualized in Figure 15 (left) depicting how 𝛼 and $| \dot { \mathbb { A } } |$ interact. In words, we obtain that thanks to the natural smoothness of DN–either stemming from the architecture or the implicit and explicit regularizers used during training–applying SIGReg on |A| directions can be sufficient to tightly constrain the entire space. We note that considering the worst case over 𝒂 or using low-discrepancy sequences for 𝒂 does not impact the asymptotic bounds, details provided in Section D.

## SGD Beats the Curse of Dimensionality

Our second argument leverages the iterative nature of DN training. Although we may use only |A| to be a few hundreds, the cumulative number of sampled directions grows linearly with training time. This resampling effect (illustrated in Figure 7, bottom) enables rapid convergence. Even small |A| achieves tight distributional matching compared to keeping the set A fixed throughout minibatches (recall thm. 5). Our experiments show that even with |A| as low as 16 can easily outperform a fixed set with |A| of order of thousands thanks to the compounding effect of resampling at each minibatch.

## Empirical Validation on Synthetic Data

We conclude this section with a controlled experiment applying (SIGReg) with gradient-based training to produce isotropic embeddings. In this setup, we directly consider embeddings 𝒁 which we will differentiate and optimized to minimize (SIGReg). By directly optimizing the embeddings we are able to observe the impact of the loss without any possible constraint and regularization that would come from the architecture. We sample 𝑁 i.i.d. samples $x _ { n }$ in a 𝐷-dimensional space. This sampling is based on an isotropic Gaussian distribution–but the first

**Algorithm 2.** LeJEPA implementation–works out-of-the-box on any dataset, with DDP, with any backbone, e.g., torchvision or timm. For non-ViT architectures $( \mathbf { e . g . } ,$ ResNet), set global\_views = all\_views. We use bs for the minibatch size, SIGReg is from algorithm 1.

```python
def LeJEPA(global_views, all_views, lambd):
    """global_views and all_views are lists of
    tensors, lambd is a scalar"""

    # embedding of global views
    g_emb = forward(torch.cat(glob_views))
    # embedding of local views
    # if resnet: skip with a_emb=g_emb
    a_emb = forward(torch.cat(all_views))

    # LeJEPA loss
    centers = g_emb.view(-1, bs, K).mean(0)
    a_emb = a_emb.view(-1, bs, K)
    sim = (centers - a_emb).square().mean()
    sigreg = mean(SIGReg(emb, global_step) for emb
        in a_emb)
    return (1-lambd)*sim + lambd*sigreg
```

two dimensions are again set to the adversarial $\mathsf { { ^ { \prime \prime } } }$ shape. That is, among the 𝐷 dimensions, only two must be transformed as all the other ones already obey the isotropic Gaussian target distribution. We then make the samples $x _ { n }$ differentiable and optimize then to minimize the value of the different statistical tests compute on 𝑀 random 𝑀 random directions. Those directions are resampled after each gradient step–which follows the procedure we will employ in LeJEPA. We present the results in Figure 6 demonstrating that even in challenging case, i.e., $D = 5 1 2$ and $M = 1 6 ,$ SIGReg is able to detect the two degenerate dimensions and unfold them back to how they should look like under the target distribution.

## 5 LeJEPA: Stable and Scalable Implementation

Having established that isotropic Gaussians are the optimal embedding distribution for foundation models (Section 3) and introduced SIGReg to achieve this distribution (def. 2), we now present the complete LeJEPA framework. We first evaluate candidate statistical tests (Sections 4.2.1 and 4.2.2) and identify characteristic function-based tests as optimal for gradient-based training (Section 4.2.3). The full LeJEPA implementation follows in Section 5.1.

## 5.1 LeJEPA: SIGReg + Prediction Loss

We now discuss the implementation of LeJEPA starting with SIGReg and followed by the prediction and total losses.

**The SIGReg Loss.** We chose (Epps–Pulley) for its provable boundedness (thm. 4) and its scalability. Its implementation follows exactly the equation except for the integrate which is estimated using a quadrature approximation. We

11

<!-- page 12 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

find that the simple trapezoidal quadrature rule is sufficient even with as few knots as $1 7 ,$ as ablated in Figure 20. In particular, we leverage the symmetry of the integrand to double the number of knots for free, see the official code. On the other hand, the use of minibatches introduces a bias vanishing at rate $O ( 1 / N )$ , as formalized below.

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Theorem 6: Vanishing gradient bias
The expectation of (Epps–Pulley) satisfies
 $\mathbb{E}\left[\widehat{L}_{n}(\theta)\right]=L(\theta)+\frac{1}{N}\int_{\mathbb{R}}w_{s}(t)\big(1-|\varphi_{P}(t)|^{2}\big)dt,$ 
therefore both the loss and its derivative have a bias of order  $O(1/n)$ . (Proof in Section B.13.)
</div>

Hence, the gradients we obtain from using (Epps–Pulley) are biased by an explicit $O ( 1 / N )$ term. We found this bias to be minimal and not a concern even for minibatches as small as 16. Unbiased alternatives include using Ustatistic debiasing of $| \phi _ { \theta } | ^ { 2 }$ or sample splitting, which we do not explore in this study. Our final implementation of the SIGReg term with Epps-Pulley statistic is provided in algorithm 1.

**The Prediction Loss.** To standardize notations, we adopt the DINO [Caron et al., 2021] setup of generating $V _ { g }$ global views and $V _ { l }$ local views, leading to a total of $V = V _ { g } + V _ { l }$ views. We set the first $1 , \ldots , V _ { g }$ indices of each $z _ { n , v }$ as the global views. For the cases without local views, simply set $V _ { l } = 0$ . The prediction loss is then given by having all views predict the global views as

$$
\mathcal {L} _ {\mathrm{pred}} (\{\boldsymbol {z} _ {n, v} \} _ {v = 1} ^ {V}) = \frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \| \boldsymbol {z} _ {n, v} - \boldsymbol {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2}\tag{5}
$$

$$
\left| = \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \left\| \frac {1}{V _ {\mathrm{g}}} \sum_ {v = 1} ^ {V _ {\mathrm{g}}} z _ {n, v} - z _ {n, v ^ {\prime}} \right\| _ {2} ^ {2} \right|\tag{6}
$$

$$
\triangleq \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \left\| \boldsymbol {\mu} _ {n} - \boldsymbol {z} _ {n, v ^ {\prime}} \right\| _ {2} ^ {2},\tag{7}
$$

where we denote $\begin{array} { r } { \pmb { \mu } _ { n }   \triangleq   \frac { 1 } { V _ { g } } \sum _ { v = 1 } ^ { V _ { g } } z _ { n , v } , } \end{array}$ , the Equation (5) to Equation (6) derivations are detailed in Section B.6.

**LeJEPA Loss.** The final total loss simply combines the above prediction loss along with SIGReg on each views as per

$$
\begin{array}{c} \mathcal {L} _ {\text {LeJEPA}} (\{\boldsymbol {x} _ {n, v} \} _ {n, v = 1} ^ {B, V}) = \frac {\lambda}{V} \sum_ {v = 1} ^ {V} \text {SIGReg} (\{\{\boldsymbol {z} _ {n, v} \} _ {n = 1} ^ {B} \}) \\ + \frac {1 - \lambda}{B} \sum_ {n = 1} ^ {B} \mathcal {L} _ {\text {pred}} ^ {(V _ {\mathrm{g}})} (\{\boldsymbol {z} _ {n, v} \} _ {v = 1} ^ {V}). \quad (\text {LeJEPA}) \end{array}
$$

We present (LeJEPA)’s implementation in algorithm 2. Altogether, the entire implementation–besides the usual model definitions, optimizers, and data loaders–only takes a few dozens lines in PyTorch (algorithms 1 and 2). The absence of prototypes, stop-gradients, and teacher-student networks makes (LeJEPA) appealing as it only contains one hyperparameter, 𝜆, balancing the trade-off between the prediction and isotropic Gaussian terms.

## 5.2 Relation to Prior Work

Prior to presenting our experiments (Section 6), we conclude by discussing how our proposed LeJEPA and SIGReg objective relate to existing frameworks in the literature.

While there is no existing solution employing such slicing and distribution matching for JEPAs, there exists similar pipelines for generative models and optimal transport. Notably, the Sliced Score Matching [Song et al., 2020] proposes to leverage univariate slicing of the space to ease the estimation of a density for generative models. In a similar vein, the sliced Wasserstein distance [Bonneel et al., 2015, Nguyen and Ho, 2023] uses such strategy to speed up and improve optimal transport. Furthermore, when the integral of the (Epps–Pulley) test is computed exactly, as opposed to our quadrature, each slice loss value recovers the kernel MMD [Sriperumbudur et al., 2010, Gretton et al., 2012, Chwialkowski et al., 2016] measuring the distance between two distributions–albeit with a quadratic complexity. Lastly, it is possible to recover some existing SSL frameworks in the limit by employing LeJEPA with a particular test–instead of the preferred (Epps–Pulley). For example, Setting $T ( \{ x _ { n } \} _ { n = 1 } ^ { B } ) \stackrel { \star } { = } \mathrm { m e a n } ( \{ x _ { n } \} _ { n = 1 } ^ { B } ) ^ { 2 } + ( \mathrm { s t d } ( \{ x _ { n } \} _ { n = 1 } ^ { B } ) - 1 ) ^ { 2 }$ and using that 𝑇 with SIGReg in LeJEPA recovers the VICReg SSL method in the limit of large number of slices. In fact, SIGReg will enforce in expectation that $\mathbb { E } [ \mathbf { Z } ] = \mathbf { 0 }$ and $\operatorname { C o v } ( \mathbf { Z } )   =   \mathbf { I } _ { d } ,$ where $\mathbf { I } _ { d }$ denotes the 𝑑 $\times   d$ identity matrix–derivations provided in Section B.14. And since our invariance term is simply the $\ell _ { 2 }$ distance between the views’ embeddings, LeJEPA recovers VICReg for this degenerate statistical test. Based on thm. 3, we however strongly advocate against such a setting as it would lead to shortcut solutions–a phenomenon already observed in VICReg.

## 6 LeJEPA: Empirical Validation

We now use the LeJEPA implementation described in Section 5.1 to demonstrate its effectiveness through comprehensive experiments. We show that LeJEPA: (i) trains reliably across diverse architectures and datasets (Section 6.1), (ii) provides an informative training loss for model selection (Section 6.2), (iii) outperforms frontier vision models on small-scale in-domain pretraining (Section 6.3), (iv) scales successfully to nearly 1 billion parameters on ImageNet-1k (Section 6.4), and (v) learns rich

12

<!-- page 13 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Chart block](images/page013-block03.png)

Figure 8. Inet100 with 400 pretraining epochs and resnet50 backbone. We depict linear probe performances as a function of 𝜆 and the number of views 𝑉 (recall (LeJEPA)). We observe that performances are stable over 𝜆–with peak performance obtain by slightly adjust 𝜆 proportionally to the number of views. The corresponding performance values are provided in Table 7.

semantic segmentation features without explicit supervision.

## 6.1 LeJEPA’s Stability Across

## Hyper-Parameters and Architectures

We now demonstrate LeJEPA’s stability across hyperparameters, architectures, and experimental setups. Additional cross-domain stability results are presented in Section 6.3.

**Stability across standard hyperparameters.** We begin by evaluating LeJEPA on ImageNet-100 and ImageNet-1K. On ImageNet-100, we train a ResNet-50 and vary the number of views and the loss weighting 𝜆 (Figure 8). Performance remains stable across both dimensions, leading us to recommend $\lambda   =   0 . 0 5$ as a robust default. On ImageNet-1K, we train a ViT-Large/14 and explore batch size, as well as the number of global $( V _ { \mathrm { g } } )$ and local (𝑉<sub>l</sub>) views (Table 1b). We find that the configuration commonly used in prior work $( V _ { \mathrm { g } } = 2 , V _ { \mathrm { l } } = 8 )$ transfers well to LeJEPA. Notably, LeJEPA achieves competitive performance with batch sizes as small as 128 on ImageNet-1K (Table 1c), suggesting reduced memory requirements compared to existing methods. We thus recommend to use $\lambda \; = \; 0.05,$ $V _ { \mathrm { g } } = 2 ,   V _ { \mathrm { l } } = 8 ,$ and batch size $\geq 1 2 8$ as starting points.

**Stability across Epps-Pulley hyperparameters.** We next examine hyperparameters specific to LeJEPA: the number of slices |𝒜| in SIGReg, the integration domain for the Epps-Pulley test (Epps–Pulley), and the number of quadrature points for numerical integration. Table 1a shows ablations on ImageNet-1K with ViT-Large/14. Both the integration domain and number of quadrature points have negligible impact on performance. This is expected: since the characteristic function is accurate at zero, the

Table 1. ViT/Large-14, on inet1k pretraining for 100 epochs and evaluated with frozen backbone linear probing (top1 accuracy, %).LeJEPA’s performance is stable across all its hyperparameters and while some may slightly improve performance, e.g., the number of slices |A| and the projector sizes, none of the choices lead to a catastrophic collapse.

(a) (Epps–Pulley) parameters

<table><tr><td rowspan="2">integration</td><td rowspan="2">num_slices</td><td colspan="3">config/bstat_n_points</td></tr><tr><td>5</td><td>17</td><td>41</td></tr><tr><td rowspan="2">[-1,1]</td><td>512</td><td>71.82</td><td>72.13</td><td>72.04</td></tr><tr><td>2048</td><td>72.88</td><td>72.30</td><td>72.69</td></tr><tr><td rowspan="2">[-3,3]</td><td>512</td><td>73.95</td><td>74.16</td><td>74.04</td></tr><tr><td>2048</td><td>75.02</td><td>74.68</td><td>74.77</td></tr><tr><td rowspan="2">[-5,5]</td><td>512</td><td>73.71</td><td>74.21</td><td>74.15</td></tr><tr><td>2048</td><td>74.50</td><td>74.80</td><td>74.77</td></tr></table>

(b) Number of local/global views

| # global_views (𝑉<sub>g</sub>) # views (𝑉 = 𝑉<sub>g</sub> + 𝑉<sub>l</sub>) | 1 | 2 | 4 |
| --- | --- | --- | --- |
| 4 | 53.06 | 72.26 | - |
| 6 | 58.65 | 73.07 | 73.68 |
| 8 | 64.46 | 74.24 | 73.94 |
| 10 | 68.97 | 74.06 | 75.08 |

<table><tr><td colspan="5">(c) Mini-batch size</td></tr><tr><td>batch_size</td><td>128</td><td>256</td><td>512</td><td>1024</td></tr><tr><td></td><td>72.20</td><td>74.15</td><td>74.72</td><td>74.07</td></tr></table>

| (d) Em | beddin | g/Projec | tor dim | . |
| --- | --- | --- | --- | --- |
| num_slices | 10 | 24 | 40 | 96 |
| emb. dim. | 512 | 2048 | 512 | 2048 |
| proj. dim. |  |  |  |  |
| 64 | 75.29 | 75.32 | 75.50 | 75.65 |
| 128 | 74.77 | 75.09 | 75.26 | 75.47 |
| 256 | 74.56 | 74.66 | 75.08 | 75.02 |
| 512 | 73.94 | 74.11 | 74.81 | 74.65 |
| 1024 | 73.65 | 73.94 | 74.71 | 74.79 |

(e) Register tokens

| reg_tokens | 0 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| num_slices |  |  |  |  |  |
| 1024 | 75.14 | 75.18 | 75.08 | 75.34 | 75.23 |
| 4096 | 75.61 | 75.58 | 75.67 | 75.63 | 75.84 |

moments of the distribution are well-characterized even with a modest integration range. The number of slices |𝒜| has a modest effect—while more slices slightly improve performance, even 512 slices yield competitive results. We thus recommend to use 17 integration points, an integration domain of [−5, 5], and 1024 slices as starting points.

13

<!-- page 14 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Inet10 | LeJEPA pretrained, frozen backbone, linear eval | 50 architectures ( < 20M params.)

![Image block](images/page014-block04.png)

Figure 9. INet10 pretraining and frozen backbone linear evaluation across 50 timm models using LeJEPA out of the box. We cross-validate the learning rate and weight-decay. While there is a small variation between the best and worst performing model, we clearly see that across 50 models spanning 8 families, LeJEPA is able to produce non-trivial representations able to solve the downstream task at SOTA levels.

**Stability across architectures.** A key advantage of LeJEPA over recent methods (e.g., ĲEPA, DINOv2) is its architecture-agnostic design. While most modern self-supervised methods are tailored to Vision Transformers, LeJEPA works across diverse architecture families without modification. To validate this claim, we pretrain approximately 50 architectures from 8 different families on ImageNet-10, selecting all models in the timm library with fewer than 20M parameters. All models are able to learn high-quality representations reaching between 91.5% to 95% top 1 accuracy with frozen backbone linear probing. It seems that models performing well in supervised learning setups are also the ones to favor for LeJEPA, such as resnets and ViTs. We thus recommend to use standard architectures such as ResNets and ViTs over specialized models like EfficientNet as stating point.

**Removal of popular heuristics.** In addition to providing reliable performance across models and datasets, LeJEPA’s provable construction enables us to remove many heuristics traditionally used to prevent collapse. First, prior work has shown both empirically and theoretically that predic tors in image JEPA (without asymmetric information) and teacher-student architectures serve primarily to prevent collapse [Grill et al., 2020, Jing et al., 2021, Tian et al., 2021, Caron et al., 2021, Chen et al., 2021]. Removing these components produces collapsed encoders, i.e., with performances at chance-level. Thanks to LeJEPA’s SIGReg loss, we can remove both the predictor and teacher-student architecture without suffering from collapse, as shown in Table 4. While a teacher-student configuration does provide a small performance boost for ViT models—consistent with observations in supervised learning via Stochastic

Weight Averaging [Izmailov et al., 2019]—it is not necessary to prevent collapse. In our setup, we apply SWA on the encoder producing 𝜇 in Equation (6). Second, recent work demonstrated that register tokens are needed to pre-vent training instabilities in vision models [Oquab et al., 2023, Siméoni et al., 2025, Darcet et al., 2023]. We show in Table 1 that such instabilities likely stem from poorly conditioned training objectives. In contrast, LeJEPA does not require register tokens and achieves stable performance with or without them. We thus recommend training without a predictor or register tokens, and optionally applying SWA with ViTs for a possible performance gain.

## Experiment Details 1

We strive for **simplicity** and thus adopt a unified pretraining pipeline. The following parameters apply to all experiments and figures unless stated otherwise in the corresponding caption and come from Section 6.1:

LeJEPA’s implementation is given in algorithm 2 with hyperparameter 𝜆

All backbones are from timm and all optimizers/schedulers are from PyTorch without modifications

We employ eight views (𝑉 = 8) containing two global views $( V _ { \mathrm { g } } = 2 )$ with resolution 224x224 and 96x96 for the local views

AdamW optimizer with lr $\in \{ 5 e - 3 , 5 e - 4 \}$ and wd ∈ {1𝑒 − 1, 1𝑒 − 2, 1𝑒 − 5}–no scheduler on weight-decay, standard linear warm-up cosine-annealing for lr

## 6.2 LeJEPA’s Training Loss is Informative of Downstream Performance

A major challenge in SSL pretraining is the lack of reliable signals conveying the quality of the learned representation. As a result, it is common to monitor a supervised

14

<!-- page 15 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Chart block](images/page015-block03.png)

![Chart block](images/page015-block04.png)

![Chart block](images/page015-block05.png)

Figure 10. (SIGReg, prediction loss) 2𝑑-plane with downstream task accuracy shown with colors from blue (low) to red (high). We clearly observe that within this plane, there exists trade-off fronts between the two terms of LeJEPA producing similar downstream performance corresponding to different values of 𝜆. Yet, those fronts are linear and pointed towards the lower left corner, i.e., LeJEPA’s training loss informs of downstream test performance across models and datasets (columns). Additional models and datasets provided in Figure 21.

![Chart block](images/page015-block07.png)

Figure 11. Spearman correlation (y-axis) between LeJEPA’s training loss and downstream accuracy on the dataset’s classification task with a frozen backbone and linear evaluation. The x-axis varies 𝛼 in Equation (8) following our scaling law of the loss w.r.t. 𝜆. Using $\alpha = 0$ recovers the plain training loss. We clearly observe a very high correlation already for $\dot { \alpha } = 0 ,$ which further increases up to 99% for $\dot { \alpha } = 0 . 4$ . The entire set of points is obtained across numerous hyper-parameters such as learning rate, weight decay, number of epochs, 𝜆–demonstrating how LeJEPA’s training loss is strongly predictive of downstream performance which can be used for label-free cross-validation.

downstream task performance, sometimes supplemented with unsupervised embedding statistics [Agrawal et $\mathrm { a l . , }$ 2022, Garrido et al., 2023, Thilak et al., 2023]. This process is highly limiting since it requires labeled data that is costly and overly specialized. This is further exacerbated in the latest JEPA models where training losses exhibit low correlation with downstream performance–and may not even decrease monotonically during training.

In contrast, we find that LeJEPA’s training loss behaves much more favorably–providing us with a meaningful signal on model quality. First, we provide in Figure 10, the 2D plane spanned by the SIGReg and prediction losses

where a clear trend with downstream task accuracy can be observed. More strikingly, the combined training loss (LeJEPA) with mixing coefficient 𝜆 exhibits very high Spearman correlation [Spearman, 1961], denoted as $\rho _ { s } ,$ of about 85% with downstream accuracy–which is considered a strong signal. This strong relationship holds across datasets and architectures. As a result, a lower LeJEPA training loss reliably indicates a better downstream performance.

We can further improve this correlation through a simple scaling law based upon the trade-off weighting hyperparameter 𝜆

$$
C ^ {(\alpha)} = \rho_ {s} \left(\frac {\text {train\_loss}}{\lambda^ {\alpha}}, \text {test\_accuracy}\right).\tag{8}
$$

By setting $\alpha \approx 0 . 4 ,$ LeJEPA’s training loss is able to achieve nearly 99% correlation with downstream performance across multiple datasets and models. We depict the changes in $\bar { C } ^ { ( \alpha ) }$ as a function of 𝛼 on multiple datasets and models in Figure 11, as well as the training LeJEPA loss against downstream performance in Figure 19. **The strong alignment between LeJEPA’s training loss and model quality enables label-free SSL model selection and cross-validation**.

## 6.3 In-Domain LeJEPA Outperforms Frontier Model Transfer Learning

A key promise of self-supervised learning is to learn universal representations that generalize across tasks and domains. However, current frontier foundation models (e.g., DINOv2/v3, ĲEPA) are pretrained on natural images forcing practitioners in specialized domains to collect large amount of labels for supervised finetuning. In fact, most frontier models can not be trained directly on those domains as the number of samples may be small and searching again for the hyper-parameters would be cum-

15

<!-- page 16 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Chart block](images/page016-block03.png)

Figure 12. Small architecture in-domain (Galaxy10) LeJEPA pretraining with linear probe evaluation using frozen backbone or full finetuning (columns) and with varying number of samples per class (x-axis). We compare against state-of-the-art foundation models (DINOv2/v3, ĲEPA) over 3 different random seeds. We observe that LeJEPA enables in-domain pretraining out of the box across architectures and able to outperform frontier foundation models. Corresponding numbers are provided in Table 3.

Table 2. Few-shot classification accuracy (percentages) on 8 datasets spanning textures, objects, and fine-grained categories. Our LeJEPA achieves superior performance on fine-grained tasks (DTD, flowers102, food101) while requiring only 100 pretraining epochs compared to I-JEPA’s 300 epochs—a 3× reduction in training time and computational resources without sacrificing downstream task performance. This efficiency gain is particularly valuable for practical applications where training budget is limited. Bold indicates best performance within the IN-1K comparison group, all numbers are percentages.

<table><tr><td rowspan="2">shots</td><td rowspan="2">model</td><td rowspan="2">params</td><td rowspan="2">pretrain</td><td rowspan="2">epochs</td><td colspan="9">Dataset</td></tr><tr><td>DTD</td><td>aircr.</td><td>cars</td><td>cifar10</td><td>cifar100</td><td>flowers102</td><td>food</td><td>pets</td><td>avg.</td></tr><tr><td rowspan="5">1</td><td>LeJEPA ViT-L</td><td>304M</td><td>IN-1K</td><td>100</td><td>33.21</td><td>9.37</td><td>3.40</td><td>51.65</td><td>27.01</td><td>48.53</td><td>17.14</td><td>46.11</td><td>29.55</td></tr><tr><td>LeJEPA ConvNeXtV2-H</td><td>660M</td><td>IN-1K</td><td>100</td><td>32.15</td><td>8.07</td><td>4.28</td><td>50.95</td><td>31.48</td><td>48.74</td><td>17.95</td><td>58.98</td><td>31.58</td></tr><tr><td>I-JEPA ViT-H</td><td>632M</td><td>IN-1K</td><td>300</td><td>27.71</td><td>9.86</td><td>4.33</td><td>56.52</td><td>30.58</td><td>44.69</td><td>14.53</td><td>53.38</td><td>30.20</td></tr><tr><td>I-JEPA ViT-H + STOP</td><td>632M</td><td>IN-1K</td><td>300</td><td>26.60</td><td>11.18</td><td>4.75</td><td>56.27</td><td>35.20</td><td>47.17</td><td>15.75</td><td>59.47</td><td>32.05</td></tr><tr><td>I-JEPA ViT-H (22K)</td><td>632M</td><td>IN-22K</td><td>900</td><td>27.98</td><td>13.00</td><td>3.45</td><td>61.84</td><td>34.70</td><td>89.72</td><td>19.62</td><td>30.86</td><td>35.15</td></tr><tr><td rowspan="5">10</td><td>LeJEPA ViT-L</td><td>304M</td><td>IN-1K</td><td>100</td><td>64.72</td><td>35.25</td><td>22.25</td><td>85.15</td><td>59.77</td><td>92.53</td><td>50.90</td><td>77.00</td><td>60.95</td></tr><tr><td>LeJEPA ConvNeXtV2-H</td><td>660M</td><td>IN-1K</td><td>100</td><td>61.84</td><td>30.67</td><td>24.46</td><td>85.74</td><td>63.29</td><td>91.78</td><td>49.32</td><td>78.53</td><td>60.70</td></tr><tr><td>I-JEPA ViT-H</td><td>632M</td><td>IN-1K</td><td>300</td><td>57.68</td><td>33.82</td><td>21.96</td><td>88.77</td><td>66.42</td><td>88.24</td><td>43.97</td><td>83.23</td><td>60.51</td></tr><tr><td>I-JEPA ViT-H + STOP</td><td>632M</td><td>IN-1K</td><td>300</td><td>57.00</td><td>39.77</td><td>25.21</td><td>90.09</td><td>70.32</td><td>90.16</td><td>45.68</td><td>85.13</td><td>62.92</td></tr><tr><td>I-JEPA ViT-H (22K)</td><td>632M</td><td>IN-22K</td><td>900</td><td>58.74</td><td>43.52</td><td>18.27</td><td>94.83</td><td>75.23</td><td>98.94</td><td>49.06</td><td>67.66</td><td>63.28</td></tr><tr><td rowspan="5">all</td><td>LeJEPA ViT-L</td><td>304M</td><td>IN-1K</td><td>100</td><td>78.30</td><td>57.01</td><td>57.28</td><td>96.50</td><td>83.71</td><td>91.21</td><td>82.05</td><td>89.74</td><td>79.48</td></tr><tr><td>LeJEPA ConvNeXtV2-H</td><td>660M</td><td>IN-1K</td><td>100</td><td>76.60</td><td>52.99</td><td>54.88</td><td>96.15</td><td>81.34</td><td>91.11</td><td>77.64</td><td>89.76</td><td>77.56</td></tr><tr><td>I-JEPA ViT-H</td><td>632M</td><td>IN-1K</td><td>300</td><td>73.32</td><td>56.61</td><td>54.47</td><td>97.54</td><td>86.42</td><td>86.47</td><td>81.02</td><td>92.11</td><td>78.50</td></tr><tr><td>I-JEPA ViT-H + STOP</td><td>632M</td><td>IN-1K</td><td>300</td><td>73.87</td><td>61.95</td><td>61.27</td><td>98.02</td><td>87.78</td><td>88.08</td><td>81.72</td><td>92.88</td><td>80.70</td></tr><tr><td>I-JEPA ViT-H (22K)</td><td>632M</td><td>IN-22K</td><td>900</td><td>75.67</td><td>65.39</td><td>49.79</td><td>98.46</td><td>89.95</td><td>98.54</td><td>81.58</td><td>87.19</td><td>80.82</td></tr></table>

![Image block](images/page016-block07.png)

Figure 13. Emergent Object Segmentation via Last Layer Thresholding. LeJEPA naturally learns to segment and track salient objects (shown in attention maps on the right of each video) without explicit supervision. The results display impressive visual quality and strong temporal consistency across video frames (videos provided on our [project page](https://rbalestr-lab.github.io/lejepa/)). This emergent capability demonstrates the rich semantic representations learned through our self-supervised approach.

16

<!-- page 17 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

![Image block](images/page017-block03.png)

Figure 14. LeJEPA learns rich semantic representations through self-supervised learning. PCA visualization of last-layer features from LeJEPA (ViT-Large, 100 epochs on ImageNet-1K). For each image, features are independently projected to RGB using the first 3 principal components. Without any supervision, LeJEPA spontaneously develops semantically meaningful representations: notice how warm colors (red/- magenta/pink) consistently capture foreground objects (parrot bodies, dog face), while cool colors (cyan/green/yellow) represent backgrounds and foliage. This emergent object-background separation and perceptual grouping discovered the visual structure of the world purely from unlabeled data.

bersome yet necessary [Assran et al., 2022].

To demonstrate LeJEPA’s versatility and ability to resolve that current pain-point, we propose to pretrain directly on a new domain without any change in the loss or the pretraining pipeline. We select the Galaxy10 dataset, a galaxy morphology classification task that differs significantly from natural images in both visual structure and statistical properties [Balestriero et al., 2025]. The dataset contains 11,000 training samples across 10 galaxy types. For LeJEPA, we use the default hyper-parameters and pre-train for 400 epochs a variety of backbones. We compare against the latest DINOv2, DINOv3 and ĲEPA. We report in Figure 12 the top1 accuracy for linear probing both with frozen backbone and full-finetuning. We observe that **in-domain pretraining with LeJEPA substantially outperforms state-of-the-art frontier models (DINOv2, DINOv3) on both linear probing and full finetuning**. Additional datasets and backbones are provided in Table 5 depicting LeJEPA’s ability to train in-domain, even with a dataset with 1000 samples (flowers102). Coupling this result with the stability of LeJEPA across architectures and hyper-parameters should offer a promising alternatives in domains not yet accounted for by the latest frontier models.

## 6.4 LeJEPA Scales Across Data and Models

We now propose to apply LeJEPA over a larger pretraining dataset, i.e., Imagenet-1k, and over larger backbones such as ViT/Large (0.3B), ConvNextV2-Huge (0.6B). For those two models, we reach an online linear probe accuracy on inet1k of 77.1% and 78.5% respectively. Beyond in-distribution performances, we also explore transfer learning. For those experiments, our baselines are ĲEPA with a ViT-Huge (0.6B) which is the closest to our setup, and we also include a recent improved version of ĲEPA including additional stochastic prediction tasks [Bar et al., 2023] that is coined ĲEPA + STOP. For LeJEPA, we employ the same recipe as described in Section 6.1 and report transfer learning performances with frozen backbone in Table 2. We observe that we consistently outperform ĲEPA while employed a smaller model and shorted training schedule. Beyond top1 accuracy, we also echo our findings from Section 6.2 about LeJEPA’s training loss quality. In our setup, we observe a very stable and smooth training curve indicating a stable optimization landscape removing the need for careful hyperparameter selection (recall thm. 4). We provide an example on a ViT-gigantic (1.8B parameters) in Figure 1.

## 6.5 Emergent Semantic Structure in LeJEPA Representations

A hallmark of successful self-supervised learning is the emergence of semantically meaningful attention patterns

17

<!-- page 18 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

without explicit supervision [Caron et al., 2021]. To assess whether LeJEPA learns such structure, we visualize the attention maps of the learned representations. Following DINO [Caron et al., 2021], we apply PCA to the embeddings and visualize the first principal components, which reveal clear correspondence to object boundaries and salient regions (Figure 14). Furthermore, we explore whether these attention patterns can enable unsupervised video segmentation—a challenging task requiring temporal consistency and object understanding. By thresholding the self-attention maps of the [CLS] token, we obtain binary masks that track objects across frames without any segmentation labels during training. As shown in Figure 13, **LeJEPA’s attention naturally segments foreground objects from background with remarkable temporal co-herence**, suggesting that the learned representations capture both spatial semantics and temporal structure. This emergent capability demonstrates that LeJEPA’s stability focused objective does not sacrifice the semantic richness of learned features.

## 7 Conclusion

We have established a principled theoretical framework for JEPA-based self-supervised learning that fundamentally resolves its core pathologies. Our contributions span theory and practice: we proved that isotropic Gaussian embeddings uniquely minimize worst-case downstream risk, introduced SIGReg as a tractable and provably correct method to enforce this distribution, and demonstrated that this approach eliminates representational collapse by design–and not through ad-hoc combinations of teacherstudent networks, stop-gradients, or asymmetric architectures.

We validate LeJEPA across domains and over 60 architectures including gigantic versions with 1.8B parameters. In spite of its simplicify , LeJEPA matches state-of-the-art performance while requiring fewer than 50 lines of core implementation. Critically, our approach provides what SSL has long needed: a mathematically rigorous foundation that directly informs practical algorithm design.

## Acknowledgments

We would like to thank Mike Rabbat and Lucas Maes for providing valuable feedbacks on the manuscript.

## References

Haneen Arafat Abu Alfeilat, Ahmad BA Hassanat, Omar Lasassmeh, Ahmad S Tarawneh, Mahmoud Bashir Alhasanat, Hamzeh S Eyal Salman, and VB Surya Prasath. Effects of distance measure choice on k-nearest neighbor classifier performance: a review. Big data, 7(4):221–248, 2019.

Robert A Adams and John JF Fournier. Sobolev spaces, volume 140. Elsevier, 2003.

Kumar K Agrawal, Arnab Kumar Mondal, Arna Ghosh, and Blake Richards. a-req: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. Advances in Neural Information Processing Systems, 35:17626–17638, 2022.

Theodore W Anderson and Donald A Darling. Asymptotic theory of certain" goodness of fit" criteria based on stochastic processes. The annals of mathematical statistics, pages 193–212, 1952.

Mahmoud Assran, Randall Balestriero, Quentin Duval, Florian Bordes, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, and Nicolas Ballas. The hidden uniform cluster prior in self-supervised learning. arXiv preprint arXiv:2210.07277, 2022.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.

Randall Balestriero and Yann LeCun. Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods. Advances in Neural Information Processing Systems, 35:26671–26685, 2022.

Randall Balestriero and Yann LeCun. Learning by reconstruction produces uninformative features for perception. arXiv preprint arXiv:2402.11337, 2024.

Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al. A cookbook of self-supervised learning. arXiv preprint arXiv:2304.12210, 2023.

Randall Balestriero, Nicolas Ballas, Mike Rabbat, and Yann LeCun. Gaussian embeddings: How jepas secretly learn your data density. arXiv preprint arXiv:2510.05949, 2025.

Amir Bar, Florian Bordes, Assaf Shocher, Mahmoud Assran, Pascal Vincent, Nicolas Ballas, Trevor Darrell, Amir Globerson, and Yann LeCun. Stochastic positional embeddings improve masked image modeling. arXiv preprint arXiv:2308.00566, 2023.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.

18

<!-- page 19 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Jan Beirlant, Edward J Dudewicz, László Györfi, Edward C Van der Meulen, et al. Nonparametric entropy estimation: An overview. International Journal of Mathematical and Statistical Sciences, 6(1):17–39, 1997.

Chris M Bishop. Training with noise is equivalent to tikhonov regularization. Neural computation, 7(1):108–116, 1995.

Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006.

Gunnar Blom. Statistical estimates and transformed betavariables. PhD thesis, Almqvist & Wiksell, 1958.

Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and radon wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision, 51(1):22–45, 2015.

Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah. Signature verification using a" siamese" time delay neural network. Advances in neural information processing systems, 6, 1993.

Jerome S Bruner and Leo Postman. On the perception of incongruity: A paradigm. Journal of personality, 18(2): 206–223, 1949.

Russel E Caflisch. Monte carlo and quasi-monte carlo methods. Acta numerica, 7:1–49, 1998.

Torsten Carleman. Les Fonctions quasi analytiques: leçons professées au College de France. Gauthier-Villars, 1926.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jé- gou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020a.

Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems, 33:22243–22255, 2020b.

Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9640–9649, 2021.

Kacper Chwialkowski, Heiko Strathmann, and Arthur Gretton. A kernel test of goodness of fit. In International conference on machine learning, pages 2606–2615. PMLR, 2016.

Romain Cosentino, Anirvan Sengupta, Salman Avestimehr, Mahdi Soltanolkotabi, Antonio Ortega, Ted Willke, and Mariano Tepper. Toward a geometrical understanding of self-supervised contrastive learning. arXiv preprint arXiv:2205.06926, 2022.

Thomas M Cover. Elements of information theory. John Wiley & Sons, 1999.

Harald Cramér. On the composition of elementary errors: First paper: Mathematical deductions. Scandinavian Actuarial Journal, 1928(1):13–74, 1928.

Harald Cramér and Herman Wold. Some theorems on distribution functions. Journal of the London Mathematical Society, 1(4):290–294, 1936.

Marco Cuturi, Olivier Teboul, and Jean-Philippe Vert. Differentiable ranking and sorting using optimal transport. Advances in neural information processing systems, 32, 2019.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.

Josef Dick and Friedrich Pillichshammer. Digital nets and sequences: discrepancy theory and quasi–Monte Carlo integration. Cambridge University Press, 2010.

Ted Dunning. The t-digest: Efficient estimates of distributions. Software Impacts, 7:100049, 2021.

Ted Dunning and Otmar Ertl. Computing extremely accurate quantiles using t-digests. arXiv preprint arXiv:1902.04023, 2019.

Gustav Elfving. The asymptotical distribution of range in samples from a normal population. Biometrika, 34(1/2): 111–119, 1947.

Thomas W Epps and Lawrence B Pulley. A test for normality based on the empirical characteristic function. Biometrika, 70(3):723–726, 1983.

Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for self-supervised representation learning. In International conference on machine learning, pages 3015–3024. PMLR, 2021.

David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling language-free visual representation learning. arXiv preprint arXiv:2504.01017, 2025.

19

<!-- page 20 of 50 -->

LeJEPA: Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Ronald Aylmer Fisher. Statistical methods for research workers. Number 5. Oliver and Boyd, 1928.

Karl Friston. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 11(2):127–138, 2010.

Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank. In International conference on machine learning, pages 10929–10974. PMLR, 2023.

Gene H Golub, Per Christian Hansen, and Dianne P O’Leary. Tikhonov regularization and total least squares. SIAM journal on matrix analysis and applications, 21(1): 185–194, 1999.

Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT press Cambridge, 2016.

Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In Proceedings of the ieee/cvf International Conference on computer vision, pages 6391–6400, 2019.

Richard Langton Gregory. Perceptions as hypotheses. Philosophical Transactions of the Royal Society of London. B, Biological Sciences, 290(1038):181–197, 1980.

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. The journal of machine learning research, 13(1):723–773, 2012.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. Stochastic optimization of sorting networks via continuous relaxations. arXiv preprint arXiv:1903.08850, 2019.

AK Gupta. Estimation of the mean and standard deviation of a normal population from a censored sample. Biometrika, 39(3/4):260–273, 1952.

Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 297–304. JMLR Workshop and Conference Proceedings, 2010.

JM Hammersley and KW Morton. The estimation of location and scale parameters from grouped data. Biometrika, 41(3/4):296–301, 1954.

Felix Hausdorff. Momentprobleme für ein endliches intervall. Mathematische Zeitschrift, 16(1):220–248, 1923.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

H von Helmholtz et al. Handbook of physiological optics. Voss, Leipzig, 1867.

R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670, 2018.

C. A. R. Hoare. Quicksort. The Computer Journal, 5(1):10–16, 01 1962. ISSN 0010-4620. doi: 10.1093/comjnl/5.1.10. URL [https://doi.org/10.1093/comjnl/5.1.10](https://doi.org/10.1093/comjnl/5.1.10).

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization, 2019. URL [https://arxiv.org/abs/1803.05407](https://arxiv.org/abs/1803.05407).

Carlos M Jarque and Anil K Bera. Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics letters, 6(3):255–259, 1980.

Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. arXiv preprint arXiv:2110.09348, 2021.

Harry Joe. Estimation of entropy and other functionals of a multivariate density. Annals of the Institute of Statistical Mathematics, 41(4):683–697, 1989.

Thomas Kerdreux, Alexandre Tuel, Quentin Febvre, Alexis Mouche, and Bertrand Chapron. Efficient self-supervised learning for earth observation via dynamic dataset curation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3017–3027, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

20

<!-- page 21 of 50 -->

LeJEPA: Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Diederik P Kingma, Danilo J Rezende, Shakir Mohamed, and Max Welling. Semi-supervised learning with deep generative models. Advances in neural information processing systems, 27, 2014.

A. N. Kolmogorov. Sulla determinazione empirica di una legge di distribuzione. Giornale dell’Istituto Italiano degli Attuari, 4:83–91, 1933.

Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.

Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015.

Erich Leo Lehmann and Joseph P Romano. Testing statistical hypotheses. Springer, 2005.

Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. Self-supervised learning: Generative or contrastive. IEEE transactions on knowledge and data engineering, 35(1):857–876, 2021.

Zhuang Ma and Michael Collins. Noise contrastive estimation and negative sampling for conditional models: Consistency and statistical efficiency. arXiv preprint arXiv:1809.01812, 2018.

Tobias Maltenberger, Ivan Ilic, Ilin Tolovski, and Tilmann Rabl. Evaluating multi-gpu sorting with modern interconnects. In Proceedings of the 2022 International Conference on Management of Data, pages 1795–1809, 2022.

George Marsaglia. Choosing a point from the surface of a sphere. The Annals of Mathematical Statistics, 43(2): 645–646, 1972.

Charles Masson, Jee E Rim, and Homin K Lee. Ddsketch: A fast and fully-mergeable quantile sketch with relativeerror guarantees. arXiv preprint arXiv:1908.10693, 2019.

David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. In International Conference on Artificial Intelligence and Statistics, pages 875–884. PMLR, 2020.

H Mhaskar, F Narcowich, and J Ward. Spherical marcinkiewicz-zygmund inequalities and positive quadrature. Mathematics of computation, 70(235):1113–1130, 2001.

Erik G Miller. A new class of entropy estimators for multi-dimensional densities. In 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP’03)., volume 3, pages III–297. IEEE, 2003.

Frederick Mosteller. On some useful “inefficient” statistics. Springer, 2006.

Elizbar A Nadaraya. On estimating regression. Theory of Probability & Its Applications, 9(1):141–142, 1964.

Francis J Narcowich, Pencho Petrushev, and Joseph D Ward. Localized tight frames on spheres. SIAM Journal on Mathematical Analysis, 38(2):574–594, 2006.

Jerzy Neyman and Egon Sharpe Pearson. Ix. on the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character, 231(694-706):289–337, 1933.

Khai Nguyen and Nhat Ho. Energy-based sliced wasserstein distance. Advances in Neural Information Processing Systems, 36:18046–18075, 2023.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.

Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver Deussen. Monotonic differentiable sorting networks. arXiv preprint arXiv:2203.09630, 2022.

RoL Plackett. Linear estimation from censored data. The Annals of Mathematical Statistics, 29(1):131–142, 1958.

Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of mutual in formation. In International conference on machine learning, pages 5171–5180. PMLR, 2019.

M Mahibbur Rahman and Z Govindarajulu. A modification of the test of shapiro and wilk for normality. Journal of Applied Statistics, 24(2):219–236, 1997.

Bryan Rodas, Natalie Montesino, Jakob Ambsdorf, David Klindt, and Randall Balestriero. Diet-cp: Lightweight and data efficient self supervised continued pretraining. arXiv preprint arXiv:2509.06990, 2025.

Samarendra Nath Roy. On a heuristic method of test construction and its use in multivariate analysis. The Annals of Mathematical Statistics, 24(2):220–238, 1953.

21

<!-- page 22 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986.

Claude E Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Samuel S Shapiro and RS Francia. An approximate analysis of variance test for normality. Journal of the American statistical Association, 67(337):215–216, 1972.

Samuel Sanford Shapiro and Martin B Wilk. An analysis of variance test for normality (complete samples). Biometrika, 52(3-4):591–611, 1965.

Ravid Shwartz Ziv and Yann LeCun. To compress or not to compress—self-supervised learning and information theory: A review. Entropy, 26(3):252, 2024.

Ravid Shwartz-Ziv, Randall Balestriero, and Yann LeCun. What do we maximize in self-supervised learning? arXiv preprint arXiv:2207.10081, 2022.

Bernard W Silverman. Density estimation for statistics and data analysis. Routledge, 2018.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cĳo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in artificial intelligence, pages 574–584. PMLR, 2020.

Charles Spearman. The proof and measurement of association between two things. 1961.

Bharath K Sriperumbudur, Arthur Gretton, Kenji Fukumizu, Bernhard Schölkopf, and Gert RG Lanckriet. Hilbert space embeddings and metrics on probability measures. The Journal of Machine Learning Research, 11: 1517–1561, 2010.

Shiliang Sun and Rongqing Huang. An adaptive k-nearest neighbor algorithm. In 2010 seventh international conference on fuzzy systems and knowledge discovery, volume 1, pages 91–94. IEEE, 2010.

Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin, 2 (4):160–163, 1991.

Gábor J Székely and Maria L Rizzo. A new test for multi-variate normality. Journal of Multivariate Analysis, 93(1): 58–80, 2005.

Ivan Tanasic, Lluís Vilanova, Marc Jordà, Javier Cabezas, Isaac Gelado, Nacho Navarro, and Wen-mei Hwu. Comparison based sorting for systems with multiple gpus. In Proceedings of the 6th Workshop on General Purpose Processor Using Graphics Processing Units, pages 1–11, 2013.

Kashvi Taunk, Sanjukta De, Srishti Verma, and Aleena Swetapadma. A brief review of nearest neighbor algorithm for learning and classification. In 2019 international con ference on intelligent computing and control systems (ICCS), pages 1255–1260. IEEE, 2019.

Vimal Thilak, Chen Huang, Omid Saremi, Laurent Dinh, Hanlin Goh, Preetum Nakkiran, Joshua M Susskind, and Etai Littwin. Lidar: Sensing linear probing performance in joint embedding ssl architectures. arXiv preprint arXiv:2312.04000, 2023.

Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? Advances in neural information processing systems, 33:6827–6839, 2020.

Yuandong Tian, Xinlei Chen, and Surya Ganguli. Understanding self-supervised learning dynamics without contrastive pairs. In International Conference on Machine Learning, pages 10268–10278. PMLR, 2021.

Edward C Tolman. Cognitive maps in rats and men. Psychological review, 55(4):189, 1948.

Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani, Aviv Regev, and Randall Balestriero. Joint embedding vs reconstruction: Provable benefits of latent space prediction for self supervised learning. arXiv preprint arXiv:2505.12477, 2025.

Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol, and Léon Bottou. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research, 11(12), 2010.

Huy V Vo, Vasil Khalidov, Timothée Darcet, Théo Moutakanni, Nikita Smetanin, Marc Szafraniec, Hugo Touvron, Camille Couprie, Maxime Oquab, Armand Joulin, et al. Automatic data curation for self-supervised learning: A clustering-based approach. arXiv preprint arXiv:2405.15613, 2024.

Hermann Von Helmholtz. Handbuch der physiologischen Optik, volume 9. L. Voss, 1867.

Richard Von Mises. Probability, statistics, and truth. Courier Corporation, 1981.

22

<!-- page 23 of 50 -->

LeJEPA: Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, and Xinlei Chen. On the importance of asymmetry for siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16570–16579, 2022.

Geoffrey S Watson. Smooth regression analysis. Sankhya:¯ The Indian Journal of Statistics, Series A, pages 359–372, 1964.

George S Watson. Goodness-of-fit tests on a circle. Biometrika, 48(1/2):109–114, 1961.

S Weisburg and C Binham. An approximate analysis of variance test for non-normality suitable for machine computation. Technometrics, 17:133–134, 1975.

Shichao Zhang, Xuelong Li, Ming Zong, Xiaofeng Zhu, and Ruili Wang. Efficient knn classification with different numbers of nearest neighbors. IEEE transactions on neural networks and learning systems, 29(5):1774–1785, 2017.

Yifan Zhang, Zhiquan Tan, Jingqin Yang, Weiran Huang, and Yang Yuan. Matrix information theory for self-supervised learning. arXiv preprint arXiv:2305.17326, 2023.

23

<!-- page 24 of 50 -->

## LeJEPA Appendix

## A Additional Details on Nonlinear Probing

## A.1 kNN Probing

To allow for more flexible evaluation of the pretrained encoder $f _ { \theta } ,$ it is standard to work with a 𝑘-NN prober [Taunk et al., 2019], both for regression and classification. We rely on the radial 𝑘-NN variation that leverages a sample-dependent 𝑘–improving performance for non uniform distributions of samples [Sun and Huang, 2010, Zhang et al., 2017, Abu Alfeilat et al., 2019].

We denote the underlying embedding density as $p _ { z } \in C ^ { 3 }$ with derivatives of order up to 3 bounded, and finite Fisher information and covariance. This regularity condition is fulfilled by current encoders. The unknown labels come from the target function $\eta : \mathbb { R } ^ { K } \to \mathbb { R }$ , assumed $C ^ { \check { 2 } }$ . We handle classification tasks by setting $\eta ( z ) = \mathbb { P } ( Y = 1 \mid z )$ . The training consists of the 𝑁 embeddings along with their training labels $\{ ( z _ { n } , \eta ( z _ { n } ) ) \} _ { n = 1 } ^ { N ^ { ' } } ,$ , where we will denote $\pmb { y } _ { n } \triangleq \eta ( z _ { n } )$ . The prediction for a query vector 𝒒 is formed as

$$
\widehat {y} (q) := \frac {1}{y (q)} \sum_ {n: \| z _ {n} - q \| \leq r _ {0}} y _ {n},\tag{kNN}
$$

with $\boldsymbol { y } ( \boldsymbol { q } ) \triangleq \# \{ n : \left\| \boldsymbol { z } _ { n } - \boldsymbol { q } \right\| \leq r _ { 0 } \}$ counting the number of samples within a 𝑟-radius ball around 𝒒. The radius 𝑟 controls how many neighbors predictions are averaged to form the query’s prediction. As per the linear probing’s lemma. 1, we can characterize the bias of the estimator Equation (kNN) at a particular query point, as formalized below.

## Lemma 4: k-NN Pointwise Bias

The (kNN) estimator has bias at query 𝒒 given by

$$
\text {Bias} (\boldsymbol {q}) = \frac {r _ {0} ^ {2}}{d + 2} \left(\nabla \eta (\boldsymbol {q}) ^ {\top} \nabla \log p _ {z} (\boldsymbol {q}) + \frac {1}{2} \Delta \eta (z)\right)
$$

$$
+ o (r _ {0} ^ {2}),
$$

where the remainder $o ( r _ { 0 } ^ { 2 } )$ is uniform in 𝒒. (Proof in Section B.3.)

To obtain the integrated bias, $\mathbf { i . e . } ,$ over the distribution of query points, we consider the following two properties. First, the distribution of query points follow the training distribution, $\mathrm { i . e . , } ~ \boldsymbol { q } \sim p _ { z } ,$ second, target function 𝜂 has gradient which is mean-zero and isotropic with $\mathbb { E } \big [ \nabla \eta ( z ) \nabla \eta ( z ) ^ { \top } \big ] ^ { \circ } = \tau _ { g } ^ { 2 } I _ { d }$ with $\tau _ { g } ^ { 2 } \in \dot { ( 0 , \infty ) }$ uniformly in 𝒛. We also have any finite scalar-constraint on the covariance of the embeddings such as $\operatorname { T r } ( \Sigma ) \stackrel { \circ } { = } c \; \mathbf { o r } \; \| \Sigma \| _ { F } = c$ for a finite constant 𝑐.

<!-- page 25 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

## Theorem 7: k-NN isotropic Gaussian Optimality

The integrated squared bias of (kNN) satisfies

$$
\mathbb {E} _ {z} \big [ \mathrm{Bias} (z) ^ {2} \big ] = \frac {r _ {0} ^ {4}}{(K + 2) ^ {2}} \tau_ {g} ^ {2} J (p) + O (r _ {0} ^ {4}),
$$

and the isotropic Gaussian is the unique minimizer of the integrated square bias. (Proof in Section B.4.)

As a result, we now have a unique minimizer for the optimal embedding density for both the linear and k-NN probes.

## A.2 Kernel Probing

As an alternative to (kNN), it is also common to leverage kernel methods, which we consider in this section.

Consider a kernel $K : \mathbb { R } ^ { K } \to \mathbb { R }$ with the following standard properties

$$
\int_ {\mathbb {R} ^ {d}} K (u) d u = 1,\tag{normalized}
$$

$$
\int_ {\mathbb {R} ^ {d}} u K (u) d u = 0,\tag{symmetric}
$$

$$
\int_ {\mathbb {R} ^ {d}} u u ^ {\top} K (u) d u = \mu_ {2} (K) I _ {d},\tag{isotropic}
$$

$$
R (K) \triangleq \int_ {\mathbb {R} ^ {d}} K (u) ^ {2} d u <   \infty ,\tag{finite roughness}
$$

for some $\mu _ { 2 } ( K ) \in ( 0 , \infty )$ , some bandwidth $h   >   0$ and denoting $K _ { h } ( t ) \triangleq   h ^ { - d } K ( t / h )$ , we remind the reader that the Nadaraya-Watson estimator, introduced in Nadaraya [1964], Watson [1964], at a query $\boldsymbol { q } \in \mathbb { R } ^ { d }$ is

$$
\widehat {\boldsymbol {y}} (\boldsymbol {q}) \triangleq \frac {\sum_ {n = 1} ^ {N} K _ {h} (\boldsymbol {q} - \boldsymbol {x} _ {n}) \boldsymbol {y} _ {n}}{\sum_ {n = 1} ^ {N} K _ {h} (\boldsymbol {q} - \boldsymbol {x} _ {n})}.\tag{NW}
$$

Similarly to (kNN), we will see that the performance of (NW) depends crucially on the distribution of the training points. We have access to our dataset of inputs from $p _ { z }$ and for each sample $z _ { n }$ the corresponding target is given from $\eta ( z _ { n } ) = \mathbb { E } [ Y _ { n } \mid z _ { n } ]$ . We also denote the corresponding conditional variance of the target function at that point as $v ( x ) = \operatorname { V a r } ( Y _ { i } \mid X _ { i } = x )$ . We follow the regularity conditions of the k-NN probing derivations and additionally assume that $\textit { 1 p }$ has sufficiently light tails so that for each coordinate 𝑗, $\operatorname* { l i m } _ { \| x \| \to \infty } p ( x ) = 0$ and $\operatorname* { l i m } _ { \| x \| \to \infty } x _ { j } p ( x ) = 0$ . We first derive the pointwise bias and variance for $\widehat { y } ( q )$

## Lemma 5: Kernel Bias and Variance

For any fixed $\boldsymbol { q } \in \mathbb { R } ^ { d }$ with $p ( q ) > 0 ,$ as $h \rightarrow 0$ and $n h ^ { d } \to \infty ,$

$$
\mathrm{Bias} \big [ \widehat {\pmb {y}} (\pmb {q}) \big ] = \frac {h ^ {2} \mu_ {2} (K)}{2} \Big (\Delta \pmb {y} (\pmb {q}) + 2 \nabla \pmb {y} (\pmb {q}) ^ {\top} \nabla \log p (\pmb {q}) \Big) + o (h ^ {2}),
$$

$$
\mathrm{Var} \big [ \widehat {\boldsymbol {y}} (\boldsymbol {q}) \big ] = \frac {R (K)}{n h ^ {d}} \frac {v (\boldsymbol {q})}{p (\boldsymbol {q})} + o \big ((n h ^ {d}) ^ {- 1} \big).
$$

The $o ( \cdot )$ terms are uniform over compact sets where $p$ is bounded away from zero. (Proof in Section B.5.)

We now show that, under a fixed mean and total-covariance constraint on $p _ { z _ { 1 } }$ , the isotropic Gaussian distribution uniquely minimizes the bias and variance of the kernel regression estimator at any test point. We restrict the smoothness class of the target function using

$$
\mathcal {M} (L, B) \triangleq \left\{m \in C ^ {2} (\mathbb {R} ^ {d}): \| \nabla \boldsymbol {y} (\boldsymbol {q}) \| \leq L, \right.
$$

$$
\left| \Delta \boldsymbol {y} (\boldsymbol {q}) \right| \leq B, \forall \boldsymbol {q} \in \mathbb {R} ^ {d} \Bigg \},
$$

25

<!-- page 26 of 50 -->

LeJEPA: Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

allowing us to formalize below the worst case integrated bias and the optimal density for 𝑧.

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Theorem 8: Kernel isotropic Gaussian Optimality
The integrated squared bias of (NW) satisfies
$\sup_{m \in \mathcal{M}(L,B)} \mathbb{E}_z \left[ \text{Bias}\left[\widehat{\boldsymbol{y}}(z)\right] \right] \leq \left( \frac{h^2 \mu_2(K)}{2} \right)^2 \left( 2B^2 + 8L^2 J(p) \right) + o(h^4)$,
and the integrated variance is independent of $p$. Among all densities $p$ on $\mathbb{R}^d$ with total-variance constrained, e.g., $\text{Tr}(\Sigma) = c$, the isotropic Gaussian is the unique minimizer. (Proof in Section B.7.)
</div>

## B Proofs

## B.1 Proof of lemma. 1

Proof. Our proof follows standard derivations when it comes to studying the bias of an estimator. Let’s consider the ridge regression problem (Tikhonov regularized least squares estimator) with close form estimator

$$
\hat {\boldsymbol {\beta}} = (\mathbf {X} ^ {T} \mathbf {X} + \lambda_ {\mathrm{wd}} \mathbf {I}) ^ {- 1} \mathbf {X} ^ {T} \mathbf {Y}.\tag{9}
$$

The labels are formed from the ground truth parameter $\beta _ { \mathrm { t r u e } }$ with centered error, as per $\mathbf { Y } = \mathbf { X } \mathbf { \beta } _ { \mathrm { t r u e } } + \mathbf { \varepsilon } \; \mathrm { w h e r e } \; \mathbb { E } [ \mathbf { \varepsilon } ] = \mathbf { 0 }$ We can now look at the bias of our estimator given by

$$
\begin{array}{r l} \mathrm{Bias} (\hat {\pmb {\beta}}) & = \mathbb {E} [ \hat {\pmb {\beta}} ] - \pmb {\beta} _ {\mathrm{true}} \\ & = (\mathbf {X} ^ {T} \mathbf {X} + \lambda_ {\mathrm{wd}} \mathbf {I}) ^ {- 1} \mathbf {X} ^ {T} \mathbf {X} \pmb {\beta} _ {\mathrm{true}} - \pmb {\beta} _ {\mathrm{true}} \\ & = - \lambda_ {\mathrm{wd}} (\mathbf {X} ^ {T} \mathbf {X} + \lambda_ {\mathrm{wd}} \mathbf {I}) ^ {- 1} \pmb {\beta} _ {\mathrm{true}} \\ & = - \lambda_ {\mathrm{wd}} \mathbf {Q} (\pmb {\Lambda} + \lambda \mathbf {I}) ^ {- 1} \mathbf {Q} ^ {T} \pmb {\beta} _ {\mathrm{true}} \end{array}
$$

We will now compare that bias when 𝑿 has isotropic and anisotropic covariance with same total variance:

$$
\frac {\lambda_ {1} + \lambda_ {2} + \cdots + \lambda_ {p}}{p} = \bar {\lambda}.\tag{10}
$$

For any anisotropic covariance matrix of $X ,$ denote by $q _ { 1 }$ the eigenvector with smallest eigenvalue, and let’s denote by $\kappa > 0$ a positive constant. We now define

$$
\boldsymbol {\beta} _ {\mathrm{true}} = \boldsymbol {\kappa} \cdot \mathbf {q} _ {p},\tag{11}
$$

leading to

$$
\begin{array}{c} \| \mathrm{Bias} (\hat {\boldsymbol {\beta}}) \| _ {\mathrm{isotropic}} = \frac {\lambda_ {\mathrm{wd}}}{\bar {\lambda} + \lambda_ {\mathrm{wd}}} \| \boldsymbol {\beta} _ {\mathrm{true}} \|, \\ \| \mathrm{Bias} (\hat {\boldsymbol {\beta}}) \| _ {\mathrm{non-isotropic}} = \frac {\lambda_ {\mathrm{wd}}}{\lambda_ {p} + \lambda_ {\mathrm{wd}}} \| \boldsymbol {\beta} _ {\mathrm{true}} \| \end{array}
$$

Since $\lambda _ { p } < \bar { \lambda }$ (strict inequality when not isotropic):

$$
\frac {\lambda_ {\mathrm{wd}}}{\lambda_ {p} + \lambda_ {\mathrm{wd}}} > \frac {\lambda_ {\mathrm{wd}}}{\bar {\lambda} + \lambda_ {\mathrm{wd}}}
$$

we obtain that

$$
\| \text {Bias} (\hat {\boldsymbol {\beta}}) \| _ {\text {non - isotropic}} > \| \text {Bias} (\hat {\boldsymbol {\beta}}) \| _ {\text {isotropic}}
$$

As a result, whenever the covariance matrix of 𝑿 is anisotropic, there will be downstream tasks for which the estimator bias is increased compared to having isotropic covariance matrix. Anisotropic covariance structure thus amplifies regularization bias when the true parameter vector aligns unfavorably with the data’s covariance structure. □

26

<!-- page 27 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

## B.2 Proof of lemma. 2

Proof. We use the same formula as in Section B.1 with $\lambda _ { \mathrm { w d } } = 0$ . We first see that the estimator is unbiased. We will now leverage that result to compute the covariance matrix of the estimator

$$
\left| \begin{array}{l} \mathrm{Var} (\hat {\boldsymbol {\beta}} | \mathbf {X}) = \mathbb {E} [ (\hat {\boldsymbol {\beta}} - \boldsymbol {\beta}) (\hat {\boldsymbol {\beta}} - \boldsymbol {\beta}) ^ {T} | \mathbf {X} ] \\ \quad = \mathbb {E} [ (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \mathbf {X} ^ {T} \varepsilon \varepsilon^ {T} \mathbf {X} (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} | \mathbf {X} ] \\ \quad = (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \mathbf {X} ^ {T} \mathbb {E} [ \varepsilon \varepsilon^ {T} | \mathbf {X} ] \mathbf {X} (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \\ \quad = (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \mathbf {X} ^ {T} (\sigma^ {2} \mathbf {I} _ {n}) \mathbf {X} (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \\ \quad = \sigma^ {2} (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \end{array} \right|
$$

leading to the total variance

$$
\mathrm{tr} (\mathrm{Var} (\hat {\boldsymbol {\beta}})) = \sigma^ {2} \mathrm{tr} (\mathbf {G} ^ {- 1}) = \sigma^ {2} \sum_ {j = 1} ^ {p} \frac {1}{\lambda_ {j}}
$$

where we used the eigendecomposition:

$$
\mathbf {G} = \mathbf {Q} \boldsymbol {\Lambda} \mathbf {Q} ^ {T}
$$

The function $\begin{array} { r } { f ( x ) = \frac { 1 } { x } } \end{array}$ is strictly convex on $( 0 , \infty )$ allowing us to leverage Jensen’s Inequality:

$$
\begin{array}{c} \frac {1}{K} \sum_ {k = 1} ^ {K} \frac {1}{\lambda_ {k}} > \frac {1}{\frac {1}{K} \sum_ {j = 1} ^ {K} \lambda_ {k}} \\ \Longleftrightarrow \frac {1}{K} \sum_ {k = 1} ^ {K} \frac {1}{\lambda_ {k}} > \frac {1}{K} \sum_ {k = 1} ^ {K} \frac {1}{\frac {1}{K} \sum_ {j = 1} ^ {K} \lambda_ {k}} \\ \Longleftrightarrow \sum_ {k = 1} ^ {K} \frac {1}{\lambda_ {k}} > \sum_ {k = 1} ^ {K} \frac {1}{\frac {1}{K} \sum_ {j = 1} ^ {K} \lambda_ {k}} \\ \Longleftrightarrow \operatorname{tr} (\operatorname{Var} (\hat {\boldsymbol {\beta}})) _ {\text {aniso}} > \operatorname{tr} (\operatorname{Var} (\hat {\boldsymbol {\beta}})) _ {\text {iso}} \end{array}
$$

The inequality is strict whenever the eigenvalues $\left\{ \lambda _ { j } \right\} _ { j = 1 } ^ { p }$ are not all equal.

## B.3 Proof of lemma. 4

Proof. Under PPP, conditional expectations of $\widehat { \eta } ( x )$ coincide with the normalized ball average

$$
\mathbb {E} \big [ \widehat {\eta} (x) \big ] = \frac {\int_ {\mathrm{B} (0 , r _ {0})} \eta (x + z) p (x + z) d z}{\int_ {\mathrm{B} (0 , r _ {0})} p (x + z) d z} \text {to second order in} r _ {0},
$$

which is the key surrogate used below. **Ball integrals.** For computations we use (by symmetry) for any $r > 0 ;$

$$
\int_ {\mathrm{B} (0, r)} z d z = 0, \qquad \int_ {\mathrm{B} (0, r)} z z ^ {\top} d z = \frac {\mathrm{Vol} ^ {d + 2}}{d + 2} I _ {d}, \qquad \int_ {\mathrm{B} (0, r)} \| z \| ^ {2} d z = \frac {d \mathrm{Vol} ^ {d + 2}}{d + 2}.
$$

Fix $x \in \mathbb { R } ^ { d }$ and write $z \in \mathrm { B } ( 0 , r _ { 0 } )$ for local displacements. Assume $p \in C ^ { 3 } ,   \eta \in C ^ { 2 }$ with bounded derivatives on the region of interest, and expand a second-order Taylor expansion:

$$
\begin{array}{r l} & p (x + z) = p (x) + \nabla p (x) ^ {\top} z + \frac {1}{2} z ^ {\top} H p (x) z + O (\| z \| ^ {3}), \\ & \eta (x + z) = \eta (x) + \nabla \eta (x) ^ {\top} z + \frac {1}{2} z ^ {\top} H \eta (x) z + O (\| z \| ^ {3}), \end{array}
$$

with remainders satisfying $\left| R _ { \eta } ( x ; z ) \right| \leq C _ { \eta } \left\| z \right\| ^ { 3 }$ and $\left| R _ { p } ( x ; z ) \right| \leq C _ { p } \left\| z \right\| ^ { 3 }$ uniformly for $\| z \| \leq r _ { 0 }$ . Using the ball identities

27

<!-- page 28 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

$\begin{array} { r } { \int _ { B ( 0 , r ) } z d z = 0 } \end{array}$ and $\begin{array} { r } { \int _ { B ( 0 , r ) } z z ^ { \top } d z = \frac { v _ { d } r ^ { d + 2 } } { d + 2 } I _ { d } } \end{array}$ and collecting terms up to order $r _ { 0 } ^ { d + 2 }$ , we simplify the denominator as

$$
\begin{array}{r l} \mathcal {D} (x) & \triangleq \int_ {\mathrm{B} (0, r _ {0})} p (x + z) d z \\ & = \int_ {\mathrm{B} (0, r _ {0})} \left[ p (x) + \nabla p (x) ^ {\top} z + \frac {1}{2} z ^ {\top} H p (x) z + R _ {p} (x; z) \right] d z \\ & = \mathrm{Vol} _ {0} ^ {d} p (x) + \frac {\mathrm{Vol} _ {0} ^ {d + 2}}{2 (d + 2)} \mathrm{tr} \big (H p (x) \big) + O (r _ {0} ^ {d + 3}), \end{array}
$$

since $\begin{array} { r } { \int z d z = 0 } \end{array}$ and $\begin{array} { r } { \int z ^ { \top } H p z d z = \mathrm { t r } ( H p ) \frac { v _ { d } r _ { 0 } ^ { d + 2 } } { d + 2 } } \end{array}$ and the denominator as

$$
\begin{array}{r l} \mathcal {N} (x) \triangleq & \int_ {\mathrm{B} (0, r _ {0})} \eta (x + z) p (x + z) d z \\ & = \int \left[ \eta (x) + \nabla \eta (x) ^ {\top} z + \frac {1}{2} z ^ {\top} H \eta (x) z \right] \left[ p (x) + \nabla p (x) ^ {\top} z + \frac {1}{2} z ^ {\top} H p (x) z \right] d z + O (r _ {0} ^ {d + 3}) \\ & = \eta (x) p (x) v _ {d} r _ {0} ^ {d} + \eta (x) \frac {v _ {d} r _ {0} ^ {d + 2}}{2 (d + 2)} \mathrm{tr} \big (H p (x) \big) + \frac {v _ {d} r _ {0} ^ {d + 2}}{d + 2} \nabla \eta (x) \cdot \nabla p (x) + \frac {v _ {d} r _ {0} ^ {d + 2}}{2 (d + 2)} p (x) \mathrm{tr} \big (H \eta (x) \big) + O (r _ {0} ^ {d + 3}). \end{array}
$$

Cubic terms vanish by symmetry, and quartic terms are $O ( r _ { 0 } ^ { d + 4 } )$ . Subtract $\eta ( x ) \mathcal { D } ( x )$ to obtain the bias numerator:

$$
\mathcal {N} (x) - \eta (x) \mathcal {D} (x) = \frac {v _ {d} r _ {0} ^ {d + 2}}{d + 2} \Big (\nabla \eta (x) \cdot \nabla p (x) + \frac {1}{2} p (x) \Delta \eta (x) \Big) + O (r _ {0} ^ {d + 3}).
$$

Write $\mathcal { D } ( x ) = v _ { d } r _ { 0 } ^ { d } p ( x ) \big ( 1 + \alpha ( x ) r _ { 0 } ^ { 2 } + O ( r _ { 0 } ^ { 3 } ) \big )$ where $\begin{array} { r } { \alpha ( x ) : = \frac { 1 } { 2 ( d + 2 ) p ( x ) } \mathrm { t r } ( H p ( x ) ) } \end{array}$ . Then

$$
\begin{array}{r l} & {\frac {\mathcal {N} (x)}{\mathcal {D} (x)} - \eta (x) = \frac {\frac {v _ {d} r _ {0} ^ {d + 2}}{d + 2} \left(\nabla \eta \cdot \nabla p + \frac {1}{2} p \Delta \eta\right) + O (r _ {0} ^ {d + 3})}{v _ {d} r _ {0} ^ {d} p \left(1 + \alpha r _ {0} ^ {2} + O (r _ {0} ^ {3})\right)}} \\ & {\qquad = \frac {r _ {0} ^ {2}}{d + 2} \left(\frac {\nabla \eta \cdot \nabla p}{p} + \frac {1}{2} \Delta \eta\right) \left(1 - \alpha r _ {0} ^ {2} + O (r _ {0} ^ {3})\right) + O (r _ {0} ^ {3})} \\ & {\qquad = \frac {r _ {0} ^ {2}}{d + 2} \Big (\nabla \eta (x) \cdot \nabla \log p (x) + \frac {1}{2} \Delta \eta (x) \Big) + o (r _ {0} ^ {2}),} \end{array}
$$

uniformly on $\mathcal { K } .$ This gives the bias formula

$$
\mathbb {E} \left[ \widehat {\eta} (x) \right] - \eta (x) = \frac {r _ {0} ^ {2}}{d + 2} \Big (\nabla \eta (x) \cdot \nabla \log p (x) + \frac {1}{2} \Delta \eta (x) \Big) + o (r _ {0} ^ {2}),
$$

completing the proof.

## B.4 Proof of thm. 7

Proof. Recall from Section B.3 that the bias term as sample 𝒙 is given by

$$
\begin{array}{r l} \mathrm{Bias} (x) & = \frac {r _ {0} ^ {2}}{d + 2} \Big (\nabla \eta (x) \cdot \nabla \log p (x) \Big) + \frac {r _ {0} ^ {2}}{2 (d + 2)} \Delta \eta (x) + o (r _ {0} ^ {2}) \\ & = \frac {r _ {0} ^ {2}}{d + 2} \big (A (x) + C (x) \big) + o (r _ {0} ^ {2}), \end{array}
$$

28

<!-- page 29 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

where we defined $A ( x ) \triangleq \nabla \eta ( x )$ · ∇ log $p ( x )$ and $\begin{array} { r } { C ( x ) \triangleq \frac { 1 } { 2 } \Delta \eta ( x ) } \end{array}$ . We now square and take expectation of $X \sim p$ and the isotropic gradient prior

$$
\begin{array}{r l} & {\mathbb {E} \big [ \mathrm{Bias} (X) ^ {2} \big ] = \mathbb {E} \big [ \left(\frac {r _ {0} ^ {2}}{d + 2}\right) ^ {2} \big (A (x) ^ {2} + 2 A (x) C (x) + C (x) ^ {2} \big) + o (r _ {0} ^ {4}) \big ]} \\ & {\qquad = \left(\frac {r _ {0} ^ {2}}{d + 2}\right) ^ {2} \Big \{\underbrace {\mathbb {E} \big [ A (X) ^ {2} \big ]} _ {\mathrm{score-gradientterm}} + \underbrace {2 \mathbb {E} \big [ A (X) C (X) \big ]} _ {\mathrm{crossterm}} + \underbrace {\mathbb {E} \big [ C (X) ^ {2} \big ]} _ {\mathrm{curvatureterm}} \Big \} + o (r _ {0} ^ {4}).} \end{array}\tag{12}
$$

(13)

We will derive each term separately, recalling that we assume an isotropic gradient prior for $\eta , \mathrm { i . e . } , \mathbb { E } \big [ \nabla \eta ( x ) \big ] = 0$ and $\mathbb { E } \big [ \nabla \eta ( x ) \nabla \eta ( x ) ^ { \top } \big ] = \tau _ { g } ^ { 2 } I _ { d } ,$ , for some $\tau _ { g } ^ { 2 } \in ( 0 , \infty )$

**1) The score-gradient term** $\mathbb { E } [ A ( X ) ^ { 2 } ]$ . Using $v ( x ) : = \nabla \log p ( x )$ for brevity:

$$
\begin{array}{r l} \mathbb {E} \big [ A (X) ^ {2} \big ] & = \mathbb {E} _ {X} \left[ \mathbb {E} _ {\eta} [ A (X) ^ {2} ] \right] \\ & = \mathbb {E} _ {X} \left[ \mathbb {E} _ {\eta} \big [ \big (\nabla \eta (x) ^ {\top} v (x) \big) ^ {2} \big ] \right] \\ & = \mathbb {E} _ {X} \left[ \mathbb {E} _ {\eta} [ \nabla \eta (x) ^ {\top} \Big (v (x) v (x) ^ {\top} \Big) \nabla \eta (x) ] \right] \\ & = \mathbb {E} _ {X} \left[ \mathbb {E} _ {\eta} [ \mathrm{tr} \Big (v (x) v (x) ^ {\top} \nabla \eta (x) \nabla \eta (x) ^ {\top} \Big) ] \right] \\ & = \mathbb {E} _ {X} \left[ \mathrm{tr} \Big (v (x) v (x) ^ {\top} \mathbb {E} _ {\eta} [ \nabla \eta (x) \nabla \eta (x) ^ {\top} ] \Big) \right] \\ & = \mathbb {E} _ {X} \left[ \tau_ {g} ^ {2} \| v (x) \| ^ {2} \right] \\ & = \tau_ {g} ^ {2} \mathbb {E} _ {X} \left[ \| v (X) \| ^ {2} \right] \\ & = \tau_ {g} ^ {2} \int_ {\mathbb {R} ^ {d}} \| \nabla \log p (x) \| ^ {2} p (x) d x \end{array}
$$

recovering the Fisher-information functional $J ( p )$ , scaled by $\tau _ { g } ^ { 2 }$

**2) The cross term** 2E[𝐴(𝑋)𝐶(𝑋)]. We have

$$
A (x) C (x) = \frac {1}{2} \big (\nabla \eta (x) ^ {\top} v (x) \big) \Delta \eta (x).
$$

Under the prior, $\nabla \eta$ is mean-zero and isotropic; if, additionally, $\Delta \eta$ is uncorrelated with $\nabla \eta$ and has zero mean (or is bounded and mean-zero after centering), then $\mathbb { E } _ { \eta } [ A ( x ) C ( x ) ] = 0$ . If one does not assume the orthogonality/vanishing covariance above, then $\mathbb { E } [ A ( X ) C ( X ) ]$ is a finite constant (depending on the joint law of derivatives of $\eta ) ,$ , and the cross term contributes

$$
\left(\frac {r _ {0} ^ {2}}{d + 2}\right) ^ {2} \cdot 2 \mathbb {E} [ A (X) C (X) ] = O (r _ {0} ^ {4}),
$$

not $o ( r _ { 0 } ^ { 4 } )$ . In that general case, the leading 𝑝-dependent term of $\mathbb { E } [ \mathrm { B i a s } ( X ) ^ { 2 } ]$ is still the score-gradient $\tau _ { g } ^ { 2 } J ( p )$

**3) The curvature term** $\mathbb { E } [ C ( X ) ^ { 2 } ]$

$$
\left| \begin{array}{l} \mathbb {E} \big [ C (X) ^ {2} \big ] = \mathbb {E} _ {X} \left[ \mathbb {E} _ {\eta} [ C (X) ^ {2} ] \right] \\ = \frac {1}{4} \mathbb {E} _ {X} \left[ \mathbb {E} _ {\eta} [ (\Delta \eta (X)) ^ {2} \right] \end{array} \right|
$$

which is independent of $p ,$ hence $\mathbb { E } \big [ C ( X ) ^ { 2 } \big ] = O ( 1 )$

29

<!-- page 30 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

**Putting it together.** Substituting into (13):

$$
\begin{array}{r l} & {\mathbb {E} \big [ \mathrm{Bias} (X) ^ {2} \big ] = \left(\frac {r _ {0} ^ {2}}{d + 2}\right) ^ {2} \left\{\tau_ {g} ^ {2} J (p) + O (1) \right\} + o (r _ {0} ^ {4})} \\ & {\qquad = \frac {r _ {0} ^ {4}}{(d + 2) ^ {2}} \tau_ {g} ^ {2} J (p) + O (r _ {0} ^ {4}),} \end{array}
$$

We show that, among all mean-zero distributions $p$ on $\mathbb { R } ^ { d }$ with a given scalar constraint on the covariance (trace, determinant, Frobenius norm, or spectral radius), the density that minimizes the Fisher-information functional

$$
J (p) := \int_ {\mathbb {R} ^ {d}} \| \nabla \log p (x) \| ^ {2} p (x) d x
$$

is the Gaussian with isotropic covariance satisfying the same scalar constraint. We proceed in two steps: (i) for fixed covariance matrix $\Sigma > 0 , J ( p )$ is minimized by the Gaussian $\mathcal { N } ( 0 , \Sigma )$ and attains the value $\mathbf { t r } ( \Sigma ^ { - 1 } ) ;$ (ii) for each scalar constraint, $\mathbf { t r } ( \Sigma ^ { - 1 } )$ is minimized by $\Sigma = s I _ { d }$ for the appropriate scalar $s > 0$

## Lemma 6: Special case: Recovery of VCReg

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Let $p$ be a mean-zero probability density on $\mathbb{R}^d$ with covariance $\Sigma = \mathbb{E}[XX^\top] &gt; 0$. Then
$J(p) \geq \text{tr}(\Sigma^{-1})$,
with equality if and only if $p = \mathcal{N}(0, \Sigma)$.
</div>

Proof. Consider the location family $p _ { \theta } ( x ) : = p ( x - \theta ) , \theta \in \mathbb { R } ^ { d }$ . Its Fisher-information matrix at $\theta$ is

$$
\mathcal {I} (\theta) = \mathbb {E} \bigl [ \nabla_ {\theta} \log p _ {\theta} (X) \nabla_ {\theta} \log p _ {\theta} (X) ^ {\top} \bigr ] = \mathbb {E} \bigl [ \nabla \log p (X) \nabla \log p (X) ^ {\top} \bigr ],
$$

so that $J ( p ) = \operatorname { t r } { \mathcal { I } } ( \theta )$ . The estimator $T ( X ) \equiv X$ is unbiased for 𝜃 under $p _ { \theta } ,$ , with $\operatorname { C o v } ( T ) = \Sigma$ . The matrix Cramér–Rao bound gives $\mathsf { C o v } ( T ) \succeq \mathcal { I } ( \theta ) ^ { - 1 } , \mathrm { i . e . } , \mathcal { I } ( \theta ) \succeq \Sigma ^ { - 1 }$ . Taking traces yields $J ( p ) \geq \operatorname { \mathsf { t r } } ( \Sigma ^ { - 1 } )$ . Equality in the matrix Cramér–Rao bound holds if and only if the score is an affine function of $X - \theta , \mathrm { i . e . } , \nabla \log p _ { \theta } ( X ) = A ( X - \theta )$ a.s. for some matrix $A ;$ integrating this identity shows $p _ { \theta }$ is Gaussian with precision matrix $- A _ { , }$ , hence $p = \mathcal { N } ( 0 , \Sigma )$ □

## Step 2: Optimizing over covariance shapes under scalar constraints

Write the eigenvalues of Σ as $\lambda_{1}, \cdots, \lambda_{d} > 0$ . Then

$$
\mathrm{tr} (\Sigma^ {- 1}) = \sum_ {i = 1} ^ {d} \frac {1}{\lambda_ {i}}.
$$

We now solve min $\textstyle \sum _ { i } 1 / \lambda _ { i }$ under each scalar constraint; in every case the minimum is attained when all $\lambda _ { i }$ are equal, $\mathbf { i . e . } ,$ , $\Sigma = s I _ { d }$

**(a) Trace constraint.** Given $\begin{array} { r } { \operatorname { \mathsf { t r } } ( \Sigma ) = \sum _ { i } \lambda _ { i } = t > 0 , } \end{array}$ by Cauchy–Schwarz,

$$
\left(\sum_ {i = 1} ^ {d} \frac {1}{\lambda_ {i}}\right) \left(\sum_ {i = 1} ^ {d} \lambda_ {i}\right) \geq \left(\sum_ {i = 1} ^ {d} 1\right) ^ {2} = d ^ {2},
$$

with equality if and only if $\lambda _ { 1 } = \dots = \lambda _ { d } .$ Hence

$$
\min _ {\Sigma > 0: \operatorname{tr} (\Sigma) = t} \operatorname{tr} (\Sigma^ {- 1}) = \frac {d ^ {2}}{t}, \quad \text {attained at} \quad \Sigma = \frac {t}{d} I _ {d}.
$$

30

<!-- page 31 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

**(b) Determinant constraint.** Given $\begin{array} { r } { \operatorname* { d e t } ( \Sigma )   =   \prod _ { i } \lambda _ { i }   =   \delta   >   0 , } \end{array}$ set $\mu _ { i } : = 1 / \lambda _ { i }$ so that $\textstyle \prod _ { i } \mu _ { i }   =   \delta ^ { - 1 }$ . By the AM–GM inequality,

$$
\frac {1}{d} \sum_ {i = 1} ^ {d} \mu_ {i} \geq \left(\prod_ {i = 1} ^ {d} \mu_ {i}\right) ^ {1 / d} = \delta^ {- 1 / d},
$$

with equality iff $\mu _ { 1 } = \cdots = \mu _ { d } , \mathrm { i . e . } ,   \lambda _ { 1 } = \cdots = \lambda _ { d }$ . Thus

$$
\min _ {\Sigma > 0: \det (\Sigma) = \delta} \operatorname{tr} (\Sigma^ {- 1}) = d \delta^ {- 1 / d}, \quad \text {attained at} \quad \Sigma = \delta^ {1 / d} I _ {d}.
$$

**(c) Frobenius-norm constraint.** Given $\begin{array} { r } { \| \Sigma \| _ { F } ^ { 2 }   =   \sum _ { i } \lambda _ { i } ^ { 2 }   =   c ^ { 2 }   >   0 , } \end{array}$ minimize $\textstyle f ( \lambda ) : =   \sum _ { i } 1 / \lambda _ { i }$ over $\lambda _ { i }   >   0$ subject to $\begin{array} { r } { g ( \lambda ) : = \sum _ { i } \lambda _ { i } ^ { 2 } = c ^ { 2 } } \end{array}$ . The Lagrangian

$$
\mathcal {L} (\lambda , \nu) = \sum_ {i = 1} ^ {d} \frac {1}{\lambda_ {i}} + \nu \left(\sum_ {i = 1} ^ {d} \lambda_ {i} ^ {2} - c ^ {2}\right)
$$

has first-order conditions $- \lambda _ { i } ^ { - 2 } + 2 \nu \lambda _ { i } = 0$ for all $\begin{array} { r } { i , \mathrm { i . e . } , \lambda _ { i } ^ { 3 } = \frac { 1 } { 2 \nu } } \end{array}$ , so all $\lambda _ { i }$ are equal. Imposing $\begin{array} { r } { \sum \lambda _ { i } ^ { 2 } = c ^ { 2 } \; \mathrm { y i e l d s } \; \lambda _ { i } = c / \sqrt { d } , } \end{array}$ hence

$$
\min _ {\Sigma > 0: \| \Sigma \| _ {F} = c} \operatorname{tr} \left(\Sigma^ {- 1}\right) = \sum_ {i = 1} ^ {d} \frac {1}{\lambda_ {i}} = \frac {d ^ {3 / 2}}{c}, \quad \text {attained at} \quad \Sigma = \frac {c}{\sqrt {d}} I _ {d}.
$$

**(d) Spectral-radius constraint.** Let the spectral radius be constrained by $\rho ( \Sigma ) = \operatorname { \mathsf { m a x } } _ { i } \lambda _ { i } \leq r$ for some $r > 0 .$ Since $x \mapsto 1 / x$ is strictly decreasing on (0, ∞),

$$
\sum_ {i = 1} ^ {d} \frac {1}{\lambda_ {i}} \geq \sum_ {i = 1} ^ {d} \frac {1}{r} = \frac {d}{r},
$$

with equality if and only if $\lambda _ { i } = r$ for all 𝑖. Therefore

$$
\min _ {\Sigma > 0: \rho (\Sigma) \leq r} \operatorname{tr} (\Sigma^ {- 1}) = \frac {d}{r}, \quad \text {attained at} \quad \Sigma = r I _ {d}.
$$

(The same conclusion holds if the constraint is $\rho ( \Sigma ) = r ,$ since one may take all eigenvalues equal to 𝑟.)

## Conclusion: Isotropic Gaussian is optimal

Combining Lemma 6 with the solutions (a)–(d), we obtain:

31

<!-- page 32 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Theorem 9: Special case: Recovery of VCReg
Fix one of the following scalar covariance constraints for a mean-zero distribution $p$ on $\mathbb{R}^d$:
    trace: tr(Cov(X)) = $t$,
    determinant: det(Cov(X)) = $\delta$,
    Frobenius norm: $\|\text{Cov}(X)\|_F = c$,
    spectral radius upper bound: $\rho(\text{Cov}(X)) \leq r$.
Then the Fisher-information functional $J(p)$ is minimized over all such $p$ by the isotropic Gaussian $p_G = \mathcal{N}(0, sI_d)$ with $s$ chosen to satisfy the constraint. The minimal values are:
    trace $t: \quad J_{\min} = \frac{d^2}{t}, \quad s = \frac{t}{d}$,
    determinant $\delta: \quad J_{\min} = d\delta^{-1/d}, \quad s = \delta^{1/d}$,
    Frobenius $c: \quad J_{\min} = \frac{d^{3/2}}{c}, \quad s = \frac{c}{\sqrt{d}}$,
    spectral radius $r: \quad J_{\min} = \frac{d}{r}, \quad s = r$.
In each case, $p_G$ is the unique minimizer (up to null sets).
</div>

Proof. For any admissible $p$ with covariance $\Sigma ,$ Lemma 6 gives $J ( p ) \geq \operatorname { \mathsf { t r } } ( \Sigma ^ { - 1 } )$ . Minimizing the right-hand side under the stated scalar constraint yields $\Sigma = s I _ { d }$ by the calculations in $( \mathsf { a } ) { - } ( \mathsf { d } )$ . Equality in Lemma 6 holds if and only if $p$ is Gaussian with that covariance, hence $p _ { G }$ uniquely attains the bound. □

## B.5 Proof of lemma. 5

Proof. Write the numerator and denominator of ${ \widehat { m } } ( x )$ as

$$
B _ {n} (x) := \sum_ {i = 1} ^ {n} K _ {h} (x - X _ {i}) Y _ {i}, \quad A _ {n} (x) := \sum_ {i = 1} ^ {n} K _ {h} (x - X _ {i}),
$$

so that $\begin{array} { r } { \widehat { m } ( x ) = \frac { B _ { n } ( x ) } { A _ { n } ( x ) } } \end{array}$ . Bias. Compute expectations using independence and change of variables. For the denominator,

$$
\begin{array}{l} \mathbb {E} [ A _ {n} (x) ] = n \mathbb {E} \big [ K _ {h} (x - X) \big ] \\ \qquad = n \int_ {\mathbb {R} ^ {d}} h ^ {- d} K \Big (\frac {x - u}{h} \Big) p (u) d u \\ \qquad = n \int_ {\mathbb {R} ^ {d}} K (t) p (x - h t) d t \qquad (t := (x - u) / h) \\ \qquad = n \int_ {\mathbb {R} ^ {d}} K (t) \Big (p (x) - h t ^ {\top} \nabla p (x) + \frac {h ^ {2}}{2} t ^ {\top} \nabla^ {2} p (x) t + o (h ^ {2}) \Big) d t \\ \qquad = n \Big (p (x) + \frac {h ^ {2}}{2} \underbrace {\int t ^ {\top} \nabla^ {2} p (x) t K (t) d t} _ {= \mu_ {2} (K) \Delta p (x)} + o (h ^ {2}) \Big), \end{array}
$$

32

<!-- page 33 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

where we used symmetry $\begin{array} { r } { \int t K ( t ) d t \; = \; 0 } \end{array}$ and isotropy $\begin{array} { r } { \int t t ^ { \top } K ( t ) d t \; = \; \mu _ { 2 } ( K ) I _ { d } } \end{array}$ , which implies $\begin{array} { r l } { \int t ^ { \top } \nabla ^ { 2 } p ( x ) t K ( t ) d t \; = } & { { } } \end{array}$ $\mu _ { 2 } ( K ) \mathrm { t r } ( \nabla ^ { 2 } p ( x ) ) = \mu _ { 2 } ( K ) \Delta p ( \mathring { x ) }$ . Similarly, for the numerator,

$$
\begin{array}{r l} & {\mathbb {E} [ B _ {n} (x) ] = n \mathbb {E} \big [ K _ {h} (x - X) Y \big ] = n \int K (t) (m p) (x - h t) d t} \\ & {\qquad = n \int K (t) \Big ((m p) (x) - h t ^ {\top} \nabla (m p) (x) + \frac {h ^ {2}}{2} t ^ {\top} \nabla^ {2} (m p) (x) t + o (h ^ {2}) \Big) d t} \\ & {\qquad = n \Big (m (x) p (x) + \frac {h ^ {2}}{2} \mu_ {2} (K) \mathrm{tr} \big (\nabla^ {2} (m p) (x) \big) + o (h ^ {2}) \Big)} \\ & {\qquad = n \Big (m (x) p (x) + \frac {h ^ {2} \mu_ {2} (K)}{2} \big (p \Delta m + m \Delta p + 2 \nabla m ^ {\top} \nabla p \big) (x) + o (h ^ {2}) \Big),} \end{array}
$$

where the last step uses the fact that t $\mathbf { r } { \big ( } \nabla ^ { 2 } ( m p ) { \big ) } = p \Delta m + m \Delta p + 2 \nabla m ^ { \top } \nabla p$ by the product rule and symmetry of mixed derivatives.

Now expand the ratio $\frac { \mathbb { E } [ B _ { n } ( x ) ] } { \mathbb { E } [ A _ { n } ( x ) ] }$ using the identity

$$
\frac {a _ {0} + h ^ {2} a _ {2} + o (h ^ {2})}{b _ {0} + h ^ {2} b _ {2} + o (h ^ {2})} = \frac {a _ {0}}{b _ {0}} + h ^ {2} \frac {a _ {2} b _ {0} - a _ {0} b _ {2}}{b _ {0} ^ {2}} + o (h ^ {2}),
$$

with $\begin{array} { r } { a _ { 0 } = m ( x ) p ( x ) ,   a _ { 2 } = \frac { \mu _ { 2 } ( K ) } { 2 } \big ( p \Delta m + m \Delta p + 2 \nabla m ^ { \top } \nabla p \big ) ( x ) ,   b _ { 0 } = p ( x ) ,   \mathrm { a n d }   b _ { 2 } = \frac { \mu _ { 2 } ( K ) } { 2 } \Delta p ( x ) } \end{array}$ . This yields

$$
\begin{array}{r l} & {\frac {\mathbb {E} [ B _ {n} (x) ]}{\mathbb {E} [ A _ {n} (x) ]} = m (x) + \frac {h ^ {2} \mu_ {2} (K)}{2} \frac {(p \Delta m + m \Delta p + 2 \nabla m ^ {\top} \nabla p) p - m p \Delta p}{p ^ {2}} \Big | _ {x} + o (h ^ {2})} \\ & {\qquad = m (x) + \frac {h ^ {2} \mu_ {2} (K)}{2} \Big (\Delta m (x) + 2 \nabla m (x) ^ {\top} \frac {\nabla p (x)}{p (x)} \Big) + o (h ^ {2}),} \end{array}
$$

which recovers our statement. Variance. Linearize $\widehat { m } ( x ) = B _ { n } ( x ) / A _ { n } ( x )$ around $( \mathbb { E } [ B _ { n } ( x ) ] , \mathbb { E } [ A _ { n } ( x ) ] )$ and use independence. To leading order,

$$
\operatorname{Var} [ \widehat {m} (x) ] \approx \frac {\operatorname{Var} [ B _ {n} (x) ]}{(\mathbb {E} [ A _ {n} (x) ]) ^ {2}}.
$$

Compute

$$
\left| \begin{array}{l} \operatorname{Var} [ B _ {n} (x) ] = \sum_ {i = 1} ^ {n} \operatorname{Var} \big (K _ {h} (x - X _ {i}) Y _ {i} \big) \quad (\text {independence}) \\ \qquad = n \mathbb {E} \big [ K _ {h} (x - X) ^ {2} \operatorname{Var} (Y \mid X) \big ] = n \mathbb {E} \big [ K _ {h} (x - X) ^ {2} v (X) \big ] \\ \qquad = n \int h ^ {- 2 d} K \Big (\frac {x - u}{h} \Big) ^ {2} v (u) p (u) d u \\ \qquad = n h ^ {- d} \int K (t) ^ {2} v (x - h t) p (x - h t) d t = n h ^ {- d} \Big (R (K) v (x) p (x) + o (1) \Big), \end{array} \right|
$$

while

$$
\mathbb {E} [ A _ {n} (x) ] = n \big (p (x) + o (1) \big).
$$

Therefore,

$$
\operatorname{Var} [ \widehat {m} (x) ] \approx \frac {n h ^ {- d} R (K) v (x) p (x)}{n ^ {2} p (x) ^ {2}} = \frac {R (K)}{n h ^ {d}} \frac {v (x)}{p (x)} + o \big ((n h ^ {d}) ^ {- 1} \big),
$$

completing the proof.

## B.6 Proof of Equation (5) to Equation (6)

Proof. Let $\begin{array} { r } { \bar { \mathbf { z } } = \frac { 1 } { V _ { g } } \sum _ { v = 1 } ^ { V _ { g } } \mathbf { z } _ { n , v } } \end{array}$ denote the mean of the first $V _ { g }$ vectors.

33

<!-- page 34 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

We prove that:

$$
\frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \| \mathbf {z} _ {n, v} - \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2} = \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \| \bar {\mathbf {z}} - \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2}\tag{14}
$$

Expanding the left-hand side:

(15)

$$
\begin{array}{r l} \mathrm{LHS} & = \frac {1}{V _ {g} V} \sum_ {v = 1} ^ {V _ {g}} \sum_ {v ^ {\prime} = 1} ^ {V} \| \mathbf {z} _ {n, v} - \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2} \\ & = \frac {1}{V _ {g} V} \sum_ {v = 1} ^ {V _ {g}} \sum_ {v ^ {\prime} = 1} ^ {V} \left(\| \mathbf {z} _ {n, v} \| _ {2} ^ {2} - 2 \mathbf {z} _ {n, v} ^ {T} \mathbf {z} _ {n, v ^ {\prime}} + \| \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2}\right) \\ & = \frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \| \mathbf {z} _ {n, v} \| _ {2} ^ {2} - \frac {2}{V _ {g} V} \sum_ {v = 1} ^ {V _ {g}} \sum_ {v ^ {\prime} = 1} ^ {V} \mathbf {z} _ {n, v} ^ {T} \mathbf {z} _ {n, v ^ {\prime}} + \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \| \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2} \\ & = \frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \| \mathbf {z} _ {n, v} \| _ {2} ^ {2} - \frac {2}{V} \bar {\mathbf {z}} ^ {T} \sum_ {v ^ {\prime} = 1} ^ {V} \mathbf {z} _ {n, v ^ {\prime}} + \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \| \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2} \end{array}\tag{16}
$$

(17)

(18)

Expanding the right-hand side:

$$
\begin{array}{r l} \mathrm{RHS} & = \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \left(\| \bar {\mathbf {z}} \| _ {2} ^ {2} - 2 \bar {\mathbf {z}} ^ {T} \mathbf {z} _ {n, v ^ {\prime}} + \| \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2}\right) \\ & = \| \bar {\mathbf {z}} \| _ {2} ^ {2} - \frac {2}{V} \bar {\mathbf {z}} ^ {T} \sum_ {v ^ {\prime} = 1} ^ {V} \mathbf {z} _ {n, v ^ {\prime}} + \frac {1}{V} \sum_ {v ^ {\prime} = 1} ^ {V} \| \mathbf {z} _ {n, v ^ {\prime}} \| _ {2} ^ {2} \end{array}\tag{19}
$$

(20)

To complete the proof, we verify that:

$$
\frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \| \mathbf {z} _ {n, v} \| _ {2} ^ {2} = \| \bar {\mathbf {z}} \| _ {2} ^ {2}\tag{21}
$$

Expanding the right-hand side:

$$
\| \bar {\mathbf {z}} \| _ {2} ^ {2} = \left\| \frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \mathbf {z} _ {n, v} \right\| _ {2} ^ {2}\tag{22}
$$

$$
= \frac {1}{V _ {g} ^ {2}} \sum_ {v = 1} ^ {V _ {g}} \sum_ {v ^ {\prime \prime} = 1} ^ {V _ {g}} \mathbf {z} _ {n, v} ^ {T} \mathbf {z} _ {n, v ^ {\prime \prime}}\tag{23}
$$

$$
= \frac {1}{V _ {g}} \sum_ {v = 1} ^ {V _ {g}} \| \mathbf {z} _ {n, v} \| _ {2} ^ {2}\tag{24}
$$

Therefore, $\mathrm { L H S } = \mathrm { R H S } ,$ completing the proof.

## B.7 Proof of thm. 8

Proof. For each $x ,$

$$
\mathrm{Bias} [ \widehat {m} (x) ] = \frac {h ^ {2} \mu_ {2} (K)}{2} \Big (\Delta m (x) + 2 \nabla m (x) ^ {\top} \nabla \log p (x) \Big) + o (h ^ {2}).
$$

34

<!-- page 35 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Square and integrate against $p ( x ) ;$

$$
\begin{array}{r l} & {\mathcal {B} ^ {2} (h; p, m) = \Big (\frac {h ^ {2} \mu_ {2} (K)}{2} \Big) ^ {2} \int \Big (\Delta m (x) + 2 \nabla m (x) ^ {\top} \nabla \log p (x) \Big) ^ {2} p (x) d x + o (h ^ {4})} \\ & {\qquad \leq \Big (\frac {h ^ {2} \mu_ {2} (K)}{2} \Big) ^ {2} \int \Big (2 (\Delta m (x)) ^ {2} + 2 (2 \nabla m (x) ^ {\top} \nabla \log p (x)) ^ {2} \Big) p (x) d x + o (h ^ {4})} \\ & {\qquad = \Big (\frac {h ^ {2} \mu_ {2} (K)}{2} \Big) ^ {2} \Big (2 \int (\Delta m (x)) ^ {2} p (x) d x + 8 \int (\nabla m (x) ^ {\top} \nabla \log p (x)) ^ {2} p (x) d x \Big) + o (h ^ {4}),} \end{array}
$$

where we used $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ pointwise. Since $| \Delta m ( x ) | \leq B$ for all $x ,$ we have

$$
\int (\Delta m) ^ {2} p \leq \int B ^ {2} p = B ^ {2}.
$$

For the second term, first use Cauchy–Schwarz and then integrate against $p ( x )$ to obtain

$$
\begin{array}{c} (\nabla m (x) ^ {\top} \nabla \log p (x)) ^ {2} \leq \| \nabla m (x) \| ^ {2} \| \nabla \log p (x) \| ^ {2} \leq L ^ {2} \| \nabla \log p (x) \| ^ {2} \\ \implies \int (\nabla m (x) ^ {\top} \nabla \log p (x)) ^ {2} p (x) d x \leq L ^ {2} \int \| \nabla \log p (x) \| ^ {2} p (x) d x = L ^ {2} J (p). \end{array}
$$

which can be combined with the bounds above to obtain the desired result. We similarly have for the integrated variance

$$
\mathcal {V} (h; p) = \int \left(\frac {R (K)}{n h ^ {d}} \frac {v (x)}{p (x)} + o \big ((n h ^ {d}) ^ {- 1} \big)\right) p (x) d x = \frac {R (K)}{n h ^ {d}} \int v (x) d x + o \big ((n h ^ {d}) ^ {- 1} \big),
$$

which is independent of $p .$

□

## B.8 Proof of lemma. 3

Proof. We first start by reminding the reader about the original Cramér-Wold theorem that is a function of all possible directions (not unit-norm ones).

## Theorem 10: Cramér-Wold Cramér and Wold [1936]

Let 𝑋 and 𝑌 be random vectors in $\mathbb { R } ^ { D }$ :

$$
X \stackrel {{d}} {{=}} Y \iff \langle X, a \rangle \stackrel {{d}} {{=}} \langle Y, a \rangle , \forall \boldsymbol {a} \in \mathbb {R} ^ {D}.\tag{25}
$$

Our proof will follow the same proof as for thm. 10. Necessity is immediate: i $\operatorname { i f } X { \stackrel { d } { = } } Y ,$ then every measurable function of 𝑋 has the same distribution as the corresponding function of $Y ,$ from which the linear mapping $x \mapsto \langle u , x \rangle$ for $u \in \mathbb { S } ^ { d - 1 }$ is a special case. For sufficiency, assume $\langle u , X \rangle \stackrel { d } { = } \langle u , Y \rangle$ for all $u \in \mathbb { S } ^ { d - 1 }$ . Let $\varphi _ { X } ( t ) : = \mathbb { E } \big [ e ^ { i \langle t , X \rangle } \big ]$ and $\varphi _ { Y } ( t ) : = \mathbb { E } \big [ e ^ { i \langle t , Y \rangle } \big ]$ denote the characteristic functions of 𝑋 and $Y .$ Fix an arbitrary $t \in \mathbb { R } ^ { d } ; \mathrm { i f } \; t = 0 ,$ , then $\varphi _ { X } ( 0 ) = \varphi _ { Y } ( 0 ) = 1$ . If $t \neq 0 ,$ , write $t = s u$ with $s : = \| t \| > 0$ and $u : = t / \| t \| \in \mathbb { S } ^ { d - 1 }$ . By the assumption, $\langle u , X \rangle \stackrel { d } { = } \langle u , Y \rangle$ , hence for this 𝑢 and 𝑠 we have

$$
\varphi_ {X} (t) = \mathbb {E} \big [ e ^ {i \langle t, X \rangle} \big ] = \mathbb {E} \big [ e ^ {i s \langle u, X \rangle} \big ] = \mathbb {E} \big [ e ^ {i s \langle u, Y \rangle} \big ] = \mathbb {E} \big [ e ^ {i \langle t, Y \rangle} \big ] = \varphi_ {Y} (t).
$$

Thus $\varphi _ { X } ( t ) = \varphi _ { Y } ( t )$ for all $t \in \mathbb { R } ^ { d }$ , i.e., $\varphi _ { X } \equiv \varphi _ { Y }$ on $\mathbb { R } ^ { d }$ . By the uniqueness theorem for characteristic functions, this implies $X { \stackrel { d } { = } } Y .$ . (ii) Define $\psi _ { n , t } : = \mathbb { E } \big [ e ^ { i \langle t , X _ { n } \rangle } \big ]$ and $\psi _ { t } : = \mathbb { E } \big [ e ^ { i \langle t , X \rangle } \big ]$ . Fix $t \in \mathbb { R } ^ { d }$ and decompose $t = s u$ with $s : = \| t \| \geq 0$ and $u \in \mathbb { S } ^ { d - 1 } \; ( \mathrm { t a k e } ,   \mathrm { e . g . } ,   u = t / \| t \|   \mathrm { i f } \; \bar { t } \neq 0 ,$ and any 𝑢 if $t = 0 )$ . The map $g _ { s } : \mathbb { R } \rightarrow \mathbb { R } ,   g _ { s } ( x ) = s x ,$ , is continuous. By the continuous mapping theorem applied to the real-valued random variables $\left\langle u , X _ { n } \right\rangle { \xrightarrow { d } } \left\langle u , X \right\rangle$ , we obtain

$$
\langle t, X _ {n} \rangle = s \langle u, X _ {n} \rangle \stackrel {d} {\rightarrow} s \langle u, X \rangle = \langle t, X \rangle .
$$

35

<!-- page 36 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Hence, for every fixed $t \in \mathbb { R } ^ { d }$ , the one-dimensional projections satisfy $\left\langle t , X _ { n } \right\rangle { \xrightarrow { d } } \left\langle t , X \right\rangle$ , which in turn yields pointwise convergence of characteristic functions:

$$
\psi_ {n, t} = \mathbb {E} \left[ e ^ {i \langle t, X _ {n} \rangle} \right] \longrightarrow \mathbb {E} \left[ e ^ {i \langle t, X \rangle} \right] = \psi_ {t}, \quad \text {for all} t \in \mathbb {R} ^ {d}.
$$

Therefore, by $\mathrm { L \acute { e } v y ^ { \prime } s }$ continuity theorem, $X _ { n } \stackrel { d } { \rightarrow } X$ . This completes the proof.

## B.9 Proof of thm. 2

Proof. We first formulate the following assumptions required for the proof–all of this are satisfied by typical univariate statistical tests.

$P = Q$ if and only if $P _ { a } = Q _ { a }$ for all $a \in S ^ { d - 1 }$ (population-level equivalence of laws).

$A _ { n }$ are finite sets with mesh $\Delta ( A _ { n } ) : = \operatorname { \operatorname* { s u p } } _ { u \in S ^ { d - 1 } } \operatorname * { m i n } _ { a \in A _ { n } } \| u - a \| \to 0 \; \mathrm { a s } \; n \to \infty .$

If $P \neq Q$ , there exists a separating direction $\tilde { a } ^ { \star } \in S ^ { d - 1 }$ and a neighborhood 𝑈 of $a ^ { \star }$ such that

$$
\inf _ {a \in U} \lim _ {n \to \infty} \Pr \left(T _ {a, n} \geq u _ {n} (\alpha)\right) = 1.
$$

(Intuitively: near a truly separating direction, the 1D statistic eventually exceeds the global null threshold with probability $\rightarrow 1 . )$

(i) Under $H _ { 0 }   :   P   =   Q$ , our assumption implies no separating direction exists at the population level, and the calibration of $u _ { n } ( \alpha )$ ensures $\operatorname* { P r } ( M _ { n } \geq u _ { n } ( \alpha ) ) \leq \alpha$ for all $n ,$ hence lim su $\mathbb { P } _ { n \to \infty } \operatorname* { P r } ( \Psi _ { n } = 1 ) \leq \alpha$ . (ii) Suppose $P \neq Q$ . Our assumption guarantees that there exists at least one separating direction $a ^ { \star }$ with $P _ { a ^ { \star } } \neq Q _ { a ^ { \star } }$ . Our assumption guarantees a neighborhood 𝑈 of $a ^ { \star }$ in which the projection statistics exceed the global null threshold with probability tending to 1:

$$
\inf _ {a \in U} \lim _ {n \to \infty} \Pr \left(T _ {a, n} \geq u _ {n} (\alpha)\right) = 1.
$$

By assumption, for all large 𝑛 the set $A _ { n }$ contains at least one direction $a _ { n } \in U$ (dense coverage). Therefore,

$$
\Pr (\Psi_ {n} = 1) = \Pr \left(M _ {n} \geq u _ {n} (\alpha)\right) \geq \Pr \left(T _ {a _ {n}, n} \geq u _ {n} (\alpha)\right) \longrightarrow 1,
$$

which proves consistency.

## B.10 Proof of thm. 5

Proof. For each case, consider the function $g ( a )$ on $\mathbb { S } ^ { D - 1 }$ defined by the quantity of interest (CF, CDF, or moment) at a fixed 𝑡 or $k .$ Since $f \in H ^ { \alpha } ( \mathbb { R } ^ { D } )$ , the mapping $a \mapsto g ( a )$ is in $H ^ { \alpha } ( \mathbb { S } ^ { D - 1 } )$ for each fixed 𝑡 or $k .$

Given 𝑀 samples $\{ a _ { i } \} _ { i = 1 } ^ { M }$ on the sphere, the best possible reconstruction of $g$ from its values at these points is given by spherical interpolation. By classical results on Sobolev spaces and spherical harmonics (see, e.g., Narcowich et al. [2006]), the $L ^ { 2 }$ interpolation error for functions in $H ^ { \alpha } ( \mathbb { S } ^ { D - 1 } )$ using 𝑀 points is bounded by

$$
\mathbb {E} _ {b} \left[ | g (b) - g ^ {*} (b) | ^ {2} \right] \leq C (D, \alpha) M ^ {- 2 \alpha / (D - 1)} \| g \| _ {H ^ {\alpha} (\mathbb {S} ^ {D - 1})} ^ {2},
$$

where $g ^ { * }$ is the interpolant matching $g$ at the 𝑀 sampled points. The interpolation error bound on the sphere follows from the theory of spherical harmonics and Marcinkiewicz– $- Z y$ gmund (MZ) inequalities . Any $f \in H ^ { \alpha } ( \mathbb { S } ^ { d } )$ admits a spherical harmonics expansion, and the best $L ^ { 2 }$ approximation by harmonics of degree at most 𝐿 satisfies

$$
\| f - P _ {L} f \| _ {L ^ {2} (\mathbb {S} ^ {d})} \leq (1 + L ^ {2}) ^ {- \alpha / 2} \| f \| _ {H ^ {\alpha} (\mathbb {S} ^ {d})},
$$

where $P _ { L } f$ is the projection onto harmonics of degree $\leq L$ [Narcowich et al., 2006, Lemma 2.1]. If 𝑀 points are distributed quasi-uniformly on $\mathbb { S } ^ { d }$ , then for $L \sim c M ^ { 1 / d }$ , the set forms a Marcinkiewicz– $- Z y$ gmund (MZ) set for degree 𝐿 [Mhaskar et al., 2001, Theorem 1.1]. This allows reconstruction of any function in the space of harmonics of degree at most 𝐿 from its values at these points, and the $L ^ { 2 }$ interpolation error for $f$ is bounded by

$$
\| f - I _ {M} f \| _ {L ^ {2} (\mathbb {S} ^ {d})} \leq C (1 + L ^ {2}) ^ {- \alpha / 2} \| f \| _ {H ^ {\alpha} (\mathbb {S} ^ {d})},
$$

36

<!-- page 37 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

where $I _ { M } f$ is any interpolant matching $f$ at the 𝑀 points [Narcowich et al., 2006, Theorem 3.1]. Substituting $L \sim c M ^ { 1 / d }$ yields the rate $\dot { M } ^ { - \alpha / d }$ , and thus

$$
\mathbb {E} _ {\omega} | f (\omega) - I _ {M} f (\omega) | ^ {2} \leq C (d, \alpha) M ^ {- 2 \alpha / d} \| f \| _ {H ^ {\alpha} (\mathbb {S} ^ {d})} ^ {2},
$$

with explicit $C ( d , \alpha )$ as in the main theorem. Integrating (or summing) over 𝑡 (for CF and CDF) or 𝑘 (for moments, with weights $w _ { k } )$ yields the stated bounds. The explicit constant $C ( D , \alpha )$ arises from the theory of spherical Sobolev spaces and is given above.

For the moment case, the sum over 𝑘 is weighted to ensure convergence, as higher moments may grow rapidly. The weights $w _ { k }$ can be chosen, for example, as $w _ { k } = 1 / k !$

This completes the proof.

## B.11 Proof of thm. 3

Pick distinct $x _ { 0 } , \cdots , x _ { K + 1 } \in \mathbb { R }$ and consider the linear map $\begin{array} { r } { A : \mathbb { R } ^ { K + 2 } \rightarrow \mathbb { R } ^ { K + 1 } , ( A p ) _ { r } = \sum _ { j = 0 } ^ { K + 1 } p _ { j } x _ { j } ^ { r } } \end{array}$ for $r = 0 , \cdots , K .$ . Then $\operatorname { r a n k } ( A ) \leq K + 1$ , so ker $r(A) \neq \{0\}$ . Let $v \in \ker ( A ) \setminus \{ 0 \}$ ; from $\begin{array} { r } { ( A p ) _ { 0 } = \sum _ { j } p _ { j } , } \end{array}$ , we get $\begin{array} { r } { \sum _ { j } v _ { j } = 0 , } \end{array}$ , hence 𝑣 has positive and negative entries. Choose a strictly positive probability vector 𝑝 and $\varepsilon > 0$ small such that $p ^ { \pm } : = p \pm \varepsilon v$ remain probability vectors. Then $A p ^ { + } = A p ^ { - }$ , so the distributions supported on $\{ x _ { j } \}$ with masses $p ^ { \pm }$ are distinct yet match moments up to order 𝐾.

## B.12 Proof of thm. 4

Proof. Fix the Gaussian weight

$$
w _ {s} (t) = e ^ {- s ^ {2} t ^ {2}}, \qquad s > 0,
$$

and define the population CF distance

$$
D (P, G) = \int_ {\mathbb {R}} w _ {s} (t) \bigl | \varphi_ {P} (t) - \varphi_ {G} (t) \bigr | ^ {2} d t.
$$

Let the empirical CF be

$$
\left| \widehat {\varphi} _ {N} (t) = \frac {1}{N} \sum_ {i = 1} ^ {N} e ^ {i t X _ {i}}, \right.
$$

and consider the V-statistic estimator

$$
\widehat {D} _ {V} = \int_ {\mathbb {R}} w _ {s} (t) \big | \widehat {\varphi} _ {N} (t) - \varphi_ {G} (t) \big | ^ {2} d t.
$$

We use only that $| e ^ { i t X } | = 1 , | \varphi _ { P } ( t ) | \leq 1 , | \varphi _ { G } ( t ) | \leq 1$ , and integrability of $w _ { s }$ . For each 𝑖 differentiate under the integral (dominated convergence applies because the integrand and its derivative are bounded)

$$
\begin{array}{c} \frac {\partial \widehat {D} _ {V}}{\partial X _ {i}} = \int_ {\mathbb {R}} w _ {s} (t) 2 \Re \Big (\big (\widehat {\varphi} _ {N} (t) - \varphi_ {G} (t) \big) \overline {\frac {\partial \widehat {\varphi} _ {N} (t)}{\partial X _ {i}}} \Big) d t, \\ \frac {\partial \widehat {\varphi} _ {N} (t)}{\partial X _ {i}} = \frac {1}{N} i t e ^ {i t X _ {i}}, \end{array}
$$

since $| \widehat { \varphi } _ { N } ( t ) | \leq 1$ and $| \varphi _ { G } ( t ) | \leq 1$ ,

$$
\begin{array}{r l} & {\left| \frac {\partial \widehat {D} _ {V}}{\partial X _ {i}} \right| \leq \frac {2}{N} \int w _ {s} (t) | t | \big (| \widehat {\varphi} _ {N} (t) | + | \varphi_ {G} (t) | \big) d t} \\ & {\qquad \leq \frac {4}{N} \int w _ {s} (t) | t | d t} \\ & {\qquad = \frac {4}{N s ^ {2}},} \end{array}
$$

37

<!-- page 38 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

using $\begin{array} { r } { \int _ { \mathbb { R } } e ^ { - s ^ { 2 } t ^ { 2 } } | t | d t = 1 / s ^ { 2 } } \end{array}$

$$
\left| \frac {\partial \widehat {D} _ {V}}{\partial X _ {i}} \right| \leq \frac {4}{N} \int_ {\mathbb {R}} w _ {s} (t) | t | d t = \frac {4}{N s ^ {2}}.
$$

Moreover, differentiating once more in $X _ { i }$ and using $| \widehat { \varphi } _ { N } ( t ) | \leq 1 , | \varphi _ { G } ( t ) | \leq 1$ gives a global Lipschitz bound

$$
\left| \frac {\partial^ {2} \widehat {D} _ {V}}{\partial X _ {i} ^ {2}} \right| \leq \frac {C}{N} \int_ {\mathbb {R}} w _ {s} (t) t ^ {2} d t = \frac {C}{N} \cdot \frac {\sqrt {\pi}}{2 s ^ {3}},
$$

for some absolute constant 𝐶 arising from bounded factors and product rule. Hence ECF gradients are uniformly bounded and Lipschi $\mathbf { \boldsymbol { Z } } ,$ with scale controlled only by $\left( N , s \right)$

(B) (Moment sample-gradients are polynomial in $X _ { i }$ and unbounded for $k \geq 2 . )$ Let $\widehat { D } _ { V }$ be as above. Define the moment objective

$$
\widehat {D} _ {k} = (\bar {\phi} - \mu) ^ {\top} W (\bar {\phi} - \mu), \quad \bar {\phi} := \frac {1}{N} \sum_ {i = 1} ^ {N} \phi (X _ {i}), \quad \phi (x) = (x, x ^ {2}, \dots , x ^ {k}) ^ {\top},
$$

for a symmetric positive semidefinite $W \in \mathbb { R } ^ { k \times k }$ and Gaussian target moments $\mu = \mathbb { E } _ { G } [ \phi ( Y ) ]$ . For each $i ,$

$$
\left| \begin{array}{c} \frac {\partial \widehat {D} _ {k}}{\partial X _ {i}} = \frac {2}{N} (\bar {\phi} - \mu) ^ {\top} W \frac {\partial \phi (X _ {i})}{\partial X _ {i}}, \\ \frac {\partial \phi (X)}{\partial X} = (1, 2 X, 3 X ^ {2}, \ldots , k X ^ {k - 1}) ^ {\top}. \end{array} \right|
$$

The gradient formula follows by the chain rule and linearity of $\bar { \phi } .$ Let $c : = W ( \bar { \phi } - \mu )$ and write $c _ { r }$ for its 𝑟-th coordinate. Then

$$
\left| \frac {\partial \widehat {D} _ {k}}{\partial X _ {i}} = \frac {2}{N} \sum_ {r = 1} ^ {k} c _ {r} r X _ {i} ^ {r - 1}, \right.
$$

which is a polynomial in $X _ { i }$ of degree $\mathbf { d e g } = \operatorname* { m a x } \{ r - 1 : c _ { r } \neq 0 \} \leq k - 1$ . In particular, if $c _ { k } \neq 0$ (the generic case when the top-weighted deviation is nonzero), then

$$
\left| \frac {\partial \widehat {D} _ {k}}{\partial X _ {i}} \right| \underset {| X _ {i} | \to \infty} {\longrightarrow} \infty \quad \text {as} \quad | X _ {i} | ^ {k - 1}.
$$

The expression is a nonconstant polynomial in $X _ { i }$ of degree $\mathbf { d e g } \leq k - 1$ whenever some $c _ { r } \neq 0$ with $r \geq 2$ . Thus the gradient cannot be uniformly bounded on R. If $c _ { k } \neq 0 ,$ , the leading term dominates and the magnitude grows like $| X _ { i } | ^ { k - 1 }$ proving unboundedness for $k \geq 2$ □

## B.13 Proof of thm. 6

Proof. A direct calculation shows Fix $t \in \mathbb { R } ^ { d }$ and abbreviate $Z _ { j } : = e ^ { \mathrm { i } t ^ { \top } X _ { j } }$ , so that $\begin{array} { r } { \phi _ { n } ( t ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } Z _ { j } } \end{array}$ . Note that $| Z _ { j } | = 1$ almost surely (since $t ^ { \top } X _ { j } \in \mathbb { R } )$ , and $\mathbb { E } [ Z _ { j } ] = \phi _ { \theta } ( t )$ for all 𝑗. We start from the algebraic identity

$$
\left| \phi_ {n} (t) - \psi (t) \right| ^ {2} = \phi_ {n} (t) \overline {{\phi_ {n} (t)}} - \psi (t) \overline {{\phi_ {n} (t)}} - \overline {{\psi (t)}} \phi_ {n} (t) + \left| \psi (t) \right| ^ {2}.
$$

38

<!-- page 39 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Taking expectations term by term gives

$$
\mathbb {E} \left[ \left| \phi_ {n} - \psi \right| ^ {2} \right] = \mathbb {E} \left[ | \phi_ {n} | ^ {2} \right] - \psi \mathbb {E} \left[ \overline {{\phi_ {n}}} \right] - \overline {{\psi}} \mathbb {E} \left[ \phi_ {n} \right] + | \psi | ^ {2},\tag{26}
$$

$$
= \mathbb {E} \left[ | \phi_ {n} | ^ {2} \right] - \psi \overline {{\mathbb {E} [ \phi_ {n} ]}} - \overline {{\psi}} \frac {1}{n} \sum_ {j = 1} ^ {n} \mathbb {E} [ Z _ {j} ] + | \psi | ^ {2},\tag{27}
$$

$$
= \mathbb {E} \left[ | \phi_ {n} | ^ {2} \right] - \psi \overline {{\phi_ {\theta}}} - \overline {{\psi}} \phi_ {\theta} + | \psi | ^ {2},\tag{28}
$$

$$
= \mathbb {E} \left[ | \phi_ {n} | ^ {2} \right] - 2 \mathrm{Re} \big (\overline {{\psi}} \phi_ {\theta} \big) + | \psi | ^ {2},
$$

$$
= \mathbb {E} \left[ \left| \frac {1}{n} \sum_ {j = 1} ^ {n} Z _ {j} \right| ^ {2} \right] - 2 \mathrm{Re} \big (\overline {{\psi}} \phi_ {\theta} \big) + | \psi | ^ {2},\tag{29}
$$

(30)

$$
= \frac {1}{n ^ {2}} \sum_ {j = 1} ^ {n} \sum_ {l = 1} ^ {n} \mathbb {E} \left[ Z _ {j} \overline {{Z _ {l}}} \right] - 2 \mathrm{Re} \big (\overline {{\psi}} \phi_ {\theta} \big) + | \psi | ^ {2},\tag{31}
$$

(32)

Since the $Z _ { j }$ are i.i.d.,

$$
\begin{array}{r} \mathbb {E} \left[ Z _ {j} \overline {{Z _ {l}}} \right] = \left\{ \begin{array}{l l} \mathbb {E} \left[ | Z _ {1} | ^ {2} \right] = 1, & \text {if} j = l, \\ \mathbb {E} [ Z _ {j} ] \overline {{\mathbb {E} [ Z _ {l} ]}} = \phi_ {\theta} \overline {{\phi_ {\theta}}} = | \phi_ {\theta} | ^ {2}, & \text {if} j \neq l, \end{array} \right. \end{array}
$$

hence

$$
\left| \begin{array}{c} \mathbb {E} \left[ | \phi_ {n} | ^ {2} \right] = \frac {1}{n ^ {2}} \Big (n + n (n - 1) | \phi_ {\theta} | ^ {2} \Big) \\ = \frac {1}{n} + \left(1 - \frac {1}{n}\right) | \phi_ {\theta} | ^ {2} \\ = | \phi_ {\theta} | ^ {2} + \frac {1 - | \phi_ {\theta} | ^ {2}}{n} \end{array} \right|
$$

Plugging these, we obtain

$$
\begin{array}{r l} \mathbb {E} \left[ \left| \phi_ {n} - \psi \right| ^ {2} \right] & = \left(| \phi_ {\theta} | ^ {2} + \frac {1 - | \phi_ {\theta} | ^ {2}}{n}\right) - 2 \mathrm{Re} \big (\overline {{\psi}} \phi_ {\theta} \big) + | \psi | ^ {2} \\ & = \left(| \phi_ {\theta} | ^ {2} - 2 \mathrm{Re} \big (\overline {{\psi}} \phi_ {\theta} \big) + | \psi | ^ {2}\right) + \frac {1 - | \phi_ {\theta} | ^ {2}}{n} \\ & = \left| \phi_ {\theta} - \psi \right| ^ {2} + \frac {1 - | \phi_ {\theta} | ^ {2}}{n}. \end{array}
$$

Under Dominated convergence, $\mathbb { E } [ \nabla _ { \theta } D _ { n } ( t ) ] = \nabla _ { \theta } \mathbb { E } [ D _ { n } ( t ) ]$ , hence

$$
\mathbb {E} \left[ \nabla_ {\theta} D _ {n} (t) \right] = \nabla_ {\theta} \bigl | \phi_ {\theta} (t) - \psi (t) \bigr | ^ {2} + \nabla_ {\theta} \frac {1 - | \phi_ {\theta} (t) | ^ {2}}{n},
$$

concluding the proof.

In practice one replaces $\begin{array} { r } { \int _ { \mathbb { R } } w ( t ) ( \cdot ) d t } \end{array}$ by a deterministic quadrature on a uniform grid $t _ { k } \in [ - T , T ]$ with weights $\omega _ { k } \left( \mathbf { e . g . } \right.$ trapezoidal rule) and a Gaussian window $w ( t ) = e ^ { - \alpha t ^ { 2 } }$ . All statements above remain valid with the integral replaced by $\textstyle \sum _ { k } \omega _ { k } ( \cdot ) { : }$

$$
L (\theta) \approx \sum_ {k} \omega_ {k} \bigl | \phi_ {\theta} (t _ {k}) - \psi (t _ {k}) \bigr | ^ {2}, \quad \widehat {L} _ {n} (\theta) \approx \sum_ {k} \omega_ {k} \bigl | \phi_ {n} (t _ {k}) - \psi (t _ {k}) \bigr | ^ {2},
$$

39

<!-- page 40 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

and the bias term becomes

$$
\mathrm{Bias} (\theta) = - \frac {1}{n} \sum_ {k} \omega_ {k} \nabla_ {\theta} \bigl | \phi_ {\theta} (t _ {k}) \bigr | ^ {2}.
$$

Since the grid and weights are deterministic, they do not affect unbiasedness with respect to sampling; they only introduce a deterministic approximation error to the target functional 𝐿(𝜃).

## B.14 Proof of VICReg’s Recovery

Proof. We prove this result in two parts.

Part $\mathbf { I } \mathbf { : } \mathbb { E } [ \mathbf { X } ] = \mathbf { 0 }$ Given that $\mathbb { E } [ \langle \pmb { X } , \pmb { a } \rangle ] = 0$ for all unit vectors a, and noting that $\langle { \pmb X } , { \pmb a } \rangle = { \pmb a } ^ { T } { \pmb X } ,$ we have:

$$
\mathbb {E} [ \boldsymbol {a} ^ {T} \boldsymbol {X} ] = 0 \quad \text {for all} \boldsymbol {a} \in \mathbb {R} ^ {d} \text {with} \| \boldsymbol {a} \| = 1\tag{33}
$$

By linearity of expectation:

$$
\mathbf {a} ^ {T} \mathbb {E} [ \mathbf {X} ] = 0 \quad \text {for all unit vectors} \mathbf {a}\tag{34}
$$

Let $\mu = \mathbb { E } [ \mathbf { X } ]$ . We claim that $\boldsymbol { \mu } = \mathbf { 0 }$ . Suppose, for the sake of contradiction, that $\boldsymbol { \mu } \neq \mathbf { 0 }$ . Then $\| \boldsymbol { \mu } \| _ { 2 } > 0$ . Define the unit vector:

$$
\mathbf {a} ^ {*} = \frac {\mu}{\| \mu \| _ {2}}\tag{35}
$$

Since $\pmb { \mathfrak { a } } ^ { * }$ is a unit vector, equation (33) implies:

$$
(\boldsymbol {a} ^ {*}) ^ {T} \boldsymbol {\mu} = 0\tag{36}
$$

However, substituting the definition of ${ \mathfrak { a } } ^ { * } ;$ :

$$
\left(\mathbf {a} ^ {*}\right) ^ {T} \boldsymbol {\mu} = \left(\frac {\boldsymbol {\mu}}{\| \boldsymbol {\mu} \| _ {2}}\right) ^ {T} \boldsymbol {\mu} = \frac {\boldsymbol {\mu} ^ {T} \boldsymbol {\mu}}{\| \boldsymbol {\mu} \| _ {2}} = \frac {\| \boldsymbol {\mu} \| _ {2} ^ {2}}{\| \boldsymbol {\mu} \| _ {2}} = \| \boldsymbol {\mu} \| _ {2} > 0 \left. \right\vert\tag{37}
$$

This contradiction establishes that $\boldsymbol { \mu } = \mathbf { 0 }$

**Part II:** $\operatorname { C o v } ( \pmb { X } ) = \mathbf { I } _ { d }$ Since $\mathbb { E } [ \mathbf { X } ] = \mathbf { 0 } ,$ , we have:

$$
\operatorname{Var} (\langle \mathbf {X}, \mathbf {a} \rangle) = \mathbb {E} [ (\langle \mathbf {X}, \mathbf {a} \rangle) ^ {2} ] = \mathbb {E} [ (\mathbf {a} ^ {T} \mathbf {X}) ^ {2} ]\tag{38}
$$

Expanding the quadratic form:

$$
\mathbb {E} [ (\mathbf {a} ^ {T} \mathbf {X}) ^ {2} ] = \mathbb {E} [ \mathbf {a} ^ {T} \mathbf {X} \mathbf {X} ^ {T} \mathbf {a} ] = \mathbf {a} ^ {T} \mathbb {E} [ \mathbf {X} \mathbf {X} ^ {T} ] \mathbf {a}\tag{39}
$$

Since $\mathbb { E } [ \mathbf { X } ] = \mathbf { 0 } ,$ the covariance matrix is $\operatorname { \mathsf { C o v } } ( \mathbf { X } ) = \operatorname { \mathbb { E } } [ \mathbf { X } \mathbf { X } ^ { T } ] . \operatorname { \mathsf { L e t } } \boldsymbol { \Sigma } = \operatorname { \mathsf { C o v } } ( \mathbf { X } )$ . The variance condition gives us:

$$
\mathbf {a} ^ {T} \boldsymbol {\Sigma} \mathbf {a} = 1 \quad \text {for all unit vectors} \mathbf {a}\tag{40}
$$

We now show that $\boldsymbol { \Sigma } = \mathbf { I } _ { d }$ . Step 1: Diagonal entries. For $i \in \{ 1 , 2 , \cdots , d \}$ , let $\mathbf { e } _ { i }$ denote the 𝑖-th standard basis vector. Setting $\mathbf { a } = \mathbf { e } _ { i }$ in equation (40):

$$
\mathbf {e} _ {i} ^ {T} \boldsymbol {\Sigma} \mathbf {e} _ {i} = \Sigma_ {i i} = 1\tag{41}
$$

Therefore, all diagonal entries of 𝚺 equal 1. Step 2: Off-diagonal entries. For distinct indices $i , j \in \{ 1 , 2 , \cdots , d \}$ , consider the unit vector:

$$
\mathbf {a} = \frac {\mathbf {e} _ {i} + \mathbf {e} _ {j}}{\| \mathbf {e} _ {i} + \mathbf {e} _ {j} \| _ {2}} = \frac {\mathbf {e} _ {i} + \mathbf {e} _ {j}}{\sqrt {2}}\tag{42}
$$

Applying equation (40):

$$
\mathbf {a} ^ {T} \boldsymbol {\Sigma} \mathbf {a} = \frac {1}{2} \left(\mathbf {e} _ {i} + \mathbf {e} _ {j}\right) ^ {T} \boldsymbol {\Sigma} \left(\mathbf {e} _ {i} + \mathbf {e} _ {j}\right) = 1\tag{43}
$$

40

<!-- page 41 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Expanding the quadratic form and using the symmetry of $\Sigma ;$

$$
\frac {1}{2} (\mathbf {e} _ {i} ^ {T} \mathbf {\Sigma} \mathbf {e} _ {i} + 2 \mathbf {e} _ {i} ^ {T} \mathbf {\Sigma} \mathbf {e} _ {j} + \mathbf {e} _ {j} ^ {T} \mathbf {\Sigma} \mathbf {e} _ {j}) = 1\tag{44}
$$

$$
\frac {1}{2} (\Sigma_ {i i} + 2 \Sigma_ {i j} + \Sigma_ {j j}) = 1\tag{45}
$$

$$
\left| \frac {1}{2} (1 + 2 \Sigma_ {i j} + 1) = 1 \right|\tag{46}
$$

$$
1 + \Sigma_ {i j} = 1
$$

$$
\Sigma_ {i j} = 0\tag{47}
$$

(48)

Therefore, all off-diagonal entries of 𝚺 equal zero, establishing that $\boldsymbol { \Sigma } = \mathbf { I } _ { d }$

□

## C Background

**Foundation: The Linear Regression Model** We start with the standard linear regression model:

$$
\mathrm{y} = \chi \beta + \varepsilon
$$

where:

$\mathbf { y } = [ y _ { 1 } , y _ { 2 } , \cdots , y _ { n } ] ^ { T } \in \mathbb { R } ^ { n }$ is the response vector

$\mathbf { X } \in \mathbb { R } ^ { n \times p }$ is the design matrix with $\mathbf { X } _ { i j } = x _ { i j }$

$\pmb { \beta } = [ \beta _ { 1 } , \beta _ { 2 } , \cdots , \beta _ { p } ] ^ { T } \in \mathbb { R } ^ { p }$ is the parameter vector

$\boldsymbol { \varepsilon } = [ \varepsilon _ { 1 } , \varepsilon _ { 2 } , \cdots , \varepsilon _ { n } ] ^ { T } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { n } )$ is the error vector

The error assumption means:

$$
\mathbb {E} [ \varepsilon_ {i} ] = 0, \quad \operatorname{Var} (\varepsilon_ {i}) = \sigma^ {2}, \quad \operatorname{Cov} (\varepsilon_ {i}, \varepsilon_ {j}) = 0 \text {for} i \neq j
$$

**Step 1: Deriving the OLS Estimator** To find the OLS estimator, we minimize the sum of squared residuals:

$$
\mathrm{SSR} (\boldsymbol {\beta}) = \sum_ {i = 1} ^ {n} (y _ {i} - \mathbf {x} _ {i} ^ {T} \boldsymbol {\beta}) ^ {2} = (\mathbf {y} - \mathbf {X} \boldsymbol {\beta}) ^ {T} (\mathbf {y} - \mathbf {X} \boldsymbol {\beta})
$$

Expanding this quadratic form:

$$
\mathrm{SSR} (\boldsymbol {\beta}) = \mathbf {y} ^ {T} \mathbf {y} - 2 \boldsymbol {\beta} ^ {T} \mathbf {X} ^ {T} \mathbf {y} + \boldsymbol {\beta} ^ {T} \mathbf {X} ^ {T} \mathbf {X} \boldsymbol {\beta}\tag{49}
$$

Taking the derivative with respect to $\beta ;$

$$
\frac {\partial \mathrm{SSR}}{\partial \boldsymbol {\beta}} = - 2 \mathbf {X} ^ {T} \mathbf {y} + 2 \mathbf {X} ^ {T} \mathbf {X} \boldsymbol {\beta}
$$

Setting equal to zero and solving:

$$
\left| \begin{array}{c} - 2 \mathbf {X} ^ {T} \mathbf {y} + 2 \mathbf {X} ^ {T} \mathbf {X} \boldsymbol {\beta} = \mathbf {0} \\ \mathbf {X} ^ {T} \mathbf {X} \boldsymbol {\beta} = \mathbf {X} ^ {T} \mathbf {y} \end{array} \right|
$$

Assuming $\mathbf { X } ^ { T } \mathbf { X }$ is invertible:

$$
\boxed {\hat {\boldsymbol {\beta}} = (\mathbf {X} ^ {T} \mathbf {X}) ^ {- 1} \mathbf {X} ^ {T} \mathbf {y}}
$$

## D Details on Low-Discrepancy Sequences

Quasi-Monte Carlo (QMC) methods, such as the Sobol sequence, are widely used to generate low-discrepancy samples in the unit hypercube, providing improved uniformity over purely random sampling. To obtain samples uniformly distributed on the hypersphere, each QMC point is mapped to a standard normal vector via the inverse cumulative

41

<!-- page 42 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Decay of Error Bound Constant vs. Number of Directions (D=5)

![Chart block](images/page042-block04.png)

Figure 15. Depiction of the expected BCS loss upper bound (thm. 5) for various smoothness values 𝛼. We clearly see that as the smoothness increases (blue to red), as the upper bound decreases more and more rapidly with 𝑀.

Table 3. Performance metrics across different sample sizes from Figure 12

<table><tr><td rowspan="2">Freeze Backbone</td><td rowspan="2">Model Name</td><td colspan="7">Samples per Class</td></tr><tr><td>All</td><td>1</td><td>2</td><td>5</td><td>10</td><td>100</td><td>1000</td></tr><tr><td rowspan="8">No</td><td>LeJEPA (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ConvNeXt-V2 Nano</td><td>82.72</td><td>29.42</td><td>36.65</td><td>50.94</td><td>59.85</td><td>75.34</td><td>81.97</td></tr><tr><td>LeViT-128</td><td>79.41</td><td>18.45</td><td>24.08</td><td>33.11</td><td>41.76</td><td>64.59</td><td>77.59</td></tr><tr><td>ResNet-18</td><td>82.15</td><td>23.34</td><td>31.56</td><td>43.82</td><td>54.64</td><td>73.53</td><td>81.41</td></tr><tr><td>ResNet-34</td><td>83.28</td><td>24.27</td><td>31.51</td><td>44.23</td><td>53.95</td><td>74.93</td><td>82.32</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DINOv2 Small</td><td>78.34</td><td>21.05</td><td>21.71</td><td>30.33</td><td>36.23</td><td>60.81</td><td>75.55</td></tr><tr><td>DINOv3 ViT-S/16</td><td>81.60</td><td>24.71</td><td>29.43</td><td>37.71</td><td>44.71</td><td>69.87</td><td>80.54</td></tr><tr><td rowspan="8">Yes</td><td>LeJEPA (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ConvNeXt-V2 Nano</td><td>76.52</td><td>28.74</td><td>36.65</td><td>50.60</td><td>59.50</td><td>72.62</td><td>77.24</td></tr><tr><td>LeViT-128</td><td>69.00</td><td>25.85</td><td>33.30</td><td>45.52</td><td>52.43</td><td>64.37</td><td>69.39</td></tr><tr><td>ResNet-18</td><td>75.95</td><td>30.48</td><td>38.22</td><td>50.85</td><td>58.86</td><td>72.70</td><td>76.39</td></tr><tr><td>ResNet-34</td><td>78.17</td><td>31.08</td><td>38.33</td><td>52.26</td><td>60.63</td><td>74.77</td><td>78.62</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DINOv2 Small</td><td>67.62</td><td>27.68</td><td>32.22</td><td>40.72</td><td>47.72</td><td>62.49</td><td>67.89</td></tr><tr><td>DINOv3 ViT-S/16</td><td>71.38</td><td>30.17</td><td>36.65</td><td>45.74</td><td>51.51</td><td>65.90</td><td>71.35</td></tr></table>

Table 4. Top 1 accuracy (in %) with LeJEPA pretraining on Imagenet-100 for 400 epochs (All values are percentages)

<table><tr><td rowspan="2">w/ predictor</td><td rowspan="2">backbone Projector w/ SWA</td><td colspan="3">resnet50</td><td colspan="3">vit_small_patch8_224</td><td colspan="3">vit_tiny_patch8_224</td></tr><tr><td>1-layer</td><td>2-layer</td><td>3-layer</td><td>1-layer</td><td>2-layer</td><td>3-layer</td><td>1-layer</td><td>2-layer</td><td>3-layer</td></tr><tr><td rowspan="2">False</td><td>False</td><td>79.71</td><td>82.44</td><td>83.93</td><td>76.59</td><td>80.77</td><td>81.07</td><td>71.79</td><td>76.87</td><td>80.37</td></tr><tr><td>True</td><td>79.79</td><td>82.69</td><td>83.50</td><td>79.96</td><td>83.63</td><td>84.12</td><td>75.86</td><td>82.36</td><td>80.50</td></tr><tr><td rowspan="2">True</td><td>False</td><td>79.41</td><td>82.44</td><td>83.57</td><td>77.58</td><td>79.41</td><td>81.91</td><td>67.74</td><td>77.64</td><td>80.73</td></tr><tr><td>True</td><td>78.87</td><td>82.04</td><td>82.82</td><td>77.11</td><td>81.77</td><td>82.58</td><td>69.53</td><td>78.27</td><td>79.77</td></tr></table>

distribution function (CDF), and then projected onto the sphere by normalization. This approach leverages the rotational invariance of the multivariate normal distribution, ensuring that the resulting directions are uniformly distributed on

42

<!-- page 43 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Table 5. Small architecture in-domain LeJEPA pretraining from random initialization across datasets and architectures, with frozen backbone linear evaluation. First, LeJEPA is able to produce near state-of-the-art performances on tiny dataset with only a thousand samples, e.g., flowers102. Second, on non-natural image data, LeJEPA clearly outperforms the latest frontier vision models, e.g., Galaxy10. See Figure 12 for additional experiments with varying number of training samples and with full finetuning.

|  | Pretraining # train. samples | flowers1021020 | cifar10050000 | food10175750 | inet1013000 | cifar1050000 | galaxy1011008 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LeJEPA (convnextv2_nano) 14M | in-domain | 64.34 | 69.26 | 69.59 | 90.81 | 92.22 | 76.05 |
| LeJEPA (resnet18) 11M | in-domain | 74.57 | 69.94 | 73.57 | 92.36 | 92.51 | 75.32 |
| LeJEPA (resnet34) 21M | in-domain | 71.85 | 70.44 | 74.95 | 92.80 | 93.16 | 77.29 |
| LeJEPA (resnext26ts) 8M | in-domain | 82.19 | 69.10 | 76.77 | 92.82 | 91.59 | 73.78 |
| LeJEPA (swin_tiny) 27M | in-domain | 63.94 | 65.08 | 78.40 | 92.87 | 92.67 | 74.89 |
| ĲEPA-inet22k (ViT-H/14) 630M | inet1k | 85.76 | 86.93 | 81.06 | 98.65 | 97.77 | 62.93 |

Table 6. Time (in millisecond) to compute the proposed SIGReg loss from algorithm 1 on a Tesla V100-SXM2-16GB for varying mini-batch size (𝑁), number of slices (𝑀), integration points. Results are computed over 10 runs.

| N | M | # integration points | mean (ms) | std (ms) |
| --- | --- | --- | --- | --- |
| 512 | 512 | 16 | 0.465236 | 0.011642 |
| 512 | 512 | 64 | 0.461317 | 0.003894 |
| 512 | 512 | 256 | 0.627644 | 0.003337 |
| 2048 | 512 | 16 | 1.406441 | 0.002415 |
| 8192 | 512 | 16 | 6.188304 | 0.007226 |
| 8192 | 8192 | 16 | 8.685009 | 0.038829 |
| 32768 | 512 | 16 | 26.373118 | 0.012732 |
| 512 | 2048 | 16 | 0.465614 | 0.005274 |
| 512 | 8192 | 16 | 0.670379 | 0.006854 |

Table 7. Number of Figure 8.

<table><tr><td colspan="13">resnet50</td></tr><tr><td>λ#views</td><td>0.001</td><td>0.005</td><td>0.010</td><td>0.020</td><td>0.025</td><td>0.050</td><td>0.100</td><td>0.150</td><td>0.200</td><td>0.300</td><td>0.400</td><td>0.500</td></tr><tr><td>2</td><td>81.41</td><td>82.73</td><td>83.49</td><td>82.99</td><td>82.23</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>4</td><td>79.88</td><td>83.04</td><td>84.36</td><td>84.68</td><td>84.33</td><td>83.00</td><td>82.91</td><td>81.05</td><td>78.58</td><td>-</td><td>-</td><td>-</td></tr><tr><td>8</td><td>76.67</td><td>81.58</td><td>83.59</td><td>83.49</td><td>83.76</td><td>84.32</td><td>83.66</td><td>83.07</td><td>82.16</td><td>81.00</td><td>79.25</td><td>77.72</td></tr></table>

the sphere’s surface. While the low-discrepancy property is not strictly preserved under this nonlinear mapping, the resulting samples are empirically more uniform than random samples and are standard in high-dimensional applications Marsaglia [1972], Dick and Pillichshammer [2010], Caflisch [1998].

**Require:** Number of points 𝑁, dimension 𝑑

**Ensure:** Points $\{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { N }$ quasi-uniformly distributed on $\mathbb { S } ^ { d - 1 }$

1: **for** 𝑖 = 1 to 𝑁 **do**

2: Generate $\pmb { x } _ { i } \in [ 0 , 1 ] ^ { d }$ as the 𝑖-th point of a Sobol sequence

3: Transform each component: $\widetilde{z_{i,j}} = \Phi^{-1}(x_{i,j}) \;  for  \; j = 1, \cdots, d \;  雲雲  \triangleright \Phi^{-1}$ is the inverse CDF of the standard normal 4: Normalize: $\mathbf { y } _ { i } = \mathbf { z } _ { i } / \| \mathbf { z } _ { i } \| _ { 2 }$

5: **end for**

## E Shapiro-Wilk Test

Let $X 1 < X 2 < \cdots <$ Xn denote an ordered random sample of size n from a standard normal distribution. Also, let mÂ 5 $( { \mathfrak { m } } 1 { \mathfrak { , } } { \mathfrak { m } } 2 { \mathfrak { , } } { \dots } { \mathfrak { , } } { \mathfrak { m } } { \mathfrak { n } } )$ be the vector of expected values of standard normal order statistics, and let $\mathrm { ~ V ~ } 5 \left( \mathrm { v i j } \right)$ be the corresponding

43

<!-- page 44 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

n 3 n covariance matrix, so that

$$
E \left(X _ {i}\right) = m _ {i} \quad \text {and} \quad \operatorname{cov} \left(X _ {i}, X _ {j}\right) = v _ {i j}, \quad i, j = 1, 2, \dots , n\tag{50}
$$

The W test statistic Shapiro and Wilk [1965] for normality is then denoted by

$$
\begin{array}{l} W = \frac {\left(\sum_ {i = 1} ^ {n} a _ {i} Y _ {i}\right)}{\sum_ {i = 1} ^ {n} \left(Y _ {i} - \bar {Y}\right) ^ {2}} = \frac {(\mathbf {a Y})}{S ^ {2}} \\ \mathbf {a} ^ {\prime} = (a _ {1}, a _ {2}, \ldots , a _ {n}) = \mathbf {m V} ^ {- 1} \left(\mathbf {m V} ^ {- 1} \mathbf {V} ^ {- 1} \mathbf {m}\right) ^ {- 1 / 2} \\ S ^ {2} = \sum_ {i = 1} ^ {n} \left(Y _ {i} - \bar {Y}\right) ^ {2} \end{array}\tag{51}
$$

Shapiro and Francia [1972] suggested replacing the covariance matrix V by the identity matrix I, because for large samples, the observations Yi may be treated as if they are independent (see Gupta [1952]). Another asymptotic extension was suggested by Weisburg and Binham [1975]

$$
E \left(X _ {i}\right) = m _ {i} \approx \Phi^ {- 1} \left(\frac {i - \frac {3}{8}}{n + \frac {1}{4}}\right) \quad i = 1, 2, \dots , n\tag{52}
$$

building atop Elfving $[ 1 9 4 7 ]$ ’s approximation but using $3 / 8$ instead of $\pi / 8 .$

Rahman and Govindarajulu [1997] proposed another variation using the approximation for the expected values of order statistics given by Blom [1958] and the approximations for the elements of the variance± covariance matrix given by Blom [1958], Mosteller [2006]. These approximations are

$$
E \left(X _ {i}\right) = m _ {i} \approx \Phi^ {- 1} \left(\frac {i}{N + 1}\right), \quad i = 1, 2, \dots , n\tag{53}
$$

$$
\operatorname{cov} \left(X _ {i}, X _ {j}\right) = v _ {i j} \approx \frac {p _ {i} p _ {j}}{(n + 2) f \left(m _ {i}\right) f \left(m _ {j}\right)}, \quad i, j = 1, 2, \dots , n\tag{54}
$$

$$
p _ {i} = \frac {i}{n + 1}\tag{55}
$$

We know (see Hammersley and Morton [1954], Plackett [1958])

$$
\begin{array}{c} \mathbf {V} ^ {- 1} = (n + 1) (n + 2) \\ \times \left( \begin{array}{c c c c c c} 2 \phi^ {2} (m _ {1}) & - \phi (m _ {1}) \phi (m _ {2}) & 0 & 0 & \dots & 0 \\ - \phi (m _ {1}) \phi (m _ {2}) & 2 \phi^ {2} (m _ {2}) & - \phi (m _ {2}) \phi (m _ {3}) & 0 & \dots & 0 \\ 0 & - \phi (m _ {2}) \phi (m _ {3}) & 2 \phi^ {2} (m _ {3}) & - \phi (m _ {3}) \phi (m _ {4}) & \dots & 0 \\ \vdots & & & & \\ 0 & 0 & 0 & 0 & \dots & 2 \phi^ {2} (m _ {n}) \end{array} \right) \end{array}\tag{56}
$$

## F Multivariate Statistics

We ideally would like to compare the distributions. One slight variation is to compare the Characteristic function of the distributions. Given samples ${ \pmb x } _ { 1 } , \cdots , { \pmb x } _ { N }$ , the Empirical Characteristic Function (ECF) is defined as

$$
\hat {\psi} _ {N} (\pmb {t}) = \frac {1}{N} \sum_ {n = 1} ^ {N} e ^ {- i \pmb {t} ^ {\top} \pmb {y} _ {n}}.
$$

We can now compare our ECF to the one of the target distribution and build the statistic

$$
N \int | \hat {\psi} _ {N} (\boldsymbol {t}) - \psi_ {0} (\boldsymbol {t}) | ^ {2} \omega (\boldsymbol {t}) d t = N \int | \hat {\psi} _ {N} (\boldsymbol {t}) - e ^ {- \| \boldsymbol {t} \| _ {2} / 2} | ^ {2} \omega (\boldsymbol {t}) d t,
$$

44

<!-- page 45 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

if the weighting function is given by $\omega ( \pmb { t } ) = ( 2 \pi \beta ^ { 2 } ) ^ { - d / 2 } e ^ { - \frac { \| \pmb { t } \| _ { 2 } ^ { 2 } } { 2 } }$ then the following simplification can be made

$$
\begin{array}{l} \mathrm{BHEP} _ {n, \beta} = \frac {1}{n} \sum_ {j, k = 1} ^ {n} \exp \left(- \frac {\beta^ {2} \left\| Y _ {n , j} - Y _ {n , k} \right\| ^ {2}}{2}\right) \\ \qquad - \frac {2}{(1 + \beta^ {2}) ^ {d / 2}} \sum_ {j = 1} ^ {n} \exp \left(- \frac {\beta^ {2} \left\| Y _ {n , j} \right\| ^ {2}}{2 (1 + \beta^ {2})}\right) + \frac {n}{(1 + 2 \beta^ {2}) ^ {d / 2}}. \end{array}
$$

with $\beta > 0 ,$ , Baringhaus-Henze-Epps-Pulley. From $1$ leading to the HZ test $^ 2$ uses

$$
\beta_ {n} = 2 ^ {- 1 / 2} ((2 d + 1) n / 4) ^ {1 / (d + 4)}\tag{57}
$$

the same can be done with the moment generating function $^ 3$

$$
\begin{array}{r} T _ {n, \beta} = \pi^ {d / 2} \left(\frac {1}{n} \sum_ {i, j = 1} ^ {n} \frac {1}{\beta^ {d / 2}} \exp \left(\frac {\left\| Y _ {n , i} + Y _ {n , j} \right\| ^ {2}}{4 \beta}\right) + \frac {n}{(\beta - 1) ^ {d / 2}} \right. \\ \left. - 2 \sum_ {j = 1} ^ {n} \frac {1}{(\beta - 1 / 2) ^ {d / 2}} \exp \left(\frac {\left\| Y _ {n , j} \right\| ^ {2}}{4 \beta - 2}\right)\right), \end{array}
$$

here with $\beta > 2$

There is also one combining both<sup>4</sup>!

$$
\begin{array}{l} T _ {n, \gamma} := \int_ {\mathbb {R} ^ {d}} U _ {n} ^ {2} (t) w _ {\gamma} (t) \mathrm{d} t \\ U _ {n} (t) := \sqrt {n} (R _ {n} (t) M _ {n} (t) - 1) \end{array}\tag{58}
$$

$$
\left|\begin{array}{c}T _ {n, \gamma} = \left(\frac {\pi}{\gamma}\right) ^ {d / 2} \Bigg \{\frac {1}{2 n ^ {3}} \sum_ {j, k, l, m = 1} ^ {n} \left[ \right. \exp \left(\frac {\left\| Y _ {j k} ^ {+} \right\| ^ {2} - \left\| Y _ {\ell m} ^ {-} \right\| ^ {2}}{4 \gamma}\right) \cos \left(\frac {Y _ {j k} ^ {+ \top} Y _ {\ell m} ^ {-}}{2 \gamma}\right)\\+ \exp \left(\frac {\left\| Y _ {j k} ^ {+} \right\| ^ {2} - \left\| Y _ {\ell m} ^ {+} \right\| ^ {2}}{4 \gamma}\right) \cos \left(\frac {Y _ {j k} ^ {+ \top} Y _ {\ell m} ^ {+}}{2 \gamma}\right) \Bigg ]\\- \frac {2}{n} \sum_ {j, k = 1} ^ {n} \exp \left(\frac {\left\| Y _ {n , j} \right\| ^ {2} - \left\| Y _ {n , k} \right\| ^ {2}}{4 \gamma}\right) \cos \left(\frac {Y _ {n , j} ^ {\top} Y _ {n , k}}{2 \gamma}\right) + n \Bigg \},\end{array}\right|\tag{59}
$$

and its simplified version

$$
\widetilde {T} _ {n, \gamma} := \int_ {\mathbb {R} ^ {d}} U _ {n} (t) w _ {\gamma} (t) \mathrm{d} t.\tag{60}
$$

$$
\widetilde {T} _ {n, \gamma} = \left(\frac {\pi}{\gamma}\right) ^ {d / 2} \sqrt {n} \left(\frac {1}{n ^ {2}} \sum_ {j, k = 1} ^ {n} \exp \left(\frac {\left\| Y _ {n , j} \right\| ^ {2} - \left\| Y _ {n , k} \right\| ^ {2}}{4 \gamma}\right) \cos \left(\frac {Y _ {n , j} ^ {\top} Y _ {n , k}}{2 \gamma}\right) - 1\right)\tag{61}
$$

Also one testing the derivative <sup>5</sup>

$$
\mathrm{HV} _ {n, \gamma} := n \int \| \nabla M _ {n} (t) - t M _ {n} (t) \| ^ {2} \widetilde {w} _ {\gamma} (t) \mathrm{d} t\tag{62}
$$

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>[https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203?srsltid=AfmBOoodlL-CtlqL0JVC-LcP6mOWw6VTt51\_YstdZOW4W3iuicu1VFyg](https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203?srsltid=AfmBOoodlL-CtlqL0JVC-LcP6mOWw6VTt51_YstdZOW4W3iuicu1VFyg)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>[https://www.tandfonline.com/doi/abs/10.1080/03610929008830400](https://www.tandfonline.com/doi/abs/10.1080/03610929008830400)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>3</sup>[https://arxiv.org/pdf/1711.07199](https://arxiv.org/pdf/1711.07199)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>4</sup>[https://arxiv.org/pdf/1706.03029](https://arxiv.org/pdf/1706.03029)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>5</sup>[https://arxiv.org/pdf/1901.03986](https://arxiv.org/pdf/1901.03986)</span></small>

45

<!-- page 46 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

$$
\mathrm{HV} _ {n, \gamma} = \frac {1}{n} \left(\frac {\pi}{\gamma}\right) ^ {d / 2} \sum_ {j, k = 1} ^ {n} \exp \left(\frac {\left\| Y _ {n , j , k} ^ {+} \right\| ^ {2}}{4 \gamma}\right) \left(Y _ {n, j} ^ {\top} Y _ {n, k} - \frac {\left\| Y _ {n , j , k} ^ {+} \right\| ^ {2}}{2 \gamma} + \frac {d}{2 \gamma} + \frac {\left\| Y _ {n , j , k} ^ {+} \right\| ^ {2}}{4 \gamma^ {2}}\right).\tag{63}
$$

skewness $^ 6 ;$

$$
b _ {1, d} = \frac {1}{n ^ {2}} \sum_ {j, k = 1} ^ {n} \left(Y _ {n, j} ^ {\top} Y _ {n, k}\right) ^ {3}\tag{64}
$$

skewness $^ 7 ;$

$$
\widetilde {b} _ {1, d} = \frac {1}{n ^ {2}} \sum_ {j, k = 1} ^ {n} Y _ {n, j} ^ {\top} Y _ {n, k} \left\| Y _ {n, j} \right\| ^ {2} \left\| Y _ {n, k} \right\| ^ {2}\tag{65}
$$

$$
b _ {2, d} = \frac {1}{n} \sum_ {j = 1} ^ {n} \left\| Y _ {n, j} \right\| ^ {4}
$$

which should be 0 for Gaussian and Kurtosis which should be d(d+2)

(66)

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>6</sup>[https://www.jstor.org/stable/2334770](https://www.jstor.org/stable/2334770)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>7</sup>[https://link.springer.com/article/10.1007/s13171-020-00211-6](https://link.springer.com/article/10.1007/s13171-020-00211-6)</span></small>

46

<!-- page 47 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

dimension=128, slices=10

![Chart block](images/page047-block04.png)

Figure 16. Reprise of Figure 6 for additional dimensions and number of 1d projections.

47

<!-- page 48 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Distribution of Estimator βˆ (Binary Y) (Normalized)

![Chart block](images/page048-block04.png)

Distribution of Estimator βˆ (Linear Y)

![Chart block](images/page048-block06.png)

Distribution of Estimator βˆ (Smooth Y)

![Chart block](images/page048-block08.png)

Figure 17. Depiction of the distribution of optimized 𝛽 values from OLS when comparing $Z _ { \mathrm { i s o } }$ and $Z _ { \mathrm { a n i s o } }$ from lemmas. 1 and 2. We clearly observe that the anisotropic version (blue) provides much lower variance compared to the isotropic case (red). We consider a binary classification (linear separable class) (top row), a linear regression task (middle row), and a nonlinear regression task with smooth targets (bottom row). For each case, we resample the training samples numerous times and produce an estimate for $\cdot _ { \beta }$ each time. Because the data is 2-dimensional, we can visualize the $\beta$ distribution directly.

48

<!-- page 49 of 50 -->

LeJEPA:

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Test Accuracy vs Regularization (N = 50, Linear Classification)

![Chart block](images/page049-block04.png)

Directional Alignment vs Regularization (N = 50, Linear Regression)

Test Accuracy vs Regularization (N = 200, Linear Classification)

![Chart block](images/page049-block07.png)

Test Accuracy vs Regularization (N = 1000, Linear Classification)

![Chart block](images/page049-block09.png)

![Chart block](images/page049-block10.png)

Directional Alignment vs Regularization (N = 200, Linear Regression)

![Chart block](images/page049-block12.png)

Directional Alignment vs Regularization $( N = 1 0 0 0 ,$ Linear Regression)

![Chart block](images/page049-block14.png)

Figure 18. Depiction of accuracy (top) and cosine similarity between estimated and true estimator (bottom) for the OLS setting with varying strength of Tikhonov regularization (x-axis) comparing isotropic and anisotropic embeddings. As per thm. 6, the anisotropic distribution creates a bias in the OLS estimation for nonzero regularization.

Spearman corr.: 98.53% (resnet50 galaxy10)

![Chart block](images/page049-block17.png)

![Chart block](images/page049-block18.png)

Spearman corr.: 94.52% (ViT/base-8 inet1k)

![Chart block](images/page049-block20.png)

![Chart block](images/page049-block21.png)

Spearman corr.: 97.97% (ViT/s-8 inet10)

![Chart block](images/page049-block23.png)

Figure 19. Additional figures provides in Figure 19

Spearman corr.: 93.82% (resnet18 flowers102)

![Chart block](images/page049-block26.png)

49

<!-- page 50 of 50 -->

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

LeJEPA:

![Chart block](images/page050-block03.png)

![Chart block](images/page050-block04.png)

Figure 20. Proposed trapezoid quadrature for the Epps-Pulley statistic as implemented in algorithm 1. We depict the approximation error of the integral for various distributions, demonstrate rapid convergence (faster than quadratic show in grey line) across possible embedding distributions.

![Chart block](images/page050-block06.png)

![Chart block](images/page050-block07.png)

![Chart block](images/page050-block08.png)

Figure 21. Additional figures for Figure 10.

![Chart block](images/page050-block10.png)

50