<!-- page 1 of 16 -->

arXiv:2602.24222v1 [cs.CV] 27 Feb 2026

# MUVIT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy

Albert Dominguez Mantes

Gioele La Manno†

Swiss Federal Institute of Technology

Lausanne, Switzerland

{albert.dominguezmantes,gioele.lamanno}@epfl.ch

Martin Weigert† Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI) Technische Universitat Dresden,Germany ¨

martin.weigert@tu-dresden.de

## Abstract

Modern microscopy routinely produces gigapixel images that contain structures across multiple spatial scales, from fine cellular morphology to broader tissue organization. Many analysis tasks require combining these scales, yet most vision models operate at a single resolution or derive multi-scale features from one view, limiting their ability to exploit the inherently multi-resolution nature of microscopy data. We introduce MUVIT, a transformer architecture built to fuse true multi-resolution observations from the same underlying image. MUVIT embeds all patches into a shared worldcoordinate system and extends rotary positional embeddings to these coordinates, enabling attention to integrate widefield context with high-resolution detail within a single encoder. Across synthetic benchmarks, kidney histopathology, and high-resolution mouse-brain microscopy, MUVIT delivers consistent improvements over strong ViT and CNN baselines. Multi-resolution MAE pretraining further produces scale-consistent representations that enhance downstream tasks. These results demonstrate that explicit worldcoordinate modelling provides a simple yet powerful mechanism for leveraging multi-resolution information in largescale microscopy analysis.

## 1. Introduction

Modern microscopy techniques generate images at unprecedented scale and complexity, with modalities such as lightsheet fluorescence microscopy [11, 38, 39], electron microscopy [23, 36], and digital pathology [5, 37] routinely

producing gigapixel-scale images exceeding 50K × 50K pixels. These images capture biological structures with hierarchical organization across spatial scales, from individual cells to tissue architecture to anatomical regions. A central challenge in the computational analysis of such images is that many tasks require information across these scales simultaneously. For instance, accurate semantic segmentation of local cellular structures often depends on broader anatomical context, e.g. knowing whether a cell is part of a particular tissue region can be essential for correct classification. In the last decade, deep learning approaches based on CNNs [26, 27, 29] or vision transformers [1, 8] have emerged as the standard for segmentation and detection in microscopy. These approaches, however, typically operate via tiled prediction on single-resolution crops (e.g. of size 512×512 pixels) for memory reasons, which restricts the available context to the processed tile. As a consequence, they must trade off field of view against spatial resolution, limiting their ability to access fine detail and global context simultaneously.

In this work, we propose MUVIT (**Mu**lti-Resolution **Vi**sion Transformer), an architecture that jointly processes several crops of the same image captured at different physical resolutions. Throughout this work, we use the term resolution to denote the physical scale of an observation (coarse versus fine), rather than differences in input pixel dimensions. Although some transformers are called resolutionindependent, this usually refers to handling different input sizes rather than modeling multiple physical scales [9]. Likewise, hierarchical architectures that construct feature pyramids from a single input [21] and models that derive multi-scale features only through internal downsampling, such as U-Net [26] or pyramidal vision transformers [34], do not

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">†Co-corresponding authors.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">Code: [github.com/weigertlab/muvit](https://github.com/weigertlab/muvit)</span></small>

1

<!-- page 2 of 16 -->

![Image block](images/page002-block01.png)

Figure 1. Overview of the MUVIT architecture. a) MUVIT processes multiple crops of the same scene at different physical resolutions and encodes them jointly using a shared transformer with resolution-specific RoPE position encodings. As a decoder we use b) a lightweight masked autoencoding decoder that reconstructs the masked patches for each resolution level, or c) a semantic segmentation decoder that predicts the class of each pixel at the target resolution (e.g. level 1).

access true multi-resolution information.

Our design builds on the idea that different spatial scales can be treated as complementary input “modalities” similar to multi-modal encoders such as MultiMAE [2], as long as they are linked through a shared geometric reference frame. MUVIT accepts spatially related crops that share pixel size but differ in field of view. To ensure geometric consistency across these crops, we incorporate common world-coordinates (which we define as the pixel coordinate system of the highest-resolution input) per token using Rotary Position Embeddings (RoPE) [30] which applies these coordinates directly within the internal query and key projections. This allows MUVIT to perform explicit cross-resolution attention across different physical scales. Our contributions are: i) We propose MUVIT, a multi-resolution Vision Transformer architecture that jointly processes true multi-resolution observations within a single encoder, in contrast to hierarchical methods that construct feature pyramids from a single-resolution input. ii) We incorporate absolute world coordinates into attention via RoPE, enabling cross-resolution interactions without crop alignment, showing that accurate coordinate relations are crucial for effective downstream performance. iii) We extend MAE pretraining to the multi-resolution setting and show that adding resolution levels yields increasingly informative representations. Multi-resolution MAE also accelerates downstream training, with segmentation models converging within only a few epochs. iv) We demonstrate strong downstream performance of the resulting MUVIT backbone on both synthetic and large-scale microscopy tasks, achieving strong perfor-

mance on the KPIS benchmark [32] and substantial gains over single-resolution encoders on mouse brain anatomy segmentation.

## 2. Related work

**Multi-scale learning with transformers.** Existing transformer-based methods incorporate multi-scale information in fundamentally different ways. Some approaches construct all scales internally from a single input resolution, such as hierarchical or pyramidal models like Swin [22], PVT [34], and HIPT [5]. Others train on multiple input resolutions for robustness but still infer at a single resolution, as in ResFormer [33]. Parallel multi-path models process artificially created scale variants through separate branches, including MMViT [21], CrossViT [3], MPViT [20], and MUSIQ [17]. All of these derive multi-scale features from either a single crop or independent inputs and lack explicit geometric correspondence across scale levels, as is also the case for global-local fusion approaches [6]. In contrast, MUVIT integrates multiple physically distinct resolutions extracted from a larger image within a unified encoder, using world-coordinate embeddings to align their geometry and enable cross-scale attention.

**Masked autoencoder pre-training.** Masked autoencoders (MAEs) [13] pre-train a ViT encoder by reconstructing heavily masked inputs using a lightweight decoder and have proven effective across a wide range of vision tasks. MultiMAE [2] extends this paradigm to multi-modal inputs by

2

<!-- page 3 of 16 -->

| Method | Hierarchical Stages | Physical Multi-Res Input | Joint Attention | World Coordinate Alignment |
| --- | --- | --- | --- | --- |
| MultiMAE | ✗ | ✗ | ✓ | ✗ |
| Scale-MAE | ✗ | ✗ | ✗ | ✗ |
| PVT / Swin | ✓ | ✗ | ✗ | ✗ |
| HIPT | ✓ | ✓ | ✗ | ✗ |
| MUVIT (Ours) | ✗ | ✓ | ✓ | ✓ |

Table 1. Conceptual comparison of relevant related work.

treating different modalities (e.g., RGB and depth) as separate token types with cross-modality reconstruction. Nevertheless, it assumes trivial spatial alignment at a single resolution. Motivated by this, MUVIT treats different spatial scales as distinct but geometrically related input types and uses Dirichlet-weighted masking across resolution levels to promote cross-scale learning within a shared coordinate frame. Scale-MAE [25] addresses heterogeneous image resolutions in geospatial datasets by encoding scale (via GSD) on single-resolution inputs, but does not perform joint encoding or geometric alignment across multiple FOVs. Cross-Scale MAE [31] enforces consistency across scale-augmented views of the same image. Kraus et al. [19] adapt MAEs to fluorescence microscopy, but only for single-resolution inputs. Unlike these approaches, MUVIT performs joint masked reconstruction across physically distinct resolution levels of the same scene using world-coordinate RoPE to maintain geometric alignment between scales.

**Rotary positional embeddings in vision.** While RoPE [30] has become the default positional encoding for large language models, most ViT architectures still use alternative schemes such as fixed Fourier features [8, 18]. Early adaptations of RoPE to vision include EVA-02 [10], which applies RoPE independently to each spatial axis, and RoPE-Mixed [15], which learns axis-frequency mixtures to better capture diagonal structure in high-resolution images. Further extensions such as ComRoPE [40] and LieRE [24] propose more general parametrizations through trainable matrices and Lie group formulations. In contrast to these design-oriented extensions, MUVIT uses standard axis-wise RoPE in a new setting: the rotation angles are derived from each tokens world-coordinates, enabling geometrically consistent attention across resolution levels of the same scene.

## 3. Methods

## 3.1. Input and spatial representation

MUVIT receives multi-resolution image crops and their spatial extents as input. The input consists of a tuple $( \mathbf { X } , { \mathcal { B } } )$ where $\mathbf { X } \in \mathbb { R } ^ { L \times \overleftarrow { C } \times H \times W }$ contains the image crops at L resolution levels, $\mathcal { B }   \in   \mathbb { R } ^ { L \times 2 \times 2 }$ defines the bounding box of each crop, C is the number of channels, and $H , W$ are the spatial dimensions in pixels. $\mathcal { L } = [ l _ { 1 } , l _ { 2 } , \ldots , l _ { L } ]$ with $l _ { i } \in \mathbb { N }$

specifies the downsampling factor for each resolution level. We use $l = 1$ for the finest resolution; $l > 1$ denotes l× downsampling, so an $H \times W$ crop at level l spans the same scene region as an $( l H ) \times ( l W )$ crop at level 1.

H and W remain constant across levels in pixel space, so lower resolution levels capture larger spatial context at the cost of fine-grained detail. We denote by $\mathbf { M U V I T } _ { [ l _ { 1 } , l _ { 2 } , \ldots ] }$ an encoder using resolution levels $l _ { 1 } , l _ { 2 } , \ldots$ so that a $\mathrm { { M U V I T } _ { [ 1 , 8 ] } }$ encoder processes the original resolution and an $8 \times$ downsampled version. The bounding box $\mathcal { B } _ { l } \; = \;$ $\left( \left( \mathbf{y}_{l}^{\min}, \mathbf{x}_{l}^{\min} \right), \left( \mathbf{y}_{l}^{\max}, \mathbf{x}_{l}^{\max} \right) \right)$ defines the field of view for each crop in world coordinates, which we define as the pixel co ordinate system of the highest-resolution input (i.e. level $l = 1 )$ .

**Bounding box generation.** Bounding boxes for all resolution levels are derived from the same world-coordinate extent, ensuring geometric alignment. During training, we sample nested crops at arbitrary positions and provide the model with their true bounding box coordinates for accurate spatial encoding. To test the importance of spatial informa tion, we also evaluate a baseline where each level is assigned a fixed, centered bounding box scaled by its downsampling factor (naive bbox), independent of the actual crop location. As shown in Sec. 4 such incorrect bounding box information leads to a substantial performance drop, confirming that accurate spatial information crucial for effective cross-scale learning.

For each resolution level l, non-overlapping patches (tokens) of size $P \times P$ (we use $P   =   8 )$ are extracted and projected to a D-dimensional embedding space through level-specific linear layers followed by layer normalization, denoted as $\mathbf { P E } _ { l }$ . To distinguish tokens from different resolution levels, learnable level embeddings e<sub>l</sub> are added: $\mathbf { z } _ { l } = \operatorname { P E } _ { l } ( \mathbf { X } _ { l } )   +   \mathbf { e } _ { l }$ , where $\mathbf { z } _ { l } \in \mathbb { R } ^ { \tilde { N \times D } }$ with $N   =   \left( H / P \right) \times \left( W / P \right)$ patches per level. In addition to the embedded tokens, we compute spatial coordinates for each patch from the bounding boxes. For a patch at position (i, j) in level l, its center coordinate in world space is: $\begin{array} { r } { \mathbf { p } _ { l , i , j }   =   \mathcal { B } _ { l } ^ { \mathrm { m i n } } + \left( \frac { i } { h - 1 } , \frac { j } { w - 1 } \right)   \odot   \left( \mathcal { B } _ { l } ^ { \mathrm { m a x } } - \mathcal { B } _ { l } ^ { \mathrm { m i n } } \right) } \end{array}$ where $h = H / P$ and $w = \dot { W / P }$ , and ⊙ denotes element-wise multiplication. These coordinates encode the absolute spatial position of each patch in a common world coordinate system, enabling consistent positional encoding across resolution levels. The concatenated tokens $\mathbf { z } = \widetilde { [ \mathbf { z } _ { 1 } ; \ldots ; \mathbf { z } _ { L } ] } \in \mathbb { R } ^ { ( L \cdot N ) \times D }$ and their coordinates $\mathbf { p } \: = \: [ \mathbf { p } _ { 1 } ; \ldots ; \mathbf { p } _ { L } ] \: \in \: \mathbb { R } ^ { ( L \cdot N ) \times 2 }$ are processed by the MUVIT encoder.

## 3.2. MUVIT encoder

The MUVIT encoder consists of K stacked transformer layers (we use $K = 1 2$ for all experiments) that process the full sequence of multi-resolution tokens jointly through a unified architecture. Unlike standard ViT architectures

3

<!-- page 4 of 16 -->

that apply fixed Fourier-based positional encodings to input tokens, MUVIT uses 2D axial Rotary Position Embeddings (RoPE) [10, 30] computed from world coordinates p. For each spatial dimension (y and x), the head dimension $d _ { h } = D / H$ is split into per-axis allocations $d _ { y }$ and $d _ { x }$ such that $d _ { y } + d _ { x } \leq d _ { h }$ . Each transformer layer has independent learnable RoPE parameters. For each coordinate axis $a \in \{ y , x \}$ and patch position $\mathbf { p } _ { l , i , j } ^ { ( a ) }$ , the rotation frequencies are:

$$
\theta_ {k} ^ {(a)} = \mathbf {p} _ {l, i, j} ^ {(a)} / b ^ {2 k / d _ {a}}, \quad k = 0, \dots , d _ {a} / 2 - 1\tag{1}
$$

where the frequencies $b ^ { 2 k / d _ { a } }$ are a learnable parameter (initialized to $b = 1 0 0 0 0 )$ allowing each layer to adapt its positional encoding. Crucially, computing the frequencies from the world coordinates p ensures that patches representing the same spatial location receive identical positional encodings regardless of their resolution level l thus enabling effective cross-scale information flow. This differs from learned positional embeddings, which would treat the same spatial location differently at each resolution. As we show in Sec. 4, accurate world-coordinates are essential: replacing them with naive centered (i.e. inconsistent) coordinates leads to a collapse in performance, even though the architecture and inputs remain unchanged. Nevertheless, trained MUVIT models exhibit a certain degree of robustness to world coordinate misalignment (Sec. 4).

Each transformer layer follows a standard structure with residual connections:

$$
\mathbf {x} ^ {(i)} \leftarrow \mathbf {x} ^ {(i - 1)} + \text {SelfAttn} (\mathrm{LN} (\mathbf {x} ^ {(i - 1)}), \mathbf {p})\tag{2}
$$

$$
\mathbf {x} ^ {(i)} \leftarrow \mathbf {x} ^ {(i)} + \mathrm{FFN} (\mathrm{LN} (\mathbf {x} ^ {(i)}))\tag{3}
$$

where LN denotes layer normalization and FFN is a feedforward network with GELU activation. Given the input image crops and bounding boxes $( \mathbf { X } , { \mathcal { B } } )$ , the encoder produces encoded features $\tilde { \mathbf { h } \in \mathbb { R } ^ { ( L \cdot \dot { N } ) \times D } }$ that capture multi-resolution representations with explicit world-coordinate relationships. We denote by $\mathbf { M U V I T } _ { [ l _ { 1 } , l _ { 2 } , \ldots ] }$ an encoder using levels $l _ { 1 } , l _ { 2 } , \ldots ,$ and by $\mathbf { M } \mathbf { U } \mathbf { V } \mathbf { I } \mathbf { T } _ { [ . . . ] } ^ { \mathbf { \hat { \theta } } }$ +decoder the same encoder paired with a specific decoder (e.g. segmentation). The full encoder is lightweight containing ∼25M parameters.

## 3.3. MUVIT-MAE: Self-supervised pre-training

We adapt masked autoencoding to the multi-resolution setting. During training, a random subset of tokens is masked with a high masking ratio ρ (we use $\rho = 0 . 7 5 )$ . To encourage the model to learn diverse cross-scale configurations, the proportions of visible tokens per level $( \lambda _ { 1 } , \ldots , \lambda _ { L } )$ are sampled from a Dirichlet distribution $\operatorname { D i r } ( \alpha )$ with $\alpha = 0 . 5 \: [ 2 ]$ . The encoder processes only the visible tokens and their worldcoordinates:

$$
\mathbf {h} _ {\mathrm{vis}} = \operatorname{Encoder} (\mathbf {z} _ {\mathrm{vis}}, \mathbf {p} _ {\mathrm{vis}}).\tag{4}
$$

To reconstruct the masked tokens, we use L lightweight decoders, one per resolution level, each consisting of 2 transformer layers. Learnable mask tokens are inserted at masked positions, and the first decoder layer incorporates cross-attention to the full set of visible encoder outputs with coordinate-based RoPE, enabling information flow across scales. Each decoder reconstructs the masked patches for its corresponding resolution level through a linear projection to pixel space. As loss we use mean squared error on masked patches only, averaged across all resolution levels.

## 3.4. MUVIT-seg: Semantic segmentation

For semantic segmentation tasks, we use the pre-trained MUVIT encoder to extract multi-resolution features and add task-specific decoders that we fine-tune or keep frozen during training. We explore two decoder designs: the first, inspired by UNETR [12], progressively upsamples features through a series of convolutional blocks while incorporating skip connections from intermediate encoder layers. For our 12-layer encoder with patch size 8, skip connections are taken from layers [0, 5, 11], corresponding to three decoder stages. At each decoder stage, features from the corresponding encoder layer are concatenated with the upsampled features, combining fine-grained multi-scale information from the encoder with the coarse semantic features from the de coder. The second design follows Mask2Former [7] and uses learnable mask queries that cross-attend to the multi-resolution encoder features. For both architectures, we only use the finest decoder level as the final prediction while using the following loss function:

$$
\mathcal {L} = \lambda_ {\mathrm{CE}} \cdot \mathcal {L} _ {\mathrm{CE}} (\tilde {y}, y) + \lambda_ {\mathrm{Dice}} \cdot \mathcal {L} _ {\mathrm{Dice}} (\tilde {y}, y)\tag{5}
$$

with $\tilde { y }$ the predicted segmentation and $y$ the ground truth (we use $\lambda _ { \mathrm { C E } } = \lambda _ { \mathrm { D i c e } } = 1 . 0 )$ . The loss is computed only at the finest resolution level $( l = 1 )$ , where final segmentation is produced.

## 4. Experiments

We evaluate MUVIT on three datasets that each address different aspects: a synthetic dataset designed to establish a controlled setting for validation of cross-scale behavior, a large-scale anatomical segmentation task requiring strong global/local integration, and a challenging pathology benchmark dataset that allows to externally quantify the performance of the model.

## 4.1. Datasets

**Synthetic dataset (SYNTHETIC).** We generate a synthetic dataset with controlled multi-resolution structure to validate the architecture. The dataset consists of 1638 training images, 220 validation images, and 190 test images of size 2048×2048 pixels (single channel) with procedurally

4

<!-- page 5 of 16 -->

generated concentric ring patterns using Perlin noise. The task is to segment two object classes: outer cells and inner cells. This task requires multi-resolution processing as the pixel class depends on its position relative to the global ring structure, which is only visible at coarser resolutions. This controlled setting allows us to focus on the impact of multi-resolution processing and to validate the benefits of using scale-consistent positional encodings.

**Mouse brain anatomy dataset (MOUSE).** The mouse brain anatomy dataset [28] consists of DAPI-stained microscopy slices from 12 volumetric mouse brains with anatomical region annotations for multi-class semantic segmentation. The dataset contains 618 images of approximately 13k × 9k pixels, which we split into 563 training images and 55 test images. The task is to segment 11 distinct anatomical brain regions (i.e. 12 classes total including the background class). Multi-resolution processing is critical because correctly classifying a pixel requires understanding which global anatomical region it belongs to, while fineresolution information is necessary for precise boundary delineation. We use a specimen-based split with specimen BM15 being held out as the test set.

**Kidney pathology segmentation (KPIS).** The KPIS dataset [32] consists of RGB histopathology whole-slide images (WSIs) of mouse kidney tissue of four different groups of mice models for glomeruli segmentation. The dataset contains 30 training images, 8 validation images, and 12 test images with sizes ranging from $1 6 k   -   7 1 k   \times   1 4 k   -   8 1 k$ pixels. The task to solve consists of identifying pixels of glomerular structures (binary semantic segmentation) across the whole WSIs. We hypothesized that multi-resolution processing is beneficial because glomeruli can be larger than the receptive field of commonly used architectures. They may also be located in specific anatomical regions of the tissue, thus requiring both local texture information for identification and global tissue context for localization. As the WSIs are obtained from four different groups of mice models representing different conditions and stages of chronic kidney disease, we also use this dataset to assess the representation power of MAE pretrained models in a linear probing experiment.

**Implementation details.** All models are implemented in PyTorch and trained using PyTorch Lightning. Full training details, hyperparameters, and optimization settings are provided in Supplementary Section A. During training, we sample nested multi-resolution crops by randomly generating coordinates at the finest scale and then successively sample at coarser scales while ensuring the crop of resolu tion level $l _ { i }$ is always contained in the crop of level $l _ { i + 1 } .$ Due to the large size of the images, we use Zarr pyramids to store the datasets. Zarr’s chunked storage standard allows to load only the sampled positions at multiple scales, enabling efficient memory usage and fast data loading during training

and inference.

## 4.2. Evaluation metrics

We evaluate semantic segmentation performance using the Dice coefficient. The Dice coefficient for class c is defined as $\begin{array} { r } { \mathrm { D S C } _ { c } = \frac { 2 | P _ { c } \cap G _ { c } | } { | P _ { c } | + | G _ { c } | } } \end{array}$

where $P _ { c }$ and $G _ { c }$ are the predicted and ground truth pixels for class c, respectively. For the multi-class case (MOUSE dataset), we report both per-class DICE scores and their macro average (mDSC).

## 4.3. MAE pretraining

We pretrain MUVIT encoders using MAE on all datasets, including SYNTHETIC, MOUSE and KPIS. As shown in Figure 2, MAE models successfully reconstruct masked patches across multiple resolution levels. The Dirichletsampled masking strategy described in Section 3 encourages the model to learn cross-scale relationships by varying the relative masking ratio per level while maintaining an overall masking ratio $( \rho   =   0 . 7 5 )$ . Figure 2 depicts results for different per-level masking distributions, with the model consistently reconstructing fine details at high resolution while maintaining global coherence across scales.

Table 2 shows the reconstruction error for MOUSE and KPIS comparing reconstruction error without (naive) and with the default scale-consistent bounding box coordinates across different resolution levels. The results demonstrate that accurate coordinate alignment is crucial for reconstruction at coarser scales $( e . g .$ levels $l = 8 / 3 2 / 6 4 )$ , where knowing the high resolution view’s location relative to the coarse scales enables improved reconstruction of the latter ones. At the finest resolution $( l = 1 )$ , the model achieves a low reconstruction error even with naive coordinates, suggesting that local texture patterns provide sufficient cues without global context. This suggests coarse-scale tokens rely on world-coordinate alignment to access complementary high resolution information, whereas fine-scale tokens can be reconstructed from local appearance alone.

To evaluate how MAE pretraining provides high-quality initial representations, Table 3 reports validation Dice scores during fine-tuning on MOUSE. Models initialized with MAE-pretrained MUVIT encoders converge dramatically faster

Table 2. MAE ablation study on MOUSE and KPIS. Reconstruction mean squared error is reported on 100 nested multi-resolution samples per image (test set). MUVIT [l] denotes an MAE model that only receives the evaluated level.

<table><tr><td>Dataset</td><td>Method</td><td>Level 1</td><td>Level 8</td><td>Level 32</td><td>Level 64</td></tr><tr><td rowspan="3">MOUSE</td><td> $\text{MUVIT}_{[l]}$ </td><td> $5.192 \times 10^{-3}$ </td><td> $1.628 \times 10^{-2}$ </td><td> $1.671 \times 10^{-2}$ </td><td>-</td></tr><tr><td> $\text{MUVIT}_{[1,8,32]}$  (naive)</td><td> $4.848 \times 10^{-3}$ </td><td> $1.589 \times 10^{-2}$ </td><td> $1.373 \times 10^{-2}$ </td><td>-</td></tr><tr><td> $\text{MUVIT}_{[1,8,32]}$ </td><td> $8.734 \times 10^{-3}$ </td><td> $2.274 \times 10^{-2}$ </td><td> $1.464 \times 10^{-2}$ </td><td>-</td></tr><tr><td rowspan="3">KPIS</td><td> $\text{MUVIT}_{[l]}$ </td><td> $4.352 \times 10^{-3}$ </td><td> $1.122 \times 10^{-2}$ </td><td> $1.013 \times 10^{-2}$ </td><td> $1.021 \times 10^{-2}$ </td></tr><tr><td> $\text{MUVIT}_{[1,8,32,64]}$  (naive)</td><td> $3.410 \times 10^{-3}$ </td><td> $9.998 \times 10^{-3}$ </td><td> $8.816 \times 10^{-3}$ </td><td> $7.302 \times 10^{-3}$ </td></tr><tr><td> $\text{MUVIT}_{[1,8,32,64]}$ </td><td> $3.268 \times 10^{-3}$ </td><td> $9.387 \times 10^{-3}$ </td><td> $7.941 \times 10^{-3}$ </td><td> $6.913 \times 10^{-3}$ </td></tr></table>

5

<!-- page 6 of 16 -->

a

b

![Image block](images/page006-block03.png)

Figure 2. MUVIT-MAE reconstruction results on a) MOUSE and b) KPIS. Columns show masked input and reconstruction at multiple resolution levels $( l _ { 1 } = 1$ top, $l _ { 2 } = 8$ middle, $l _ { 3 } = 3 2$ bottom). The same overall masking ratio $( \rho = 0 . 7 5 )$ is used across examples, but each column demonstrates different Dirichlet-sampled per-level masking distributions.

![Image block](images/page006-block05.png)

Figure 3. Semantic Segmentation results on SYNTHETIC. Columns show: input image, ground truth, and semantic predictions for different architectures (U-Net, DeepLabV3, MUVIT [1], MUVIT $_ { \mathrm { [ 1 , 4 ] } } { \mathrm { + U N E T R } }$ (naive, i.e. without cross-scale consistent positional encodings), and MUVIT $_ { \mathrm { [ 1 , 4 ] } } { + } \mathrm { U N E T R ) }$ . Top: full image $( 2 k   \times   2 k \; \mathrm { p x } )$ . Bottom: magnified regions (white boxes) showing detail comparison.

than all baselines: MUVIT $[ 1 { , } 8 { , } 3 2 ] ^ { \cdot }$ +Mask2Former reaches mDSC=0.843 after only 10 epochs and already exceeds the final performance of every single-resolution model (and surpasses 0.88 by epoch 25). In contrast, conventional architectures trained on large $1 0 2 4   \times   1 0 2 4$ inputs show slow and unstable early training, with most models below 0.30 Dice at epoch 10. This demonstrates that MAE pretraining yields high-quality initial features across scales that substantially accelerate optimization and improve sample efficiency.

Table 3. MOUSE training progress results. Validation mDSC (11 classes) is reported at different epochs during training.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Input Size</td><td colspan="4">Epoch</td></tr><tr><td>E1</td><td>E10</td><td>E25</td><td>E50</td></tr><tr><td>U-Net</td><td> $1024 \times 1024$ </td><td>0.086</td><td>0.110</td><td>0.288</td><td>0.412</td></tr><tr><td>DeepLabV3</td><td> $1024 \times 1024$ </td><td>0.234</td><td>0.582</td><td>0.634</td><td>0.704</td></tr><tr><td>SegFormer</td><td> $1024 \times 1024$ </td><td>0.045</td><td>0.159</td><td>0.266</td><td>0.332</td></tr><tr><td>SwinUNETRV2</td><td> $1024 \times 1024$ </td><td>0.049</td><td>0.232</td><td>0.305</td><td>0.388</td></tr><tr><td>MUViT [1] +Mask2Former</td><td> $256 \times 256$ </td><td>0.033</td><td>0.142</td><td>0.266</td><td>0.310</td></tr><tr><td>MUViT [1] +UNETR</td><td> $256 \times 256$ </td><td>0.105</td><td>0.086</td><td>0.261</td><td>0.281</td></tr><tr><td>MUViT [1,8,32] +Mask2Former</td><td> $3 \times 256 \times 256$ </td><td>0.619</td><td>0.843</td><td>0.880</td><td>0.878</td></tr><tr><td>MUViT [1,8,32] +UNETR</td><td> $3 \times 256 \times 256$ </td><td>0.297</td><td>0.748</td><td>0.852</td><td>0.864</td></tr></table>

## 4.4. Semantic segmentation on SYNTHETIC

To assess the role of multi-resolution processing in a controlled setting, we first evaluate on the SYNTHETIC dataset using two levels ([1,4]). During training we sample nested crops at different resolution levels at random global positions. Level 1 crops are always contained within the level 4 crop but not centered in order to force the model to rely on diverse spatial relationships.

Table 4 shows that $\mathrm { { M U V I T _ { [ 1 , 4 ] } { + } U N E T R } }$ accurately solves the task $( m D S C \; = \; 0 . 9 \dot { 5 } \dot { 3 } \dot { 8 } )$ , whereas all singleresolution baselines can recognize the objects but fail to classify them into the correct class $( m D S C { \approx } 0 . 5 0 )$ . Crucially, training the same MUVIT architecture with incorrect bounding box coordinates (naive) causes the performance to collapse to baseline levels $( m D S C = 0 . 3 8 6 4 )$ . This demonstrates that the scale-consistent positional encoding is an essential factor: without correct spatial alignment the model is not able to integrate cross-level information.

## 4.5. Semantic segmentation on MOUSE

We train several baseline models of different complexities () along with MUVIT on MOUSE. Table 5 shows results on the mouse brain anatomy segmentation task. All methods use

6

<!-- page 7 of 16 -->

![Image block](images/page007-block01.png)

Figure 4. Semantic segmentation results on MOUSE. Columns show: input image, ground truth, U-Net, DeepLabV3, and $\mathrm { { M U V I T _ { [ 1 , 8 , 3 2 ] } \mathrm { { + U N E T R } } } }$ semantic predictions. Top: full image (13k × 10k px). Bottom: magnified regions (white boxes) showing detail comparison.

Table 4. Semantic segmentation results on SYNTHETIC. Reported are Dice scores for each class (outer, inner) and mDSC. All baselines operate at the highest resolution (corresponding to l = 1).

| Method | outer | inner | mDSC |
| --- | --- | --- | --- |
| U-Net | 0.5433 | 0.2600 | 0.4016 |
| DeepLabV3 | 0.5473 | 0.4317 | 0.4895 |
| MUVIT<sub>[1]</sub>+UNETR | 0.5391 | 0.4615 | 0.5003 |
| MUVIT<sub>[1,4]</sub>+UNETR (naive) | 0.4662 | 0.3065 | 0.3864 |
| MUVIT<sub>[1,4]</sub>+UNETR | 0.9499 | 0.9577 | 0.9538 |

tiled prediction at the highest resolution scale, without overlap between tiles. The tile size matches the spatial input that each method was trained on. $\mathbf { M U V I T } _ { [ 1 , 8 , 3 2 ] }$ +Mask2Former achieves a m $\imath D S C = 0 . 9 0 1$ , substantially outperforming all baselines. The same architecture achieves the highest Dice coefficient in all classes but one (Caudoputamen), for which $\mathrm { { \bf M U V I T } _ { [ 1 , 8 , 3 2 ] } { + } \mathrm { { \bf U N E T R } } }$ performs best. The improvement of the $\mathbf { M U V I T } _ { [ 1 , 8 , 3 2 ] }$ variants over the rest of the methods is particularly pronounced when baselines are trained with smaller input sizes (256×256), highlighting again that ne-

cessity of sufficient spatial context for the task.

The best-performing baseline, DeepLabV3 (mDSC = 0.843), requires a substantially larger input size (1024×1024) to achieve such performance, and drops significantly when using input sizes of 512×512 and $2 5 6 \times 2 5 6 ( m D S C =$ 0.676 and 0.520 respectively). In contrast MUVIT achieves superior results to the baselines with much smaller patches $( 3 \times 2 5 6 \times 2 5 6 )$ allowing the global anatomical context to be used through coarser resolution levels without requiring to increase the spatial dimensions of the input (and thus the memory footprint). An example of one of the harder areas to identify is the Septal complex, a small region whose identification depends heavily on global anatomical positioning: $\mathrm { { \bf M } U V I T _ { [ 1 , 8 , 3 2 ] } }$ reaches a Dice of 0.859 compared to 0.734 for DeepLabV3, while most single-scale methods fail to classify it reliably. Figure 4 illustrates the general qualitative improvement brought by MUVIT, showing that our multi-resolution approach correctly identifies precise anatomical boundaries that are missed by single-resolution baselines.

Table 5. Semantic segmentation results on MOUSE (test set). Shown are DSC for each anatomical class as well as averaged (mDSC). All baselines operate at the highest resolution (corresponding to l = 1).

<table><tr><td rowspan="2">Method</td><td rowspan="2">Input Size</td><td colspan="11">Semantic class</td><td rowspan="2">mDSC</td></tr><tr><td>Hippocampus</td><td>Thalamus</td><td>Hypothalamus</td><td>Septal complex</td><td>Caudoputamen</td><td>Dorsal cortex</td><td>Ventral cortex</td><td>Lateral cortex</td><td>Corpus callosum</td><td>Ventricular system</td><td>Midbrain</td></tr><tr><td rowspan="3">U-Net [26]</td><td> $1024 \times 1024$ </td><td>0.777</td><td>0.489</td><td>0.515</td><td>0.112</td><td>0.802</td><td>0.780</td><td>0.546</td><td>0.216</td><td>0.533</td><td>0.480</td><td>0.149</td><td>0.491</td></tr><tr><td> $512 \times 512$ </td><td>0.651</td><td>0.310</td><td>0.409</td><td>0.000</td><td>0.742</td><td>0.677</td><td>0.378</td><td>0.189</td><td>0.447</td><td>0.268</td><td>0.000</td><td>0.370</td></tr><tr><td> $256 \times 256$ </td><td>0.027</td><td>0.000</td><td>0.121</td><td>0.000</td><td>0.274</td><td>0.234</td><td>0.141</td><td>0.000</td><td>0.135</td><td>0.000</td><td>0.000</td><td>0.085</td></tr><tr><td rowspan="3">DeepLabV3 [4]</td><td> $1024 \times 1024$ </td><td>0.941</td><td>0.863</td><td>0.809</td><td>0.734</td><td>0.908</td><td>0.943</td><td>0.874</td><td>0.756</td><td>0.782</td><td>0.857</td><td>0.807</td><td>0.843</td></tr><tr><td> $512 \times 512$ </td><td>0.894</td><td>0.675</td><td>0.660</td><td>0.278</td><td>0.851</td><td>0.878</td><td>0.721</td><td>0.472</td><td>0.670</td><td>0.747</td><td>0.585</td><td>0.676</td></tr><tr><td> $256 \times 256$ </td><td>0.799</td><td>0.509</td><td>0.543</td><td>0.092</td><td>0.782</td><td>0.761</td><td>0.540</td><td>0.289</td><td>0.507</td><td>0.545</td><td>0.349</td><td>0.520</td></tr><tr><td rowspan="3">SwinUNETRV2 [14]</td><td> $1024 \times 1024$ </td><td>0.850</td><td>0.510</td><td>0.448</td><td>0.020</td><td>0.756</td><td>0.882</td><td>0.706</td><td>0.226</td><td>0.621</td><td>0.699</td><td>0.000</td><td>0.520</td></tr><tr><td> $512 \times 512$ </td><td>0.810</td><td>0.342</td><td>0.430</td><td>0.005</td><td>0.753</td><td>0.829</td><td>0.641</td><td>0.147</td><td>0.472</td><td>0.557</td><td>0.000</td><td>0.453</td></tr><tr><td> $256 \times 256$ </td><td>0.707</td><td>0.283</td><td>0.310</td><td>0.000</td><td>0.691</td><td>0.700</td><td>0.454</td><td>0.000</td><td>0.307</td><td>0.216</td><td>0.000</td><td>0.333</td></tr><tr><td>MUViT [1] +Mask2Former</td><td> $256 \times 256$ </td><td>0.735</td><td>0.393</td><td>0.435</td><td>0.000</td><td>0.766</td><td>0.778</td><td>0.543</td><td>0.227</td><td>0.012</td><td>0.249</td><td>0.157</td><td>0.391</td></tr><tr><td>MUViT [1] +UNETR</td><td> $256 \times 256$ </td><td>0.779</td><td>0.410</td><td>0.311</td><td>0.000</td><td>0.762</td><td>0.775</td><td>0.570</td><td>0.068</td><td>0.132</td><td>0.466</td><td>0.000</td><td>0.388</td></tr><tr><td>MUViT [1,8,32] +Mask2Former (naive)</td><td> $3 \times 256 \times 256$ </td><td>0.924</td><td>0.851</td><td>0.791</td><td>0.764</td><td>0.852</td><td>0.930</td><td>0.849</td><td>0.644</td><td>0.756</td><td>0.831</td><td>0.832</td><td>0.820</td></tr><tr><td>MUViT [1,8,32] +Mask2Former</td><td> $3 \times 256 \times 256$ </td><td>0.956</td><td>0.934</td><td>0.896</td><td>0.859</td><td>0.922</td><td>0.966</td><td>0.932</td><td>0.873</td><td>0.819</td><td>0.859</td><td>0.891</td><td>0.901</td></tr><tr><td>MUViT [1,8,32] +UNETR (naive)</td><td> $3 \times 256 \times 256$ </td><td>0.919</td><td>0.837</td><td>0.773</td><td>0.718</td><td>0.860</td><td>0.918</td><td>0.833</td><td>0.556</td><td>0.734</td><td>0.808</td><td>0.767</td><td>0.793</td></tr><tr><td>MUViT [1,8,32] +UNETR</td><td> $3 \times 256 \times 256$ </td><td>0.954</td><td>0.928</td><td>0.889</td><td>0.796</td><td>0.924</td><td>0.958</td><td>0.922</td><td>0.802</td><td>0.809</td><td>0.842</td><td>0.887</td><td>0.883</td></tr></table>

7

<!-- page 8 of 16 -->

![Image block](images/page008-block01.png)

Figure 5. Semantic segmentation results on KPIS. Depicted in different columns are the input image, ground truth, U-Net, MUVIT $_ \mathrm { [ 1 ] ^ { + } U N E T R }$ , and $\mathrm { { \bf M U V I T } _ { [ 1 , 8 ] } { + } \bf U N E T R }$ predictions. Top: full image $( 5 2 k   \times   5 6 k \; \mathrm { p x } ) ;$ ; bottom: magnified regions (white boxes) showing detail comparison.

## 4.6. Semantic segmentation on KPIS

We finally evaluate MUVIT on the kidney pathology segmentation task, comparing against both classical architectures and recent transformer-based methods as reported in [32]. Similarly to the previous dataset, we obtain pre-dictions for the whole WSIs through tiled inference without any overlap between tiles. Table 6 shows the Dice score on the test set. $\mathrm { { M U V I T _ { [ 1 , 8 ] } } { + } U N E T R }$ achieves a Dice of 0.8958, substantially outperforming HoloHisto-4K (0.8454), a method tailored for KPIS, despite using much smaller inputs $( 2 \times 5 1 2 \times 5 1 2$ vs. $3 8 4 0   \times   2 1 6 0 )$ . Notably, $\mathrm { { M U V I T _ { [ 1 , 8 ] } { + } U N E T R } }$ also improves markedly over its single-resolution variant $\mathrm { { M U V I T _ { [ 1 ] } { + } U N E T R } }$ (Dice 0.8958

Table 6. Binary semantic segmentation results on KPIS (test set). \* indicates the score was taken from [32]. All baselines operate at the highest resolution (corresponding to l = 1).

| Method | Input size | DSC |
| --- | --- | --- |
| nnU-Net<sup>∗</sup> [16] | 1024×1024 | 0.6219 |
| UNETR<sup>∗</sup> [12] | 1024×1024 | 0.6425 |
| SegFormer<sup>∗</sup> [35] | 1024×1024 | 0.6501 |
| DeepLabV3∗ [4] | 1024×1024 | 0.6822 |
| SwinUNETR-V2∗ [14] | 1024×1024 | 0.7158 |
| SAM-ViT-B<sup>∗</sup> [18] | 1024×1024 | 0.7519 |
| SAM-ViT-H<sup>∗</sup> [18] | 1024×1024 | 0.7724 |
| HoloHisto-4K<sup>∗</sup> [32] | 3840×2160 | 0.8454 |
| U-Net [26] | 512×512 | 0.5499 |
| MUVIT<sub>[1]</sub>+Mask2Former | 512×512 | 0.8390 |
| MUVIT<sub>[1]</sub>+UNETR | 512×512 | 0.8492 |
| MUVIT<sub>[1,8]</sub>+Mask2Former | 2×512×512 | 0.8866 |
| MUVIT<sub>[1,8]</sub>+UNETR | 2×512×512 | 0.8958 |

Table 7. Linear probing results on KPIS test set (4 classes) for HIPT [5] and MUVIT encoders operating at different resolution sets. Reported is one-vs.-all (OvA) averaged ROC-AUC. Note that HIPT was used off-the-shelf and thus was not pre-trained on the KPIS dataset.

| Method | FOV size | ROC-AUC |
| --- | --- | --- |
| HIPT [5] (off-the-shelf) | 4096×4096 | 0.907 |
| MUVIT<sub>[1]</sub> | 256×256 | 0.958 |
| MUVIT<sub>[1,8]</sub> | 2×256×256 | 0.963 |
| MUVIT<sub>[1,8,32,64]</sub> | 4×256×256 | 0.988 |

vs. 0.8492), demonstrating the benefits of incorporating contextual information from coarser scales. Qualitative results in Figure 5 show that MUVIT produces sharper boundaries and captures fine tissue structures more reliably than U-Net and single-level $\mathrm { { \bf M } U V I T _ { [ 1 ] } }$ baselines, particularly in the magnified regions.

To further analyze the learned representations, we perform a linear probing experiment using features from a frozen MAE-pretrained encoder on 4 resolution levels $( 1 , 8 , 3 2 , 6 4 )$ . We train a logistic regression model on top of the extracted features in order to classify the experimental group of each WSI (4 subgroups). For this, we create an offline training set by sampling five random nested multi-resolution crops (see Section 3) per training image and a corresponding test set using 10 crops per test image. Table 7 shows the one-vs-all (OVA) ROC-AUC scores for the 4- class classification problem demonstrating the effectiveness of MUVIT for capturing information across scales: adding more resolution levels progressively improves performance from 0.958 (single-level) to 0.988 (four levels [1, 8, 32, 64]). This indicates that the encoder learns increasingly rich repre-

8

<!-- page 9 of 16 -->

sentations as more scales are incorporated. We also include a comparison on embeddings obtained using HIPT [5] off-the-shelf (ROC-AUC 0.907 with input size 4096×4096). Note that HIPT was pre-trained on a large pool of histopathology datasets which does not include KPIS.

## 4.7. Coordinate sensitivity and scaling behaviour

Despite world coordinate alignment across multiple scales being essential for performance as highlighted by the naive experiments (Tabs. 4 to 6), we wanted to test whether MUVIT models exhibit robustness to coordinate noise during inference. Note that we include an augmentation which adds scaling and shift to the samples bounding box coordinates during pre-training and fine-tuning.

We thus performed such an analysis by measuring the segmentation performance of the best performing MUVIT models for each dataset while adding Gaussian noise with standard deviation σ to the inference bounding box coordinates. We observe that MUVIT remains robust to such noise, with only minor degradation up to ≈ 32px on all datasets (Figure S3).

We also benchmarked runtime and peak memory on arbitrarily-sized inputs using a fixed patch size $p = 8 ,$ as used throughout the paper, as well as an increasing number of resolution levels $l   \in   \{ 1 , 8 , 3 2 \}$ . Figure S4 expectedly showcases that MUVIT incurs computational overhead in exchange for a substantially larger effective context.

## 5. Conclusions

We presented MUVIT, a multi-resolution Vision Transformer architecture for large-scale microscopy image analysis. The architecture processes multiple resolution levels of the same spatial region through a unified encoder, using coordinate-based Rotary Position Embeddings to preserve spatial relationships across scales. This explicit geometric encoding enables cross-scale attention between true multi-resolution observations, which differs from hierarchical architectures that rely on learned feature pyramids. Our experiments demonstrate that multi-resolution processing with accurate spatial information provides substantial benefits across synthetic and real microscopy datasets. The naive bbox baseline confirms that accurate spatial correspondence is crucial for performance, despite MUVIT exhibiting robustness when injecting noise into the world coordinates. Linear probing experiments show progressive improvement as more resolution levels are added, indicating that the encoder learns increasingly rich representations with explicit multi-scale structure. While jointly attending over all scales increases computational cost, this overhead could be reduced in future work through sparse or cross-scale attention schemes.

We have focused on semantic segmentation as a downstream task as it most directly probes the core challenge addressed by MUVIT: combining global (e.g. anatomical)

context with local, high resolution detail in gigapixel images. Assessing the impact on other common tasks for microscopy image processing, such as instance segmentation or object detection, is a natural direction for future work.

The framework is flexible and can incorporate non-nested views from different spatial regions of an image, and can be readily extended to 3D volumes where crops can be sampled from different slices while maintaining spatial consistency through coordinate-based positional encoding.

## Author contributions

A.D.M. developed the idea, implemented the software, conducted experiments and interpreted the results. G.L.M. contributed to the conceptual development, helped shape the experimental design and interpreted results. M.W. conceived the project, developed the idea, designed the architecture, implemented the software, conducted experiments and interpreted the results. G.L.M. and M.W. jointly supervised the work. All authors discussed the results and wrote the manuscript.

## Acknowledgements

We thank members of the Weigert and La Manno labs for their feedback and discussions of the project. G.L.M. received support from the Swiss National Science Foundation grant PZ00P3 193445 and from the Horizon Europe ERC Starting grant 101221782. A.D.M. and G.L.M. were supported by the Carl Zeiss AG-EPFL joint initiative Research-IDEAS. M.W. was supported by the BMFTR and by SMWK (project identification number: ScaDS.AI).

## References

[1] Anwai Archit, Luca Freckmann, Sushmita Nair, Nabeel Khalid, Paul Hilt, Vikas Rajashekar, Marei Freitag, Carolin Teuber, Melanie Spitzner, Constanza Tapia Contreras, et al. Segment anything for microscopy. Nature Methods, 22(3): 579–591, 2025. 1

[2] Roman Bachmann, David Mizrahi, Andrei Atanov, and Amir Zamir. MultiMAE: Multi-modal Multi-task Masked Autoencoders, 2022. arXiv:2204.01678 [cs]. 2, 4

[3] Chun-Fu Richard Chen, Quanfu Fan, and Rameswar Panda. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 347–356, Montreal, QC, Canada, 2021. IEEE. 2

[4] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801–818, 2018. 7, 8

[5] Richard J. Chen, Chengkuan Chen, Yicong Li, Tiffany Y. Chen, Andrew D. Trister, Rahul G. Krishnan, and Faisal Mahmood. Scaling Vision Transformers to Gigapixel Images via

9

<!-- page 10 of 16 -->

Hierarchical Self-Supervised Learning. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition $( C V P R ) ,$ pages 16123–16134, New Orleans, LA, USA, 2022. IEEE. 1, 2, 8, 9

[6] Wuyang Chen, Ziyu Jiang, Zhangyang Wang, Kexin Cui, and Xiaoning Qian. Collaborative global-local networks for memory-efficient segmentation of ultra-high resolution images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8924–8933, 2019. 2

[7] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 4

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 1, 3

[9] Qihang Fan, Quanzeng You, Xiaotian Han, Yongfei Liu, Yunzhe Tao, Huaibo Huang, Ran He, and Hongxia Yang. Vitar: Vision transformer with any resolution. arXiv preprint arXiv:2403.18361, 2024. 1

[10] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA-02: A Visual Representation for Neon Genesis. Image and Vision Computing, 149:105171, 2024. arXiv:2303.11331 [cs]. 3, 4

[11] Adam Glaser, Jayaram Chandrashekar, Sonya Vasquez, Cameron Arshadi, Rajvi Javeri, Naveen Ouellette, Xiaoyun Jiang, Judith Baka, Gabor Kovacs, Micah Woodard, et al. Expansion-assisted selective plane illumination microscopy for nanoscale imaging of centimeter-scale tissues. Elife, 12: RP91979, 2025. 1

[12] Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R Roth, and Daguang Xu. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 574–584, 2022. 4, 8

[13] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked Autoencoders Are Scalable Vision Learners. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15979–15988, New Orleans, LA, USA, 2022. IEEE. 2

[14] Yufan He, Vishwesh Nath, Dong Yang, Yucheng Tang, Andriy Myronenko, and Daguang Xu. Swinunetr-v2: Stronger swin transformers with stagewise convolutions for 3d medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 416–426. Springer, 2023. 7, 8

[15] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary Position Embedding for Vision Transformer, 2024. arXiv:2403.13298 [cs]. 3

[16] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a self-configuring method

for deep learning-based biomedical image segmentation. Nature methods, 18(2):203–211, 2021. 8

[17] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale Image Quality Transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5128–5137, Montreal, QC, Canada, 2021. IEEE. 2

[18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3, 8

[19] Oren Kraus, Kian Kenyon-Dean, Saber Saberian, Maryam Fallah, Peter McLean, Jess Leung, Vasudev Sharma, Ayla Khan, Jia Balakrishnan, Safiye Celik, Dominique Beaini, Maciej Sypetkowski, Chi Vicky Cheng, Kristen Morse, Maureen Makes, Ben Mabey, and Berton Earnshaw. Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology. pages 11757–11768, 2024. 3

[20] Youngwan Lee, Jonghee Kim, Jeffrey Willette, and Sung Ju Hwang. MPViT: Multi-Path Vision Transformer for Dense Prediction. 2

[21] Yuchen Liu, Natasha Ong, Kaiyan Peng, Bo Xiong, Qifan Wang, Rui Hou, Madian Khabsa, Kaiyue Yang, David Liu, Donald S. Williamson, and Hanchao Yu. MMViT: Multiscale Multiview Vision Transformers, 2023. arXiv:2305.00104 [cs]. 1, 2

[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 2

[23] Andreas Muller, Deborah Schmidt, Jan Philipp Albrecht, Lu- ¨ cas Rieckert, Maximilian Otto, Leticia Elizabeth Galicia Garcia, Gunar Fabig, Michele Solimena, and Martin Weigert. Modular segmentation, spatial analysis and visualization of volume electron microscopy datasets. Nature Protocols, 19 (5):1436–1466, 2024. 1

[24] Sophie Ostmeier, Brian Axelrod, Maya Varma, Michael E. Moseley, Akshay Chaudhari, and Curtis Langlotz. LieRE: Lie Rotational Positional Encodings, 2025. arXiv:2406.10322 [cs] version: 5. 3

[25] Colorado J. Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, and Trevor Darrell. Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning. pages 4088–4099, 2023. 3

[26] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 1, 7, 8

[27] Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell detection with star-convex polygons. In International conference on medical image computing and computerassisted intervention, pages 265–273. Springer, 2018. 1

[28] Frederik Filip Stæger, Kristian Nygaard Mortensen, Malthe Skytte Nordentoft Nielsen, Bjorn Sigurdsson, Louis Krog ¨

10

<!-- page 11 of 16 -->

Kaufmann, Hajime Hirase, and Maiken Nedergaard. A threedimensional, population-based average of the c57bl/6 mouse brain from dapi-stained coronal slices. Scientific Data, 7(1): 235, 2020. 5

[29] Carsen Stringer, Tim Wang, Michalis Michaelos, and Marius Pachitariu. Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1):100–106, 2021. 1

[30] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 2, 3, 4

[31] Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, and Hairong Qi. Cross-Scale MAE: A Tale of Multi-Scale Exploitation in Remote Sensing. 3

[32] Yucheng Tang, Yufan He, Vishwesh Nath, Pengfeig Guo, Ruining Deng, Tianyuan Yao, Quan Liu, Can Cui, Mengmeng Yin, Ziyue Xu, Holger Roth, Daguang Xu, Haichun Yang, and Yuankai Huo. Holohisto: End-to-end gigapixel wsi segmentation with 4k resolution sequential tokenization, 2024. 2, 5, 8

[33] Rui Tian, Zuxuan Wu, Qi Dai, Han Hu, Yu Qiao, and Yu-Gang Jiang. ResFormer: Scaling ViTs with Multi-Resolution Training. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22721–22731, Vancouver, BC, Canada, 2023. IEEE. 2

[34] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 568–578, 2021. 1, 2

[35] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34:12077–12090, 2021. 8

[36] C Shan Xu, Song Pang, Gleb Shtengel, Andreas Muller, ¨ Alex T Ritter, Huxley K Hoffman, Shin-ya Takemura, Zhiyuan Lu, H Amalia Pasolli, Nirmala Iyer, et al. An open-access volume electron microscopy atlas of whole cells and tissues. Nature, 599(7883):147–151, 2021. 1

[37] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier Gonzalez, Yu Gu, et al. A whole-slide foundation ´ model for digital pathology from real-world data. Nature, 630 (8015):181–188, 2024. 1

[38] Bin Yang, Merlin Lange, Alfred Millett-Sikking, Xiang Zhao, Jordao Bragantini, Shruthi VijayKumar, Mason Kamb, Rafael ˜ Gomez-Sj ´ oberg, Ahmet Can Solak, Wanpeng Wang, et al. ¨ Daxi—high-resolution, large imaging volume and multi-view single-objective light-sheet microscopy. Nature methods, 19 (4):461–469, 2022. 1

[39] Zizhen Yao, Cindy TJ Van Velthoven, Michael Kunst, Meng Zhang, Delissa McMillen, Changkyu Lee, Won Jung, Jeff Goldy, Aliya Abdelhak, Matthew Aitken, et al. A highresolution transcriptomic and spatial atlas of cell types in the whole mouse brain. Nature, 624(7991):317–332, 2023. 1

[40] Hao Yu, Tangyu Jiang, Shuning Jia, Shannan Yan, Shunning Liu, Haolong Qian, Guanghao Li, Shuting Dong, and Chun Yuan. ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices. 3

11

<!-- page 12 of 16 -->

# MUVIT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy

Supplementary Material

## A. Training details and hyperparameters

This section provides comprehensive implementation details for reproducibility. Note that the full codebase (including experiments code) and datasets will be released upon publication.

## A.1. Hardware specifications

All models were trained on the Capella HPC cluster at TU Dresden. Its nodes consist of 2x AMD EPYC 7282 CPUs (64 cores). Runs booked up to 256 GB RAM (DDR-4800). For MAE pre-training, models were trained on a single node in a multi-GPU setting (4x Nvidia H100, 96GB). All semantic segmentation models (including the baselines) were trained on a single node using a single GPU (1x Nvidia H100, 96 GB)

## A.2. MAE pre-training

**Architecture.** The encoder consists of 12 transformer layers with embedding dimension $D = 5 1 2$ and patch size $P = 8 .$ Each resolution level has independent patch embedding layers followed by learnable level embeddings. The decoder uses L lightweight networks (one per level), each with 2 transformer layers and dimension 256. The first decoder layer incorporates cross-attention to the full set of visible encoder outputs with coordinate-based RoPE. All transformer layers (including those in the decoder) use the coordinatebased RoPE embeddings described in the main text.

**Masking strategy.** We use a masking ratio of $\rho = 0 . 7 5$ with Dirichlet-weighted sampling across resolution levels. The proportions of masked tokens per level $( \lambda _ { 1 } , \ldots , \lambda _ { L } )$ are sampled from $\operatorname { D i r } ( \alpha )$ with $\alpha   =   0 . 5 .$ , encouraging the model to learn cross-scale relationships by varying the relative masking ratio per level.

**Data sampling.** Due to the large-scale nature of the data, we employ a sampling-based strategy in which a nested multi-resolution crop is randomly sampled from each image (see Section 4). We therefore use the term training sample to denote such a multi-resolution crop extracted from a training image. We define an epoch as a pass of 10000 training samples to the network during training. Sampled crops are normalized with percentile-based normalization $( p _ { m } , p _ { M } =$ 1, 99.8) for MOUSE and by dividing by 255 for KPIS.

**Multi-resolution configuration.** For KPIS, we use resolution levels [1, 8, 32, 64] during MAE pre-training. For MOUSE, we use levels [1, 8, 32] for both MAE pre-training. All levels use the same pixel-space patch size $( 2 5 6 \times 2 5 6$

pixels), with lower resolution levels capturing progressively larger spatial context.

**Optimization.** MAE pre-training uses a batch size of 48/64 for 2000/1000 epochs for MOUSE/KPIS epochs with learning rate $6 \times 1 0 ^ { - 5 }$ . We use the AdamW optimizer with gradient clipping at 0.5 and deterministic cosine annealing learning rate schedule. Early stopping is not used.

**Loss function.** The loss function for MAE pre-training is the mean squared error (MSE) (computed on masked patches only) averaged across all resolution levels.

**Data augmentation.** Used augmentations include random horizontal and vertical flips as well as intensity jittering and bounding box perturbations (random shifts and scaling). Note that the horizontal and vertical flips affect the whole nested crop (i.e. they are not applied independently to each resolution level) and that bounding boxes are augmented accordingly.

## A.3. Segmentation fine-tuning

**Architecture.** We use the same pre-trained encoder archi tecture (12 transformer layers, token embedding dimension $D = 5 1 2 )$ and couple it with either a UNETR-style decoder or Mask2Former decoder as described in Section 3.

**Data sampling.** We use the same sampling strategy as for MAE pre-training. Note that the target semantic segmentation mask is only sampled at the highest resolution level. For KPIS we ensure that at least half the training samples contain at least a positive class pixel. We define an epoch as a pass of 4096 training samples to the network during training for the semantic segmentation task. Sampled crops are normalized with percentile-based normalization $( p _ { m } , p _ { M } = 1 , 9 9 . 8 )$ for MOUSE and by dividing by 255 for KPIS.

**Multi-resolution configuration.** For KPIS, we extract only levels [1, 8] from the MAE pretrained encoder for segmentation fine-tuning (as shown in Tab. 6). For MOUSE, we use the full encoder. All levels use the same pixel-space patch size (256×256 pixels for MOUSE, 512×512 for KPIS differing from the pre-training patch size).

**Optimization.** Semantic segmentation training uses a batch size of 16 for MOUSE (except MUVIT+UNETR variants which used a batch size of 8 due to memory limitations) and a batch size of 8 for KPIS. All our models were trained for 100 epochs. We employ asymmetric learning rates: decoder learning rate $1 \times 1 0 ^ { - 4 }$ , encoder learning rate $1 \times 1 0 ^ { - 5 }$ (0.1× decoder rate). We use the AdamW optimizer with gradient clipping at 0.5 and cosine annealing schedule.

1

<!-- page 13 of 16 -->

**Loss function.** We use a combined cross-entropy and Dice loss: $\mathcal { L }   =   \lambda _ { \mathrm { C E } } \cdot \mathcal { L } _ { \mathrm { C E } } ( \tilde { y } , y )   +   \lambda _ { \mathrm { D i c e } } \cdot \mathcal { L } _ { \mathrm { D i c e } } ( \tilde { y } , y )$ with $\lambda _ { \mathrm { C E } } = 1 . 0$ and $\lambda _ { \mathrm { D i c e } } = 0 . 1$ . For KPIS, we double the weight to the foreground class to aid with the imbalance between foreground and background. The Dice loss excludes the background class (class 0) and is computed separately per resolution level before averaging.

**Data augmentation.** We use the same augmentation set as in the MAE pre-training case.

## B. MUVIT components

## B.1. Number of parameters

Table S1. Number of parameters for MUVIT components and variants.

| Module | Parameters |
| --- | --- |
| MUVIT<sub>[1]</sub> (encoder) | 25.24M |
| MUVIT<sub>[1,8,32]</sub> (encoder) | 25.31M |
| MUVIT<sub>[1]</sub>+MAE decoder | 26.44M |
| MUVIT<sub>[1,8,32]</sub>+MAE decoder | 28.88M |
| MUVIT<sub>[1]</sub>+Mask2Former decoder | 33.44M |
| MUVIT<sub>[1,8,32]</sub>+Mask2Former decoder | 36.80M |
| MUVIT<sub>[1]</sub>+UNETR decoder | 31.66M |
| MUVIT<sub>[1,8,32]</sub>+UNETR decoder | 35.02M |

## B.2. Semantic Segmentation Decoder

**UNETR decoder** The UNETR decoder uses skip connec tions from multiple intermediate encoder layers to preserve fine-grained spatial information during upsampling, as introduced in the original UNETR work. The decoder consists of a progressive upsampling pathway that combines transformer-encoded features with skip connections at multiple resolution levels. We extract intermediate features from arbitrarily selected encoder layers (by default, evenly spaced across the encoder depth) and project them to a D-dimensional vector $( D   =   2 5 6 )$ . The decoder operates through multiple upsampling stages, where each stage performs bilinear upsampling followed by residual convolutional blocks that fuse the upsampled features with corresponding skip connections from the intermediate encoder outputs. The last skip connection involves directly processing the original input image at the highest resolution level instead of the encoder outputs. Each upsampling stage consists of bilinear interpolation followed by two regular convolutional blocks (3×3 convolutional layers with batch normalization and a ReLU activation). Skip connections are incorporated through element-wise addition after using

$1 \times 1$ convolutions to match the channel axes. The final output layer produces segmentation predictions at full input resolution through a $1 \times 1$ convolution that maps from the embedding dimension $D$ to the target number of output channels/classes.

**Mask2Former decoder** The Mask2Former decoder follows the query-based segmentation paradigm introduced in the original work. The decoder consists of three main components: a pixel decoder that progressively upsamples encoder features to full resolution, a set of learnable mask queries that represent potential objects, and a transformer decoder that refines these queries. For the pixel decoder we use a series of bilinear upsampling layers followed by regular convolutional blocks (as described above) in order to create dense mask features at the original image resolution. We initialize $| M _ { q } |$ learnable embeddings of dimension $D = 2 5 6$ that serve as mask queries, where the number of queries is set to be at least equal to the number of output classes. The transformer decoder refines the mask queries using multiple transformer layers. Each layer applies self-attention among the queries to model relationships between different potential masks, followed by cross-attention between queries and the encoder features. We use RoPE for the spatial features obtained from the encoder to maintain spatial awareness during cross-attention, while queries remain position-agnostic so that they can be used as learnable representations. We then feed the refined queries to a layer normalization module followed by an MLP to produce the mask embeddings. The final segmentation masks are generated by computing the dot product between these mask embeddings and the pixel decoder’s mask features, followed by a refinement convolution that maps from $| M _ { q } |$ to the desired number of output channels.

2

<!-- page 14 of 16 -->

## C. Semantic segmentation results

C.1. MOUSE

![Image block](images/page014-block03.png)

Figure S1. Semantic segmentation results on four test images of MOUSE. The images correspond to different brain sections of the same specimen, exhibiting anatomical changes and, thus, showcasing different semantic classes predictions. Columns show (left to right): input image, ground truth, U-Net, DeepLabV3, and $\mathrm { { \bf M } U V I T _ { [ 1 , 8 , 3 2 ] } } .$ +UNETR semantic predictions. The input sizes of the baselines are 1024×1024 while the input for $\mathrm { { M U V I T _ { [ 1 , 8 , 3 2 ] } \mathrm { { + U N E T R } } } }$ is $3 \times 2 5 \dot { 6 } \times \dot { 2 5 6 }$ . Magnified regions (white boxes) show detailed comparison.

3

<!-- page 15 of 16 -->

C.2. KPIS

![Image block](images/page015-block02.png)

Figure S2. Semantic segmentation results on three test whole slide images (WSIs) of KPIS. The images correspond to three different conditions. Columns show (left to right): input image, ground truth, U-Net, $\mathrm { { M U V I T _ { [ 1 , 8 ] } \mathrm { { + } U N E T R } } } ,$ , and $\mathrm { { \bf M } U V I T _ { [ 1 , 8 ] } } \mathrm { { \bf + } }$ UNETR semantic predictions. The spatial input size of all models is 512×512. Magnified regions (white boxes) show detailed comparison.

4

<!-- page 16 of 16 -->

## C.3. Coordinate sensitivity and scaling behaviour

![Chart block](images/page016-block02.png)

Figure S3. Effect of adding Gaussian noise of varying standard deviation σ on MUVIT segmentation performance across different datasets.

![Chart block](images/page016-block04.png)

![Chart block](images/page016-block05.png)

Figure S4. Runtime and memory scaling of MUVIT. Hue denotes input spatial dimensions.

5