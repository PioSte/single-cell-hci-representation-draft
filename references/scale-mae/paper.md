<!-- page 1 of 16 -->

arXiv:2212.14532v4 [cs.CV] 22 Sep 2023

# Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning

Colorado J Reed<sup>1,2\*</sup>, Ritwik Gupta<sup>1\*</sup>, Shufan Li<sup>1\*</sup>, Sarah Brockman<sup>3</sup>, Christopher Funk<sup>3</sup>, Brian Clipp<sup>3</sup>, Kurt Keutzer<sup>1</sup>, Salvatore Candido<sup>2</sup>, Matt Uyttendaele<sup>2</sup>, Trevor Darrell<sup>1</sup>

<sup>1</sup>Berkeley AI Research; <sup>2</sup>Meta AI, FAIR; <sup>3</sup>Kitware Inc. correspondence to ritwikgupta@berkeley.edu

## Abstract

Large, pretrained models are commonly finetuned with imagery that is heavily augmented to mimic different conditions and scales, with the resulting models used for various tasks with imagery from a range of spatial scales. Such models overlook scale-specific information in the data for scale-dependent domains, such as remote sensing. In this paper, we present Scale-MAE, a pretraining method that explicitly learns relationships between data at different, known scales throughout the pretraining process. Scale-MAE pre-trains a network by masking an input image at a known input scale, where the area of the Earth covered by the image determines the scale of the ViT positional encoding, not the image resolution. Scale-MAE encodes the masked image with a standard ViT backbone, and then decodes the masked image through a bandpass filter to reconstruct low/high frequency images at lower/higher scales. We find that tasking the network with reconstructing both low/high frequency images leads to robust multiscale representations for remote sensing imagery. Scale-MAE achieves an average of a 2.4 − 5.6% non-parametric kNN classification improvement across eight remote sensing datasets compared to current state-of-the-art and obtains a 0.9 mIoU to 1.7 mIoU improvement on the SpaceNet building segmentation transfer task for a range of evaluation scales.

## 1. Introduction

Remote sensing data is captured from satellites and planes through a mixture of sensors, processing pipelines, and viewing geometries. Depending on the composition and relative geometry of the sensor to the Earth, each image’s Ground Sample Distance (GSD - the physical distance between two

Input Image

Ground Truth

Scale-MAE

Vanilla MAE

![Image block](images/page001-block13.png)

Figure 1. Scale-MAE learns better representations for multiscale tasks compared to vanilla MAE. (Column 1) The top image spans an area at 0.3m GSD and the bottom image shows the same region at a coarser GSD. (Columns 2-4) The following columns show a ground truth building segmentation, Scale-MAE segmentation from a finetuned UperNet, and segmentation from an analogously finetuned UperNet from a vanilla MAE, respectively. Scale-MAE demonstrates better performance across images at both scales. See the supplementary material for more examples.

adjacent pixels in an image) can vary from 0.3m to 1km, so a 100x100 pixel image could span anywhere from an Olympicsize swimming pool (900 m<sup>2</sup>) to almost the entire country of Jamaica (10,000 km<sup>2</sup>). The data within each image, and the corresponding objects and points of interest, can therefore vary across wide spatial ranges. Data from these multiscale sensors provide critical and complementary information for various operational and research applications in areas such as atmospheric, hydrologic, agricultural, and environmental monitoring [45, 52].

Few modern computer vision methods have explicitly addressed multiscale remote sensing imagery [35]. Nevertheless, the remote sensing vision community has increasingly used large, pretrained models [13, 20], where such applications finetune a pretrained model for a single source of

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">\*Denotes co-first authorship. Co-first authors will prioritize their names on their resumes/websites.</span></small>

<!-- page 2 of 16 -->

![Image block](images/page002-block01.png)

Figure 2. Scale-MAE employs the Masked Autoencoder framework. An input image is patchified and masked before being passed into an MAE encoder. A Ground Sample Distance Positional Encoding (GSDPE) is added to the encoder input, which scales the positional encodings to the area of ground covered. The Scale-MAE decoders has three stages: (1) Decoding, which uses a smaller number of transformer layers than MAE to decode the encoded values (2) Upsampling, which progressively deconvolves the decoded feature map to a larger size before being passed through the Laplacian Blocks (abbreviated LB, see Section 3), (3) Reconstruction, which then reconstructs low and high frequency features at different scales. These outputs are used to compute an aggregate loss with ground truth low and high frequency features, where following super resolution literature [2], an L1 loss is used for high frequency output to better reconstruct edges and an L2 loss is used for low frequency output to better reconstruct average values.

data at a specific scale [13, 20, 22, 32, 41]. In this paper we present Scale-MAE, a masked reconstruction model that explicitly learns relationships between data at different, known scales throughout the pretraining process. By leveraging this information, Scale-MAE produces a pretrained model that performs better across a wide range of GSDs and tasks.

Masked Autoencoders [26] offer self-supervised learning without explicit augmentations. A standard Masked Autoencoder resizes/crops an image, masks the majority of the transformed image, and then tasks a Vision Transformer (ViT) based autoencoder with embedding the unmasked com ponents. A decoding ViT then decodes the full image from these learned embeddings, where the decoder is later dis carded and the encoder is used to produce representations for an unmasked input image.

Existing MAE-based pretraining approaches fail to generalize across domains with images at multiple scales. Scale-MAE (Figure 1) overcomes this through a GSD-based positional encoding derived from the land area covered in the image. This informs the ViT of both the position and scale of the input image. Scale-MAE also uses a Laplacian-pyramid decoder to encourage the network to learn multiscale rep resentations. The embeddings are decoded to two images containing low and residual high frequency information, respectively – see Figure 2. As we discuss in Section 3, this structure allows the ViT decoder to use fewer parameters than MAE while still producing strong representations across multiple scales.

We show that Scale-MAE leads to better performing, more robust multiscale representations than both a standard MAE and a recently proposed, state-of-the-art MAEs SatMAE [13] and ConvMAE [21] across remote sensing datasets with a variety of scale and resolution characteristics. To the best of our knowledge Scale-MAE is the first self-

supervised MAE to include scale-aware positional encoding and Laplacian pyramids. In our experiments, Scale-MAE achieves an average of a 5.6% nonparametric kNN classification improvement across eight remote sensing datasets compared to current state-of-the-art in addition to a 0.9 mIoU to 1.7 mIoU improvement on the SpaceNet building segmentation transfer task for a range of evaluation scales (see Figure 1).

## 2. Related Work

**Representation learning and the Masked Autoencoder.** Representation learning aims to extract meaningful, intrinsic features from data for downstream use [5]. In practice, this often entails pretraining a deep network so that a lightweight learning routine can then finetune it for a particular downstream task, see [15, 16, 17, 24, 27, 30, 37, 49, 66]. The Masked Autoencoder (MAE) is a recent state-of-the-art self-supervised representation learning method in computer vision that pretrains a ViT encoder by masking an image, feeding the unmasked portion into a transformer-based encoder, and then tasking the decoder with reconstructing the input image [26]. MAEs fail to leverage scale information in scale-dependent domains as they are often reliant on absolute or relative positional encodings. To the best of our knowledge, Scale-MAE is the first MAE-based self-supervised learning method to incorporate a scale-variant positional encoding.

**Remote Sensing Representation Learning** Neumann et al. [46] were one of the first to exhaustively share results on existing representation learning and semi-supervised learning techniques for remote sensing imagery. Gao et al. [22] demonstrated the effectiveness of MAE pretraining for re-

<!-- page 3 of 16 -->

mote sensing image classification. Ayush et al. [3] leveraged the metadata from remote sensing images via spatially aligned but temporally separated images as positive pairs for contrastive learning and predicted the latitude and longitude as pretext tasks. Gupta et al. [25] demonstrated the use of MAEs as a pretraining approach for passive and active remote sensing imagery. Their method introduced flexible “adapters” which could be used interchangeably with an encoder for a set of input imagery modes. Cong et al. [13] introduced the SatMAE, which used temporal and spectral metadata in a positional encoding to encode spatio-temporal relationships in data. The temporal data contains the year, month, and hour enabling understanding of long-term change with the year, weather information from the month, and hour information for the time of day. Further Liu et al. [41] and Ibanez ˜ et al. [32] have shown that MAE architectures can be used for band selection in hyperspectral remote sensing images, significantly reducing data redundancy while maintaining high classification accuracy. Scale-MAE leverages inherent absolute scale information information present in scale-dependent domains as a way to learn robust, multiscale features that reduce data usage downstream.

**Super-resolution** Super-resolution has proven effective in improving accuracy within remote sensing images due to the extremely small size of objects within the image [51]. Previous works have aimed to learn continuous implicit representations for images at arbitrary resolutions to aid the super-resolution task. These representations are used to upsample the images either to specific scales [38] or to arbitrary resolutions [10, 31, 61]. Most super-resolution work aims to increase the resolution of the input image, whereas Scale-MAE produces both higher and lower resolution im ages. There is some work on super-resolution for satellite imagery, but much of this work is focused on synthetically creating high-resolution datasets for use with models trained specifically for high-resolution data [28, 35]. Scale-MAE, however, utilizes super-resolution as a means to obtain multiscale representations during pretraining.

**Multiscale Features** Because images can contain objects of many different pixel resolutions, the vision community has proposed many methods to extract multiscale features. These include spatial pyramids [6, 34, 36, 50] and dense sampling of windows [33, 62, 63]. These approaches have been combined by methods such as [19], in which dense histogramof-gradient features are computed for each feature pyramid level. Rather than using classical computer vision techniques to extract multiscale features, convolutional neural networks have been used to build deep multiscale features. CNNs with subsampling layers inherently build feature pyramids, a property exploited explicitly by models such as the Feature Pyramid Network and the Single-Shot Detector, amongst

others [23, 39, 40]. Recently, this multiscale idea has been extended to vision transformers by [18], who show that this architecture improves various video recognition and image classification tasks, as well as in [21, 67] which proposes various hybrid CNN-MAE architectures that yield multi-scale features during MAE pretraining. Different from these works, Scale-MAE uses a Laplacian pyramid decoder as a way to force an encoder to learn multiscale features with the ViT architecture.

## 3. Scale-MAE

This section describes the Scale-MAE pretraining frame work as illustrated in Figure 2. Scale-MAE is a self-supervised pretraining framework based on the Masked Autoencoder (MAE) [26]. Scale-MAE makes two contributions to the MAE framework. Standard MAE-based methods use absolute or relative positional encodings to inform the ViT of the position of the unmasked components, where an image at resolution r will have the same positional encodings regardless of the image content. Scale-MAE introduces the Ground Sample Distance (GSD) based positional encoding that scales in proportion to the area of land in an image, regardless of the resolution of the image. In addition, Scale-MAE introduces the Laplacian-pyramid decoder to the MAE framework to encourage the network to learn multiscale representations. Embeddings from a ViT encoder are decoded to a lower resolution image that captures the lower frequency information and a higher resolution image that captures the high-frequency information. We formalize Scale-MAE in the following subsections by first specifying the necessary MAE background, describing the GSD-based positional encoding, and then explaining the Laplacian-pyramid decoder.

**Setup** Let $I \; \in \; \mathbb { R } ^ { H \times W \times C }$ represent an input image of height H, width W, and C channels. The MAE patchifies I into a sequence S of independent patches of height and width P pixels, where each of the $N _ { p } { \mathrm { ~ p a t c h e s } } ,   s \in S$ has dimension $s   \in   \mathbb { R } ^ { P ^ { 2 } C }$ . A fraction, $m ,$ of the patches are then removed and the remaining patches are then passed through a projection function (e.g., a linear layer) to project the patches S into D dimensions, $f _ { E }   :   \mathbb { R } ^ { P ^ { 2 } C }   \to   \mathbb { R } ^ { D }$ , to obtain embedded patches $S _ { E } = f _ { E } ( S )$ . An $\mathbb { R } ^ { 2 }$ positional encoding vector, is then added to the embedded patches with

$$
v _ {x} (p o s, 2 i) = \sin \frac {p o s}{1 0 0 0 0 ^ {\frac {2 i}{D}}}\tag{1}
$$

$$
v _ {y} (p o s, 2 i + 1) = \cos \frac {p o s}{1 0 0 0 0 ^ {\frac {2 i}{D}}}\tag{2}
$$

where pos is the position of the patch along the given axis and i is the feature index (visualized in Figure 3), exactly as introduced in [54]. These positional encodings are then

<!-- page 4 of 16 -->

concatenated and added to the embedded patches, which are then fed into a ViT encoder. After the encoder, the removed m patches are then placed back into their original location in the sequence of patches where a learned mask token represents the masked patches that were not encoded. Another positional encoding vector is added to all patches and a sequence of transformer blocks decodes these patches to form the original input image, which is used as the learning target.

**Input** Scale-MAE performs a super resolution reconstruction, where the input image I is downsampled from a higher resolution image $I _ { \mathrm { h r } }$ at the ground truth GSD. Instead of targeting the input image, Scale-MAE targets high frequency and low frequency components of $I _ { \mathrm { h r } }$ , which is common in Laplacian pyramid super resolution models [64], where the high frequency component is at the same resolution as the ground truth image $I _ { \mathrm { h r } }$ and the low frequency component is at the same resolution as the input image $I ,$ as shown in Figure 2. Following many works in super resolution [64], the low frequency target image is obtained by interpolating $I _ { \mathrm { h r } }$ to a much lower resolution, $r _ { \mathrm { l o w } }$ and then interpolating to the same resolution as the input image I. The high frequency target image is obtained by downsampling $I _ { \mathrm { h r } }$ to another lower resolution $r _ { \mathrm { h i g h - l o w } }$ , and then upsampling to the same resolution as the ground truth image $I _ { h r }$ and subtracting this image $I _ { \mathrm { h f } } = I _ { \mathrm { h r } } - I _ { \mathrm { h i g h - l o w } }$ . The supplementary material provide more information on the upsampling/downsampling methodology. The key components for Scale-MAE are described next.

**GSD Positional Encoding** Images from scale-dependent domains have a metric which defines the absolute scale for the image. This metric has different names across domains and is referred to as the Ground Sample Distance (GSD) in remote sensing. The GSD is critical to understanding, conceptually, the kinds of features that will be available in an image. An image with finer GSD (lower number) will have higher frequency details than an image with coarser GSD (high number). Models are generally unaware of absolute scale when learning over a set of data. Specifically, even if they implicitly learn that all images in a dataset share a varying resolution from input-space augmentations, then these models do not explicitly condition on the GSDs encountered in unseen data.

We extend the positional encoding from Equation (2) to include GSD by scaling the positional encoding relative to the land area covered in an image as depicted in Figure 3 and mathematically:

$$
v _ {g s d, x} (p o s, 2 i) = \sin \frac {g}{G} \frac {p o s}{1 0 0 0 0 ^ {\frac {2 i}{D}}}\tag{3}
$$

$$
v _ {g s d, y} (p o s, 2 i + 1) = \cos \frac {g}{G} \frac {p o s}{1 0 0 0 0 ^ {\frac {2 i}{D}}}\tag{4}
$$

GSDPE

PE

![Image block](images/page004-block09.png)

Figure 3. Ground Sample Distance Positional Encoding (GS-DPE). (Left) Input images at the same pixel resolution but different GSDs are shown. The image on the bottom is a subset of the image on the top. (Center) This overlap in location, albeit at a different resolution, is reflected in the GSDPE. The finer image with smaller spatial extent is represented by a corresponding subsection of the overall sine wave on the bottom. (Right) A standard positional encoding is strictly dependent on the image resolution and uses the same embedding for both. The colors behind the sine waves show the intensity and quantization of the encoding.

where g is the GSD of the image and G is a reference GSD, nominally set to 1m. Intuitively, an object imaged at a finer resolution has more pixels representing it. When imaging the same object at a coarser resolution, those pixels must map to fewer pixels. In Equation (4), we interpolate the positional encoding by a factor of $\frac { G } { g }$ to account for the ordering of the coarser set of pixels. This simple idea underpins the GSD Positional Encoding, visualized in Figure 3.

**Scale-MAE decoder** The standard MAE learns representations by tasking a network with reconstructing an image after masking out most of its pixels. While the standard MAE decoder reconstructs the input image at the same scale as its input, the objective of Scale-MAE is to learn multi-scale representations. We draw on works from progressive super-resolution such as [56], that learn a high resolution, high frequency image and a lower resolution low frequency image, that when combined together, yield the input image at a higher resolution.

The Scale-MAE introduces a novel decoder which de-

<!-- page 5 of 16 -->

codes to multiple scales with a progressive Laplacian decoder architecture, replacing the traditional MAE “decoder”, which is really a Transfomer encoder. This architecture consists of three stages: decoding, upsampling, and recon struction, which are shown in Figure 2 and detailed below.

Decoding follows the standard MAE decoder where following the encoder, the removed m patches are then placed back into their original location in the sequence of patches where a learned mask token represents the masked patches that were not encoded, a positional encoding is added, and then a series of transformer layers decode all patches. In contrast to the standard MAE decoder, the Scale-MAE decoder uses fewer transformer layers (e.g. 3 layers instead of 8), which reduces the parameter complexity as quantified in Section 5. The output of these layers is then fed into the upsampling stage.

Upsampling The latent feature maps from the decoding stage are progressively upsampled to 2x and 4x resolution using deconvolution blocks, where the first deconvolution is 2x2 with stride 2 that outputs a feature map at 2x the input resolution (28 in Figure 2), followed by a LayerNorm and GELU, and then another 2x2 deconvolution layer that outputs a feature maps at 2x the previous resolution (56 in Figure 2). See the supplementary material for a full architectural diagram.

Reconstruction After having been upsampled, the lower resolution and higher resolution feature maps are passed into Laplacian Blocks (LBs in Figure 2) that reconstruct high and low resolution images for the high and low frequency reconstruction, respectively. Architecturally, the Laplacian Blocks consist of a sequence of three sub-blocks: a Laplacian Feature Mapping Block, a Laplacian Upsample Block, and a Laplacian Pyramid Reconstruction Block. The Feature Mapping Block is used to project features within a particular layer of the Laplacian Pyramid back to the RGB space. The Laplacian Upsample Block represents a learnable upsample function that maps latent features from one layer of the Laplacian Pyramid to a higher level. Finally, the Laplacian Pyramid Reconstruction Block is used to reconstruct information at the different frequencies in RGB space. Following super resolution literature [2], an L1 loss is used for high frequency output to better reconstruct edges and an L2 loss is used for low frequency output to better reconstruct average values. The supplementary material has architectural diagrams for each block.

## 4. Experiments

We investigate the quality of representations learned from Scale-MAE pretraining through a set of experiments that explore their robustness to scale as well as their transfer performance to additional tasks. First, we present our main experiments in Section 4.1 and compare with SatMAE [13], a current state-of-the-art MAE for remote sensing imagery,

Mask

![Image block](images/page005-block08.png)

Figure 4. Scale-MAE reconstruction. Examples from Functional Map of the World are shown. From left to right, an input image at 224x224 resolution is shown. Its corresponding mask is visualized as well. Columns 3 and 4 show the low and high frequency produced by the Scale-MAE decoder. The last column is the reconstruction obtained from summing the low and high frequency features together.

ConvMAE [21], a state-of-the-art multiscale MAE, as well as several other approaches detailed throughout. The exact implementation of Scale-MAE for the main experiments was determined through a set of ablation experiments presented in Section 4.2.

We pretrain a ViT-Large model with Scale-MAE using the Functional Map of the World (FMoW) [12] RGB training set, which consists of 363.6k images of varying image resolution and GSD, for 800 epochs. The initial higher resolution image $I _ { \mathrm { h r } }$ is taken as a random $4 4 8 \mathrm { { \tt { p } } x ^ { 2 } }$ crop of the input image, and the input image I is then a downsampled $2 2 4 \mathrm { { \tt { p x } } ^ { 2 } }$ from $I _ { \mathrm { h r } } .$ The low frequency groundtruth is obtained by downscaling $I _ { \mathrm { h r } }$ to $1 4 \mathrm { { p x } ^ { 2 } } ^ { \mathrm { { \tilde { } } } }$ and then upscaling to $2 2 4 \mathrm { { \tt { p x } } ^ { 2 } }$ , while the high frequency groundtruth is obtained by downscaling $I _ { \mathrm { h r } }$ to $5 6 \mathrm { { p x } ^ { 2 } }$ and then upscaling to $4 4 8 \mathrm { { p x } ^ { 2 } }$ and subtracting this image from $I _ { \mathrm { h r } } .$

Figure 4 shows examples of the masked input, low resolution/frequency, high resolution/frequency, and combined reconstruction of FMoW images during training. The low resolution/frequency images capture color gradients and landscapes, while the residual high resolution/frequency images capture object edges, roads, and building outlines.

## 4.1. Representation Quality

We evaluate the quality of representations from Scale-MAE by freezing the encoder and performing a non-parametric k-nearest-neighbor (kNN) classification with eight different remote sensing imagery classification datasets

<!-- page 6 of 16 -->

![Chart block](images/page006-block01.png)

Figure 5. Learning better representations at all scales. Scale-MAE (blue) features perform better than state-of-the-art. We evaluate kNN accuracy on eight datasets with a large variance in GSD. Scale-MAE consistently produces better results at coarser resolutions. In addition to using evaluation datasets at different GSDs, to further test the multiscale representations, we create multiple test sets for each dataset in which we downsampled the full resolution validation set to coarser GSDs at fixed percentages: $X_{val}^{G\%},G\in\{12.5,25,50,100\}$ , where EuroSat does not include the 12.5% because the images are at a resolution of 64px, our patch size is 16px, and an input image of 8px is too small.

with different GSDs, none of which were encountered during pretraining. The kNN classifier operates by encoding all train and validation instances, where each embedded in stance in the validation set computes the cosine distance with every other embedded instance in the training set. The instance is classified correctly if the majority of its k-nearestneighbors are in the same class as the validation instance, and incorrectly if they are in any other.

The reasoning behind the kNN classifier evaluation is that a strong pretrained network will output semantically grouped representation for unseen data of the same class. This evaluation for the quality of representations occurs in other notable works [7, 9, 57]. In addition to using evaluation datasets at different GSDs, to further test the multiscale representations, we create multiple test sets for each dataset. Since we cannot synthesize data at a finer GSD than the provided ground truth, we only downsample the full resolution validation set to coarser GSDs at fixed percentages: $X _ { v a l } ^ { G \% } , G \in \{ 1 2 . 5 , 2 5 , 5 0 , 1 0 0 \}$

Our analysis uses eight different land-use classification datasets: RESISC-45 [11], the UC Merced Land Use Dataset [65], AiRound and CV-BrCT [43], MLRSNet [48], EuroSAT [29], Optimal-31 [55], WHU-RS19 [14], SpaceNet v1 and v2 [53], and Functional Map of the World [12]. The datasets used span a wide range of GSDs, e.g., MLRSNet consists of data captured from aerial platforms with 0.1m GSD, while RESISC45 has imagery from medium-resolution satellites at >30m GSD. In some cases, the datasets present imagery at mixed GSDs which are not specified, in which case we assume an approximate constant GSD: see the supplementary material for all details. Furthermore, we provide an expanded set of experiments with linear probing and finetuning in the supplementary material.

Average Accuracy (%)

| Dataset | Scale-MAE | SatMAE | ConvMAE |
| --- | --- | --- | --- |
| AiRound | 63.2 | 57.8 | 59.7 |
| CV-BrCT | 69.7 | 66.2 | 68.4 |
| EuroSAT | 86.7 | 84.4 | 88.8 |
| MLRSNet | 81.7 | 75.0 | 79.5 |
| OPTIMAL-31 | 65.5 | 55.7 | 61.7 |
| RESISC | 70.0 | 61.0 | 67.0 |
| UC Merced | 75.0 | 69.8 | 70.0 |
| WHU-RS19 | 79.5 | 78.5 | 77.0 |

Table 1. Scale-MAE performs better, across all GSDs (as in Figure 5), for all datasets we experimented with compared to SatMAE. The average improvement across all datasets for Scale-MAE compared to SatMAE is 5.6% and 2.4% compared to ConvMAE with ViT-Large backbones.

We run kNN classification with k = 20. Figure 5 shows that Scale-MAE outperforms SatMAE and ConvMAE across GSD scales in the different evaluation datasets and across relative GSD scales within individual datasets. For example, the UC Merced has a GSD of 0.3m, but evaluating at scales [12.5%,100%] provides an artificial GSD range of [0.3m, 2.4m]. On this example, we see that Scale-MAE provides the largest performance gap at the 2.4m GSD, with similar performance at 0.3m.

Across all other evaluation datasets and wider range of GSDs, Scale-MAE outperforms SatMAE and ConvMAE, where Scale-MAE outperforms both methods by a larger gap as the GSD increasingly varies from the original GSD, indicating that Scale-MAE learns representations that are more robust to changes in scale for remote sensing imagery. We

<!-- page 7 of 16 -->

outperform SatMAE by an average of 5.6% and ConvMAE by an average of 2.4% across all resolutions and datasets (see Table 1). UC Merced at 100% of the true GSD is the only evaluation where SatMAE outperforms Scale-MAE. The supplementary material contains an extensive table demonstrating kNN classification results with varying k.

**Linear probing and finetuning** We perform linear classification on the RESISC-45 and FMoW-RGB datasets. We fine-tune for 50 epochs using the same hyperparameter settings as SatMAE [13]: a base learning rate of $5 \times 1 0 ^ { - 3 }$ , a weight decay of $5 \times 1 0 ^ { - 3 }$ . We do not use temporal data for classification. For RESISC-45, we fine-tune for 100 epochs with a base learning rate of $4 \times 1 0 ^ { - 3 }$ , a weight decay of $5 \times 1 0 ^ { - 3 }$ , and a global batch size of 256 across 2 GPUs. The learning rate on the backbone is multiplied by a factor of 0.1. We use RandomResizedCrop for augmentation. We train on 224x224 images and evaluate 256x256 images because we found evaluating at a higher scale improves the performance of all models. We report both the performance of end-to-end fine-tuning and linear probing with a frozen backbone. The linear probing setup was the same as finetuning except the learning rate was 0.1. The results are shown in Table 2 and Table 3.

| Model | Backbone | Frozen/Finetune |
| --- | --- | --- |
| Scale-MAE | Vit-Large | 89.6/95.7 |
| SatMAE [13] | Vit-Large | 88.3/94.8 |
| ConvMAE [21] | ConvVit-Large | 81.2/95.0 |
| MAE [26] | Vit-Large | 88.9/93.3 |

Table 2. Transfer classification results on RESISC-45. Frozen indicates a linear probe and finetune is a full end-to-end finetuning of the entire model.

| Model | Backbone | Top-1/Top-5 |
| --- | --- | --- |
| Scale-MAE | ViT-Large | 77.9/94.3 |
| SatMAE † [13] | ViT-Large | 72.4/91.9 |
| MAE [26] | ViT-Large | 68.4/90.3 |
| ConvMAE [21] | ConvVit-Large | 74.1/91.4 |
| SatMAE ∗ [13] | ViT-Large | 77.8/- |
| GASSL [4] | ResNet-50 | 71.55/- |
| MoCo-V2 [27] | ResNet-50 | 64.34/- |

Table 3. Full finetuning results on FMoW-RGB. †: We reproduce SatMAE and ConvMAE by taking their publicly available codebases and pretraining on FMoW dataset for 800 epochs. The results differ from their reported results, but are evaluated consis tently with ours. \* Reports the results from the SatMAE paper [13].

**Semantic segmentation transfer** We use the SpaceNet v1 building segmentation dataset [53] to evaluate semantic segmentation results on contrastive and MAE-based pretraining

methods. Prior methods relied on the PSANet [68] segmentation architecture, while Scale-MAE uses the UperNet [58] segmentation architecture which is more common for ViT backbones. For even comparison, we test the current stateof-the-art SatMAE and ConvMAE methods with UperNet as well. Results are detailed in Table 4.

| Method | Backbone | Model | mIoU |
| --- | --- | --- | --- |
| Sup. (Scratch) | ResNet50 | PSANet | 75.6 |
| GASSL [3] | ResNet50 | PSANet | 78.5 |
| Sup. (Scratch) | ViT-Large | PSANet | 74.7 |
| SatMAE [13] | ViT-Large | PSANet | 78.1 |
| Sup. (Scratch) | ViT-Large | UperNet | 71.6 |
| Vanilla MAE | ViT-Large | UperNet | 77.9 |
| SatMAE | ViT-Large | UperNet | 78.0 |
| ConvMAE | ViT-Large | UperNet | 77.6 |
| Scale-MAE | ViT-Large | UperNet | 78.9 |

Table 4. Semantic segmentation results on SpaceNet v1. Scale-MAE outperforms other methods across backbone and segmentation architectures, where Sup. (Scratch) indicates a supervised model trained from scratch (a randomly initialized network).

With the same pretraining settings, Scale-MAE outperforms SatMAE by 0.9 mIoU, ConvMAE by 1.3 mIoU, and a vanilla MAE by 1.0 mIoU. Scale-MAE outperforms all other prior work, including GASSL [3], which SatMAE did not outperform on the mean Intersection over Union (mIoU) metric for semantic segmentation. Particularly, Scale-MAE increases the gap in performance as the resolution of input imagery becomes coarser, highlighting the absolute scaleinvariance introduced by our method.

In Figure 6, we compare SpaceNet v1 evaluations across downscaled images (at 50%, 75%, and 100% of the original image size) for Scale-MAE, SatMAE, and ConvMAE. Similar to the classification results, Scale-MAE maintains higher semantic segmentation performance over both methods, even with images at a coarser GSD. In fact, the performance gap grows at coarser GSDs. Compared to the next-best-performing method at the input GSD, Scale-MAE is 0.9 mIoU higher, at 75% GSD Scale-MAE is 1.2 mIoU higher, and at 50% Scale-MAE is 1.7 mIoU higher.

In Table 5, we further evaluate Scale-MAE, SatMAE, and ConvMAE across SpaceNet v1, SpaceNet v2 [53], INRIA Aerial Image [44], and GID-15 [59] remote sensing datasets at native resolution. Scale-MAE outperforms both comparable methods across all benchmarks.

## 4.2. Ablations

We ablate the key components of the Scale-MAE pretraining framework. For these experiments, we use a lightweight pretraining setting, where we pretrain for 300 epochs on

<!-- page 8 of 16 -->

![Chart block](images/page008-block01.png)

Figure 6. SpaceNet v1 evaluation across downscaled images for both Scale-MAE and SatMAE. Scale-MAE maintains higher semantic segmentation performance over SatMAE, even with images of coarser GSD.

|  | SN1RI | SH | S VE | N2PA | KH | INR.- | G15- |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Conv. | 77.6 | 78.7 | 82.2 | 78.3 | 74.8 | 82.2 | 37.4 |
| Sat. | 78.0 | 81.9 | 86.6 | 80.3 | 76.1 | 83.0 | 44.3 |
| Scale | 78.9 | 82.2 | 87.4 | 81.1 | 77.1 | 84.2 | 46.2 |

Table 5. mIoU on semantic segmentation tasks. SN1/2 (SpaceNet v1/2), RI: Rio, SH: Shanghai, VE: Vegas, PA: Paris, KH: Khartoum; INR: INRIA; G15: GID-15. Conv., Sat., and Scale. are ConvMAE, SatMAE, and Scale-MAE.

| Method | GSDPE | KNN 50% | KNN 100% |
| --- | --- | --- | --- |
| Vanilla MAE |  | 72.8 | 77.8 |
| Vanilla MAE | ! | 75.4 | 78.5 |
| MAE + LP |  | 75.3 | 79.6 |
| Scale-MAE | ! | 78.1 | 80.7 |

Table 6. Ablation results indicating the importance of GSDPE as determined by a KNN classification on RESISC-45 at a relative GSD of 50% and 100% of its native GSD. Using the GSDPE leads to better performance for both Scale-MAE and the Vanilla MAE. MAE + LP denotes the vanilla MAE with the addition of our progressive Laplacian decoder.

FMoW (rather than 800) and use a ViT-Base encoder (rather than ViT-Large), and evaluate using a kNN evaluation on RESISC-45 at 100% and 50% of its native GSD. The key contributions that we ablate are as follows: the GSD positional encoder in Table 6, in which we find that the GSD postional encoder benefits both Scale-MAE and Vanilla MAE across resolutions. In Table 8, we see that the number of transformer layers can be reduced from 8 to 3 compared to a Vanilla MAE, which results in a performance improvement. The standard masking rate of 75% still appears optimal for Scale-MAE according to the results in Table 7.

In Table 9 we ablate the necessity of the low and high resolution reconstructions. Specifically, we test reconstructing the low resolution image only, the high resolution image, and

| Mask Rate | KNN 50% | KNN 100% |
| --- | --- | --- |
| 70% | 77.3 | 79.3 |
| 75% | 78.1 | 80.7 |
| 80% | 78.1 | 79.9 |

Table 7. Ablation results indicating that a 75% mask rate is optimal as determined by a KNN classification on RESISC-45 at a relative GSD of 50% and 100% of its native GSD.

a combined image (rather than independent low/high recon structions). In this case, when the high resolution component is reconstructed, we do not use the low-resolution residual, but rather, directly reconstruct the high resolution result. The “Combined” entry combines the low and high resolution results instead of treating them as separate learning objectives. The separate low/high resolution reconstructions obtain the best performance and robustness to changes in scale.

## 5. Discussion

In this section, we share observations about Scale-MAE, sketch our vision for future work, and discuss high-level questions about Scale-MAE.

**Computational complexity.** Scale-MAE requires a much smaller decoder than vanilla MAE—instead of a decoder depth of eight, Scale-MAE works well with a depth of three. In fact, with 322.9M vs 329.5M parameters using ViT-Large, Scale-MAE is smaller than vanilla MAE. However, GPU memory usage for equal batch sizes are higher for Scale-MAE since we reconstruct a higher resolution image in the Scale-MAE Decoder.

**Multi-spectrality and modality.** Electro-optical (EO) satellites, such as the ones comprising the datasets mentioned in this work, capture light at different wavelengths. Each wavelength has a different sensor, and each sensor can have a different resolution. Scale-MAE requires input tensors to be stacked to pass through the model. This means that we are unable to use Scale-MAE when the input image’s bands are all of different GSDs. Additionally, synthetic aperture radar (SAR) imagery is another form of remote sensing where resolution varies across a single band. Extending Scale-MAE to work with different resolution bands and modalities is reserved for future work.

**Can the Scale-MAE methodology be applied to other backbones?** Methods such as ConvNeXt [42] provide competitive performance compared to Transformers. The core components of our work can be integrated, with additional work, into different architectures. The Laplacian Decoder in Scale-MAE can be engineered to ingest convo-

<!-- page 9 of 16 -->

| Decoding Layers | KNN 50% | KNN 100% |
| --- | --- | --- |
| 1 | 76.0 | 78.4 |
| 2 | 77.9 | 80.4 |
| 3 | 78.1 | 80.7 |
| 4 | 77.5 | 80.0 |
| 8 | 77.7 | 78.9 |

Table 8. Ablation results indicating that fewer transformer layers in the decoding stage tend to work better for Scale-MAE as determined by a KNN classification on RESISC-45 at a relative GSD of 50% and 100% of its native GSD.

| Low Res High Res Combined | KNN 50% | KNN 100% |
| --- | --- | --- |
| ! | 77.6 | 80.2 |
| ! | 72.9 | 74.3 |
| ! | 77.2 | 80.3 |
| ! ! | 78.1 | 80.7 |

Table 9. These ablation results indicate that reconstructing both the low resolution and high resolution components lead to robust performance. Note: when the high resolution component is reconstructed, the low-resolution residual is not used—the high resolution result is directly reconstructed. The “Combined” entry merges the low and high resolution results instead of treating them as separate losses. The evaluations are a kNN classification (k=20) on RESISC-45 at relative GSDs 50% and 100% of its native GSD.

lutional feature maps. Existing work on scale-aware CNNs can be extended to work with the Laplacian Decoder.

**Evaluating on more remote sensing datasets.** The field of remote sensing has had a renaissance in the last five years with the amount of available datasets. These can be generic, like Functional Map of the World, to highly specific, such as identifying illegal airstrips in Brazil [1, 8] or identifying illegal fishing vessels [47]. In fact, there are so many small, specific remote sensing datasets that entire review papers are written to enumerate them [60]. We chose to focus datasets with properties of remote sensing that are relevant to multiscale representation learning.

## 6. Conclusion

Remote sensing imagery has accelerated the rate of scientific discovery in a broad set of disciplines. With increasingly precise methods to extract environmental indicators using computer vision methods, automated understanding of re motely sensed sources has become a mainstay in scientific literature. Remote sensing payloads are diverse and capture data at a wide range of resolutions, a feature heavily utilized by scientists. Current computer vision methods for remote sensing necessitate the training of a new model per input resolution. Not only is the training process expensive, but the overhead of curating a dataset at multiples scales makes

this a daunting task.

We introduce Scale-MAE, a pretraining framework which introduces scale invariance into encoders that are used for a diverse set of downstream tasks. Our insights into scale-inclusive positional encodings and progressive multi-frequency feature extraction result in models that perform significantly better than state-of-the-art pretraining methods across (1) multiple scales and (2) many benchmarks.

Our goal is to take the extremely diverse and rich source of information present in remote sensing imagery and make it simple to use with minimal training iterations required. With the introduction of Scale-MAE, we hope to further accelerate the rate at which scientific disciplines create impact.

## Acknowledgements

We deeply thank Kyle Michel from Meta for providing us with his help during our time of need. Satellite imagery and derived images used in this paper in are from datasets which redistribute imagery from Google Earth, DigitalGlobe, and Copernicus Sentinel 2022 data. Trevor Darrell’s group was supported in part by funding from the Department of Defense as well as BAIR’s industrial alliance programs. Ritwik Gupta is supported by the National Science Foundation under Grant No. DGE-2125913.

## References

[1] Manuela Andreoni, Blacki Migliozzi, Pablo Robles, and Denise Lu. The Illegal Airstrips Bringing Toxic Mining to Brazil’s Indigenous Land. The New York Times, 2022.

[2] Saeed Anwar, Salman Khan, and Nick Barnes. A deep journey into super-resolution: A survey. ACM Computing Surveys (CSUR), 53(3):1–34, 2020.

[3] Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, and Stefano Ermon. Geography-Aware Self-Supervised Learning. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10161–10170, Montreal, QC, Canada, Oct. 2021. IEEE.

[4] Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, and Stefano Ermon. Geography-aware self-supervised learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10181–10190, 2021.

[5] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–1828, 2013.

[6] P. Burt and E. Adelson. The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications, 31(4):532–540, Apr. 1983.

[7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging Properties in Self-Supervised Vision Transformers. In

<!-- page 10 of 16 -->

2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9630–9640, Montreal, QC, Canada, Oct. 2021. IEEE.

[8] Fen Chen, Ruilong Ren, Tim Van de Voorde, Wenbo Xu, Guiyun Zhou, and Yan Zhou. Fast Automatic Airport Detection in Remote Sensing Images Using Convolutional Neural Networks. Remote Sensing, 10(3):443, Mar. 2018.

[9] Xinlei Chen and Kaiming He. Exploring Simple Siamese Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15750–15758, 2021.

[10] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8628–8638, 2021.

[11] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote Sensing Image Scene Classification: Benchmark and State of the Art. Proceedings of the IEEE, 105(10):1865–1883, Oct. 2017.

[12] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional Map of the World. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6172–6180, Salt Lake City, UT, June 2018. IEEE.

[13] Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, and Stefano Ermon. SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery, Oct. 2022.

[14] Dengxin Dai and Wen Yang. Satellite Image Classification via Two-Layer Sparse Coding With Biased Image Representation. IEEE Geoscience and Remote Sensing Letters, 8(1):173–176, Jan. 2011.

[15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[16] Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In International conference on machine learning, pages 647–655. PMLR, 2014.

[17] Dumitru Erhan, Aaron Courville, Yoshua Bengio, and Pascal Vincent. Why does unsupervised pre-training help deep learning? In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 201–208. JMLR Workshop and Conference Proceedings, 2010.

[18] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6824–6835, 2021.

[19] Pedro Felzenszwalb, David McAllester, and Deva Ramanan. A discriminatively trained, multiscale, deformable part model. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8, June 2008.

[20] Anthony Fuller, Koreen Millard, and James R. Green. SatViT: Pretraining Transformers for Earth Observation. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2022.

[21] Peng Gao, Teli Ma, Hongsheng Li, Ziyi Lin, Jifeng Dai, and Yu Qiao. ConvMAE: Masked Convolution Meets Masked Autoencoders, May 2022.

[22] Yuan Gao, Xiaojuan Sun, and Chao Liu. A general self-supervised framework for remote sensing image classification. Remote Sensing, 14(19):4824, 2022.

[23] Golnaz Ghiasi and Charless C. Fowlkes. Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016, Lecture Notes in Computer Science, pages 519–534, Cham, 2016. Springer International Publishing.

[24] Ian J. Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, Cambridge, MA, USA, 2016. [http://www.deeplearningbook.org](http://www.deeplearningbook.org).

[25] Ritwik Gupta, Colorado Reed, Anja Rohrbach, and Trevor Darrell. Accelerating Ukraine Intelligence Analysis with Computer Vision on Synthetic Aperture Radar Imagery. http://bair.berkeley.edu/blog/2022/03/21/ukraine-sar-maers/, Mar. 2022.

[26] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked Autoencoders Are Scal- ´ able Vision Learners, Dec. 2021.

[27] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

[28] Yutong He, Dingjie Wang, Nicholas Lai, William Zhang, Chenlin Meng, Marshall Burke, David Lobell, and Stefano Ermon. Spatial-temporal super-resolution of satellite im agery via conditional pixel synthesis. Advances in Neural Information Processing Systems, 34:27903–27915, 2021.

[29] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Introducing Eurosat: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. In IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, pages 204–207, July 2018.

[30] Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In International conference on machine learning, pages 4182–4192. PMLR, 2020.

[31] X Hu, H Mu, X Zhang, Z Wang, T Tan, and J Meta-SR Sun. A magnification-arbitrary network for super-resolution. In Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, pages 15–20, 2019.

[32] Damian Ibanez, Ruben Fernandez-Beltran, Filiberto Pla, and ˜ Naoto Yokoya. Masked auto-encoding spectral-spatial transformer for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing, 2022.

[33] Pengpeng Ji, Shengye Yan, and Qingshan Liu. Region-Based Spatial Sampling for Image Classification. In 2013 Seventh International Conference on Image and Graphics, pages 874–879, July 2013.

[34] Jan J. Koenderink. The structure of images. Biological Cybernetics, 50(5):363–370, Aug. 1984.

<!-- page 11 of 16 -->

[35] Pawel Kowaleczko, Tomasz Tarasiewicz, Maciej Ziaja, Daniel Kostrzewa, Jakub Nalepa, Przemyslaw Rokita, and Michal Kawulok. Mus2: A benchmark for sentinel-2 multi-image super-resolution. arXiv preprint arXiv:2210.02745, 2022.

[36] S. Lazebnik, C. Schmid, and J. Ponce. Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), volume 2, pages 2169–2178, June 2006.

[37] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015.

[38] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, ´ Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photorealistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4681–4690, 2017.

[39] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature Pyramid Networks for Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2117–2125, 2017.

[40] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg. SSD: Single Shot MultiBox Detector. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016, Lecture Notes in Computer Science, pages 21–37, Cham, 2016. Springer International Publishing.

[41] Yufei Liu, Xiaorun Li, Ziqiang Hua, Chaoqun Xia, and Liaoying Zhao. A band selection method with masked convolutional autoencoder for hyperspectral image. IEEE Geoscience and Remote Sensing Letters, 2022.

[42] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022.

[43] Gabriel Machado, Edemir Ferreira, Keiller Nogueira, Hugo Oliveira, Matheus Brito, Pedro Henrique Targino Gama, and Jefersson Alex dos Santos. AiRound and CV-BrCT: Novel Multiview Datasets for Scene Classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 14:488–503, 2021.

[44] Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat, and Pierre Alliez. Can semantic labeling methods generalize to any city? the inria aerial image labeling benchmark. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS). IEEE, 2017.

[45] Malachy Moran, Kayla Woputz, Derrick Hee, Manuela Girotto, Paolo D’Odorico, Ritwik Gupta, Daniel Feldman, Puya Vahabi, Alberto Todeschini, and Colorado J. Reed. Snowpack Estimation in Key Mountainous Water Basins from Openly-Available, Multimodal Data Sources, Aug. 2022.

[46] Maxim Neumann, Andre Susano Pinto, Xiaohua Zhai, and ´ Neil Houlsby. In-domain representation learning for remote sensing. ArXiv, abs/1911.06721, 2019.

[47] Fernando Paolo, Tsu-ting Tim Lin, Ritwik Gupta, Bryce Goodman, Nirav Patel, Daniel Kuster, David Kroodsma, and Jared Dunnmon. xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery. In Thirty Sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, Oct. 2022.

[48] Xiaoman Qi, Panpan Zhu, Wang Yuebin, Liqiang Zhang, Junhuan Peng, Mengfan Wu, Jialong Chen, Xudong Zhao, Ning Zang, and P. Takis Mathiopoulos. MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding. ISPRS Journal of Photogrammetry and Remote Sensing, 169:337–350, Nov. 2020.

[49] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. Preprint, 2018.

[50] A. Rosenfeld and M. Thurston. Edge and Curve Detection for Visual Scene Analysis. IEEE Transactions on Computers, C-20(5):562–569, May 1971.

[51] Jacob Shermeyer and Adam Van Etten. The effects of superresolution on object detection performance in satellite imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–0, 2019.

[52] Beth Tellman, Jonathan A. Sullivan, and Colin S. Doyle. Global Flood Observation with Multiple Satellites. In Global Drought and Flood, chapter 5, pages 99–121. American Geophysical Union (AGU), 2021.

[53] Adam Van Etten, Dave Lindenbaum, and Todd M. Bacastow. SpaceNet: A Remote Sensing Dataset and Challenge Series, July 2019.

[54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[55] Qi Wang, Shaoteng Liu, Jocelyn Chanussot, and Xuelong Li. Scene Classification With Recurrent Attention of VHR Remote Sensing Images. IEEE Transactions on Geoscience and Remote Sensing, 57(2):1155–1167, Feb. 2019.

[56] Yifan Wang, Federico Perazzi, Brian McWilliams, Alexander Sorkine-Hornung, Olga Sorkine-Hornung, and Christopher Schroers. A Fully Progressive Approach to Single-Image Super-Resolution. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 977–97709, Salt Lake City, UT, USA, June 2018. IEEE.

[57] Zhirong Wu, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. Unsupervised Feature Learning via Non-Parametric Instance Discrimination. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3733–3742, 2018.

[58] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified Perceptual Parsing for Scene Understanding. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, editors, Computer Vision – ECCV 2018, volume 11209, pages 432–448. Springer International Publishing, Cham, 2018.

<!-- page 12 of 16 -->

[59] Xin-Yi Tong, Gui-Song Xia, Qikai Lu, Huangfeng Shen, Shengyang Li, Shucheng You, Liangpei Zhang. Land-cover classification with high-resolution remote sensing images using transferable deep models. Remote Sensing of Environment, doi: 10.1016/j.rse.2019.111322, 2020.

[60] Zhitong Xiong, Fahong Zhang, Yi Wang, Yilei Shi, and Xiao Xiang Zhu. EarthNets: Empowering AI in Earth Observation, Oct. 2022.

[61] Xingqian Xu, Zhangyang Wang, and Humphrey Shi. Ultrasr: Spatial encoding is a missing key for implicit image function-based arbitrary-scale super-resolution. arXiv preprint arXiv:2103.12716, 2021.

[62] Shengye Yan, Xinxing Xu, Dong Xu, Stephen Lin, and Xuelong Li. Beyond Spatial Pyramids: A New Feature Extraction Framework with Dense Spatial Sampling for Image Classification. In European Conference on Computer Vision 2012, pages 473–487, Aug. 2012.

[63] Shengye Yan, Xinxing Xu, Dong Xu, Stephen Lin, and Xuelong Li. Image Classification With Densely Sampled Image Windows and Generalized Adaptive Multiple Kernel Learning. IEEE Transactions on Cybernetics, 45(3):381–390, Mar. 2015.

[64] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, Jing-Hao Xue, and Qingmin Liao. Deep learning for single image super-resolution: A brief review. IEEE Transactions on Multimedia, 21(12):3106–3121, 2019.

[65] Yi Yang and Shawn Newsam. Bag-of-visual-words and spatial extensions for land-use classification. In Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems, GIS ’10, pages 270–279, New York, NY, USA, Nov. 2010. Association for Computing Machinery.

[66] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European conference on computer vision, pages 818–833. Springer, 2014.

[67] Renrui Zhang, Ziyu Guo, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li, and Peng Gao. Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training, Oct. 2022.

[68] Hengshuang Zhao, Yi Zhang, Shu Liu, Jianping Shi, Chen Change Loy, Dahua Lin, and Jiaya Jia. PSANet: Pointwise Spatial Attention Network for Scene Parsing. In Proceedings of the European Conference on Computer Vision (ECCV), pages 267–283, 2018.

<!-- page 13 of 16 -->

## A. Datasets

In our experiments, we used a total of ten datasets (Table 10) for the tasks of land-use/land-cover classification and semantic segmentation. There are a large amount of remote sensing datasets in existence. Many remote sensing datasets fundamentally capture the same data with minor changes in location or distribution. We selected datasets with key, representative properties. These properties include (1) a diversity in the amount of kinds of classes/objects represented, (2) a large spectrum of ground sample distances from (ideally) known sensor configurations, and (3) pansharpened, othrorectified, and quality controlled imagery and labels. We capture these properties in Table 10.

## A.1. Diversity in classes

For both pretraining and downstream evaluations, it is a desirable property to include as much geographic and class diversity as possible. In order to capture a wide amount of classes in remote sensing, it is necessary to include multiple localities and environments. This property serves as a proxy for the amount of unique “features” available in the dataset.

| Dataset | Resolution (px) | GSD (m) | Number of Images | Number of Classes | Task Type |
| --- | --- | --- | --- | --- | --- |
| AiRound [43] | 500 | 0.3 - 4800 | 11,753 | 11 | C |
| CV-BrCT [43] | 500 | 0.3 - 4800 | 24,000 | 9 | C |
| EuroSAT [29] | 64 | 10 | 27,000 | 10 | C |
| MLRSNet [48] | 256 | 0.1 - 10 | 109,161 | 46 | C |
| Optimal-31 [55] | 256 | 0.5 - 8 | 1,860 | 31 | C |
| RESISC-45 [11] | 256 | 0.2 - 30 | 31,500 | 45 | C |
| UC Merced [65] | 256 | 0.3 | 2,100 | 21 | C |
| WHU-RS19 [14] | 256 | 0.5 | 1050 | 19 | C |
| fMoW [12] | Various | 0.3 | 1,047,691 | 62 | C |
| SpaceNet v1 [53] | Various | 0.5 | 6,940 | 2 | SS |

Table 10. Statistics of all datasets used in our experiments. Task types are classification (C) and semantic segmentation (SS).

## A.2. Spectrum of GSDs

Scale-MAE is built to be invariant to the input absolute scale of the dataset. Many datasets are collected from a single sensor and processed in a uniform fashion. To validate that our method works with many resolutions, we included datasets which are collected from a variety of sensors but then processed in a uniform fashion. This excludes differences in processing as a factor affecting our experiments and narrowly targets resolution instead.

## A.3. Quality control

It is hard to assess the quality of remote sensing datasets without manually verifying a majority of instances of the data. We mandated that images used are pansharpened (and therefore the highest resolution possible to extract from the sensor), orthorectified (and therefore well-aligned with the geodetic ellispoid), and projected to the same coordinate reference system. This eliminates large differences in sensor-to-image processing.

## B. Laplacian and Upsampling Block Architectures

Figure 7 illustrates the architecture of Laplacian and Upsampling block architectures described below.

## B.1. Laplacian Block

Laplacian Blocks are used to reconstruct the target at a specific resolution and frequency. A Laplacian Block consists of a chain of Feature Mapping Block, which distills information at a specific frequency, followed by one final Reconstruction Block, which generates the final output. A Feature Mapping Block consists of a 3x3 depth-wise convolution layer with GELU activation, followed by 1x1 convolution. A Reconstruction Block consists of a 4x4 transpose convolution layer followed by a

<!-- page 14 of 16 -->

<table><tbody><tr><td colspan="2">Dataset Res</td><td>Scale. Sat. Conv.</td><td>Scale. Sat. Conv.</td><td>Scale. Sat. Conv.</td></tr><tr><td>AiRound</td><td>163264128256496</td><td>0.401 0.375 0.4230.561 0.510 0.5390.689 0.607 0.6580.743 0.650 0.6810.729 0.662 0.6580.670 0.664 0.620</td><td>0.396 0.367 0.4010.536 0.491 0.5170.643 0.579 0.6210.690 0.600 0.6220.678 0.621 0.6020.609 0.613 0.566</td><td>0.370 0.355 0.4030.541 0.492 0.5390.692 0.604 0.6660.749 0.660 0.6900.731 0.663 0.6760.685 0.669 0.632</td></tr><tr><td>CV-BrCT</td><td>163264128256496</td><td>0.522 0.478 0.5670.653 0.615 0.6560.744 0.701 0.7110.763 0.725 0.7320.761 0.725 0.7270.737 0.727 0.709</td><td>0.485 0.443 0.5130.588 0.560 0.5920.674 0.635 0.6440.710 0.662 0.6670.694 0.666 0.6640.656 0.657 0.631</td><td>0.524 0.475 0.5850.695 0.644 0.6990.780 0.727 0.7540.805 0.758 0.7820.802 0.770 0.7710.792 0.771 0.765</td></tr><tr><td>EuroSAT</td><td>163264</td><td>0.744 0.727 0.8260.901 0.876 0.8980.956 0.931 0.940</td><td>0.699 0.695 0.7880.869 0.854 0.8630.935 0.913 0.914</td><td>0.751 0.729 0.8350.912 0.871 0.9090.960 0.934 0.947</td></tr><tr><td>MLRSNet</td><td>163264128256</td><td>0.563 0.491 0.6070.772 0.677 0.7440.893 0.815 0.8510.936 0.875 0.8940.918 0.892 0.882</td><td>0.535 0.461 0.5490.726 0.625 0.6880.849 0.754 0.7920.892 0.814 0.8340.862 0.840 0.817</td><td>0.551 0.479 0.6170.772 0.684 0.7620.911 0.839 0.8760.950 0.899 0.9180.940 0.913 0.910</td></tr><tr><td>OPTIMAL-31</td><td>163264128256</td><td>0.354 0.322 0.4390.574 0.500 0.5870.793 0.609 0.6980.816 0.670 0.7140.739 0.681 0.646</td><td>0.312 0.298 0.3700.567 0.508 0.5450.742 0.561 0.5980.731 0.646 0.5950.653 0.638 0.550</td><td>0.317 0.319 0.4180.565 0.519 0.5610.782 0.646 0.6880.809 0.694 0.7250.761 0.731 0.693</td></tr><tr><td>RESISC</td><td>163264128256</td><td>0.382 0.347 0.4580.628 0.527 0.6010.798 0.667 0.7310.864 0.748 0.7980.826 0.758 0.762</td><td>0.370 0.327 0.4280.597 0.505 0.5680.754 0.631 0.6770.819 0.699 0.7430.761 0.708 0.690</td><td>0.353 0.323 0.4350.609 0.508 0.5920.803 0.667 0.7340.882 0.762 0.8170.850 0.771 0.788</td></tr><tr><td>UC Merced</td><td>163264128256</td><td>0.524 0.472 0.5980.767 0.670 0.6830.842 0.795 0.7710.858 0.788 0.7500.762 0.802 0.700</td><td>0.400 0.370 0.4620.605 0.535 0.5930.719 0.729 0.6520.662 0.738 0.6550.595 0.757 0.590</td><td>0.512 0.488 0.6170.828 0.682 0.7260.884 0.842 0.8450.884 0.847 0.8380.851 0.842 0.817</td></tr><tr><td>WHU-RS19</td><td>163264128256</td><td>0.545 0.445 0.5760.650 0.729 0.6700.850 0.805 0.8330.970 0.910 0.8820.960 0.940 0.892</td><td>0.400 0.380 0.5620.610 0.675 0.5760.770 0.730 0.6800.890 0.890 0.6850.880 0.925 0.709</td><td>0.525 0.490 0.6310.760 0.690 0.7540.920 0.840 0.8370.985 0.895 0.9410.975 0.945 0.931</td></tr></tbody></table>

Table 11. Scale-MAE outperforms SatMAE and ConvMAE on kNN classification across a variety of k, across a variety of resolutions. kNN Classification results for Scale-MAE, SatMAE and ConvMAE across a variety of k. Resolution is reported in pixels.

3x3 depth-wise convolution layer, a 1x1 convolution layer, and a 2x2 transpose convolution layer. In our experiments, we have two Feature Mapping Blocks per Laplacian Block.

<!-- page 15 of 16 -->

![Image block](images/page015-block01.png)

Figure 7. (top) The Laplacian Block (LB) is a fully convolutional architecture consists of a chain of Feature Mapping Block followed by one final Reconstruction Block. (bottom) The UpSampling Block (UB) consists of a series of transpose convolution layers separated by LayerNorm and GELU activation.

## B.2. Upsampling Block

Upsampling Blocks are used to upsample the feature map to a higher resolution. It consists of a series of 2x2 transpose convolution layers with LayerNorm and GELU activation between them. The number of such transposed convolution layers are a function of the output and input resolution. This is a progressive process in which we repetitively upsample the feature map by a factor of 2 until we reach the desired target resolution. Figure 7 illustrates the architecture of these two blocks.

## C. Evaluation Details

As discussed in the main experimental section, we investigated the quality of representations learned from Scale-MAE pretraining through a set of experiments that explore their robustness to scale as well as their transfer performance to additional tasks. We provide more information and details on these evaluations here. In order to compare with SatMAE [13] and ConvMAE [21], for our main experiments, we pretrained Scale-MAE with a ViT-Large model using the Functional Map of the World (FMoW) RGB training set, which consists of 363.6k images of varying image resolution and GSD. The initial higher resolution image $I _ { \mathrm { h r } }$ is taken as a random $4 4 8 \mathrm { { \tt { p } } x ^ { 2 } }$ crop of the input image, and the input image I is then a downsampled $2 2 4 \mathrm { { \tt { p x } } ^ { 2 } }$ from $I _ { \mathrm { h r } }$ . The low frequency groundtruth is obtained by downscaling $\bar { I } _ { \mathrm { h r } } \; \mathrm { t o } \; 1 4 \mathrm { p x } ^ { 2 }$ and then upscaling to $2 2 4 \mathrm { { \tt { p x } } ^ { 2 } }$ , while the high frequency groundtruth is obtained by downscaling $I _ { \mathrm { h r } }$ to $5 6 \mathrm { { p x } ^ { 2 } }$ and then upscaling to $4 4 8 \mathrm { { \tt p x } ^ { 2 } }$ and subtracting this image from $I _ { \mathrm { h r } }$ . This is a common method for band pass filtering used in several super resolution works, where a high to low to high resolution interpolation is used to obtain only low frequency results, and then high frequency results are obtained by subtracting the low frequency image.

As further discussed in the main experimental section, we evaluate the quality of representations from Scale-MAE by freezing the encoder and performing a nonparametric k-nearest-neighbor (kNN) classification with eight different remote sensing imagery classification datasets with different GSDs, none of which were encountered during pretraining. All kNN evaluations were conducted on 4 GPUs. Results are in Table 11. The kNN classifier operates by encoding all train and validation instances, where each embedded instance in the validation set computes the cosine distance with each embedded instance in the training set, where the instance is classified correctly if the majority of its k-nearest-neighbors are in the same class as the validation instance. The justification for a kNN classifier evaluation is that a strong pretrained network will output semantically grouped representation for unseen data of the same class. This evaluation for the quality of representations occurs

<!-- page 16 of 16 -->

![Image block](images/page016-block01.png)

Figure 8. Visualization of Segmentation Results on SpaceNet. The left, center, right columns are ground truth labels, Scale-MAE and vanilla MAE, respectively. The top row shows a 0.3m GSD image and the bottom row shows a 3.0m GSD image. As shown in the figure, Scale-MAE performs better at both higher and lower GSDs.

in other notable works [7, 9, 57].

## D. Visualization of SpaceNet Segmentation

Figure 8 shows an additional set of segmentation examples comparing Scale-MAE and vanilla MAE pre-trained on FMoW and finetuned on SpaceNet v1. The left, center, right columns are ground truth labels, Scale-MAE and vanilla MAE respectively. The top row shows a 0.3m GSD image and the bottom row shows a 3.0m GSD image. As shown in the figure, Scale-MAE performs better at both higher and lower GSDs.

## E. Glossary

## E.1. Ground sample distance

Ground sample distance (GSD) is the distance between the center of one pixel to the center of an adjacent pixel in a remote sensing image. GSD is a function of sensor parameters (such as its dimensions and focal length), image parameters (the target dimensions of the formed image), and the geometry of the sensor with respect to the object being imaged on the Earth. Remote sensing platforms frequently have multiple sensors to capture different wavelengths of light. Each of these sensors have varying parameters, resulting in different GSDs for an image of the same area. Additionally, the ground is not a uniform surface with changes in elevation common across the swath of the sensor. In total, a remote sensing platform has a sense of absolute scale that varies along two dimensions: (1) spectrally depending on the sensor used to capture light, and (2) spatially depending on surface elevation.