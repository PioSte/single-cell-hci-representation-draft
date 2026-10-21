<!-- page 1 of 20 -->

arXiv:2503.10845v2 [cs.LG] 1 Aug 2025

# Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation

Leonard Waldmann<sup>1,2∗</sup> Ando Shah<sup>1∗</sup> Yi Wang<sup>2</sup> Nils Lehmann<sup>2,3</sup> Adam J. Stewart<sup>2,3</sup> Zhitong Xiong<sup>2</sup> Xiao Xiang Zhu<sup>2,3</sup> Stefan Bauer<sup>2</sup> John Chuang<sup>1</sup>

<sup>1</sup>University of California, Berkeley; <sup>2</sup>Technical University of Munich; <sup>3</sup>Munich Center for Machine Learning correspondence to ando@berkeley.edu

## Abstract

Earth observation (EO) data features diverse sensing platforms with varying spectral bands, spatial resolutions, and sensing modalities. While most prior work has constrained inputs to fixed sensors, a new class of any-sensor foundation models able to process arbitrary sensors has recently emerged. Contributing to this line of work, we propose Panopticon, an any-sensor foundation model built on the DINOv2 framework. We extend DINOv2 by (1) treating images of the same geolocation across sensors as natural augmentations, (2) subsampling channels to diversify spectral input, and (3) adding a cross attention over channels as a flexible patch embedding mechanism. By encoding the wavelength and modes of optical and synthetic aperture radar sensors, respectively, Panopticon can effectively process any combination of arbitrary channels. In extensive evaluations, we achieve state-of-the-art performance on GEO-Bench, especially on the widely-used Sentinel-1 and Sentinel-2 sensors, while out-competing other any-sensor models, as well as domain adapted fixed-sensor models on unique sensor configurations. Panopticon enables immediate generalization to both existing and future satellite missions, advancing sensor-agnostic EO.

## 1. Introduction

Earth observation (EO) satellites provide essential data for monitoring our planet, from detecting illegal mining [58] and dark vessels [47], to field delineation [28] and yield prediction [52]. Recent years have witnessed substantial progress towards remote sensing foundation models (RSFMs) that have adapted computer vision methods designed for RGB imagery to EO data [12, 45, 51, 72].

However, compared to natural images, remote sensing data is remarkably heterogeneous [53, 61, 64]. EO platforms generate tremendously varied data across multiple dimensions, including spatial resolution (centimeters to kilometers), spectral characteristics (multispectral, hyperspectral, radar), revisit times (continuous to static), swath widths, preprocessing levels (raw, top of atmosphere, surface reflectance), and sensing mechanisms (active and passive). While RSFMs began as sensor-specific models [12, 26, 36, 56, 67], the current trend is towards more flexible models that can deal with a variety of EO data [1, 17, 34, 59, 70, 75].

Most recently, an emerging class of any-sensor models has begun addressing the fundamental challenge of sensor agnosticism in EO. With only a few examples at the time of writing [50, 74], these models process data from arbitrary sensor configurations—including unseen combinations of spectral bands, resolutions, and modalities—without requiring sensor-specific adaptations during inference. They achieve this by decoupling spectral and spatial processing while incorporating knowledge of sensor-specific properties such as spectral response functions (SRFs) and ground sampling distances (GSDs).

In this work, we propose Panopticon, a new any-sensor model (see Section A for the etymology of this name). Building upon the DINOv2 [46] framework, we make three key modifications for stronger integration of satellite data as a distinct data modality [53]. First, instead of distilling knowledge within a single image, we regard a given geolocation or “footprint” as the object of interest and consider snapshots from different sensors of that footprint as augmented views of the same object. This approach enables us to sample a broad spectrum of natural sensor-to-sensor variation by incorporating images from different channel characteristics, modalities, timestamps, and processing levels. Second, in addition to standard DINOv2 spatial augmentations, we apply a spectral augmentation by subsampling channels of multispectral (MS), hyperspectral (HS), and synthetic aperture radar (SAR) training data. Third, to

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">∗ Denotes co-first authorship, ordered randomly. Co-first authors will prioritize their names on their resumes/websites Code: [https://github.com/Panopticon-FM/panopticon](https://github.com/Panopticon-FM/panopticon)</span></small>

<!-- page 2 of 20 -->

![Image block](images/page002-block01.png)

Figure 1. Architecture of Panopticon. The highlighted blocks indicate modifications to the DINOv2 framework. When generating views, images of the same geographic footprint captured by different sensors at different times, scales, etc., are treated as augmentations. The spectral patch embedding block uses a shared patchifier to tokenize incoming views and a cross attention with a single query to fuse tokens within each spatial patch across channels.

handle views with varying numbers of channels, we implement a cross-attention over channels with added positional embeddings that encode sensor-specific spectral information into a unified representation.

To evaluate spectral flexibility, we simulate new sensors from HS datasets using spectral convolution [6], and subject models to reduced spectral and scale information to assess their invariance to these properties, in addition to assessing performance on sensors not encountered during pre-training. We demonstrate that Panopticon also achieves state-of-the-art performance on GEO-Bench [32] which comprises of popular sensors such as Sentinel-1 (S1) and Sentinel-2 (S2). To address sensor distribution shifts in fixed-sensor models, we employ patch embedding retraining as parameter-efficient domain adaptation [5, 24]. A surprising finding is that DINOv2 serves as a remarkably strong baseline in these comparisons, hinting at the nature of future work. Through comprehensive experiments across 23 MS, HS, and SAR datasets, we provide valuable insights into the capabilities and limitations of current approaches to sensor-agnostic learning in EO.

In summary, our contributions are as follows:

• We introduce Panopticon, a RSFM capable of building representations for any MS, HS, or SAR sensor, regardless of GSD or spectral characteristics.

• While maintaining competitive-to-state-of-the-art performance on established benchmarks, Panopticon outperforms other any-sensor models on more unique sensor

configurations that cover important application domains.

• We find that spectral progressive pre-training combined with a wider embedding dimension and the use of spectral embeddings in the channel attention module are crucial for optimal performance.

## 2. Related work

**Fixed-sensor RSFMs** Approaches vary on how to process different sensors with the same model. USat [25] and msGFM [21] use sensor-specific patch embeddings feeding into a single encoder, with USat adding positional information to the tokens. SatMAE [12], Galileo [60], and AnySat [1] employ more granular approaches with patch embeddings for groups of channels. Video-inspired models like Prithvi [27] and S2MAE [35] patchify in spatiotemporal or spatio-spectral dimensions. Another line of work uses separate encoders and aligns the resulting sensorspecific representations through contrastive and reconstruction loss, as in CROMA [17], modified contrastive loss, as in IAI-SimCLR [49], or regularization of covariance matrices, as in DeCUR [69].

**Any-sensor RSFMs** To the best of our knowledge, two general-purpose any-sensor models have been proposed, both based on the MAE [22] architecture. DOFA [74] proposes dynamic weight generation with a hypernetwork [20] based on wavelengths of the input channels for the other-

<!-- page 3 of 20 -->

![Image block](images/page003-block01.png)

Figure 2. View Generation. Given a footprint, we randomly sample a snapshot performing an implicit spectral, scale, and temporal augmentation. To increase spectral variance, we apply a spectral crop, and following DINOv2, we apply a spatial crop and resize. We omit additional flips and color jittering for simplicity. Local and global views are created analogously with different augmentation parameters.

wise fixed-shaped input projection and final layer of the decoder. SenPa-MAE [50] patchifies each channel separately with the same projection and adds positional, GSD, and spectral embeddings, which are computed from the spectral response function (SRF) of the channel, to the tokens in the encoder and decoder. Concurrent to this work, Copernicus-FM [71] extends DOFA with non-spectral modalities for all Copernicus mission sensors.

**Sensor views as augmentations** Several RSFMs have utilized the naturally occurring temporal and spectral differences between images of the same footprint as augmentations. In particular, DINO-MC [73] uses temporal positive pairs, SeCo [40] incorporates seasonal differences into their branches, and CaCo [38] defines change-aware positive and negative temporal pairs. On the other hand, DINO-MM [66] randomly selects different sensors for each view as a spectral augmentation. Finally, A2MAE [76] applies metainfoaware masking to generate both spectrally and temporally varied tokens.

**DINOv2** DINO [7] matches discrete probability distributions computed by teacher and student networks from differently augmented local and global views of the same image. Student weights are updated with backpropagation while teacher weights are assigned as exponential moving average on the student weights. To prevent collapse, the teacher applies centering and sharpening. DINOv2 [46] combines DINO with the patch-wise iBOT loss [77] and scales training. Importantly, a strong DINOv2 ViT-B checkpoint distilled from a ViT-g run is available.

## 3. Method

We propose Panopticon, which builds on the DINOv2 architecture [46] with changes to view generation and patch embedding, as shown in Figure 1.

## 3.1. Sensor views as augmentation of a footprint

The DINO framework self-distills knowledge between different augmented views of the same image [7]. Instead of considering a single image as the object of interest, we consider a geographic footprint as the object of interest and treat images from different sensors as views and therefore augmentations of the same object. Since augmentations define invariances, our model is invariant particularly to spectral configurations but also to GSDs, acquisition times, and processing levels, c.f. Table 1.

## 3.2. Local and global view generation

Given a footprint with a set of snapshots X from different sensors of that footprint, let us first generate a local view. We start by selecting a snapshot $x   \in   X$ with C channels, therefore implicitly applying a combination of spectral, scale, and temporal augmentations. Since the number of unique sensors in the training dataset is limited in practice, we increase the variance of the spectral information by subsampling channels. In particular, we first sample the number of channels $C ^ { \prime } \in \{ \operatorname* { m i n } ( C , C _ { \mathrm { l o w } } ^ { l } ) , . . . , \operatorname* { m i n } ( C , C _ { \mathrm { h i g h } } ^ { l } ) \}$ within bounds $C _ { \mathrm { l o w } } ^ { l } , C _ { \mathrm { h i g h } } ^ { l }$ and then select $C ^ { \prime }$ unique channels from x. All sampling is done uniformly at random. As in DINO, we apply a random resize crop resizing to the spatial shape $W ^ { l } \mathop { \times } ^ { \frown } \widehat { H ^ { l } } .$ , c.f. Figure 2, as well as additional flips and color jittering. We repeat this process to generate $n ^ { l }$ local views. Analogously, we generate $n ^ { g }$ global views with parameters $C _ { \mathrm { l o w } } ^ { g } , C _ { \mathrm { h i g h } } ^ { g } , W ^ { g } , H ^ { g }$ . In correspondence with

<!-- page 4 of 20 -->

| Dataset | Locations | Sensors | GSD (m) | Footprint (km2) | Sensor Modalities (# bands) | Temporal Range | Processing Level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fMoW [9, 12] | 90 K | QB, GE, WV, S2 | 0.3–10 | 0.2–25 | RGB (3), MSI (4, 8, 12) | 2002–2020 | L2A |
| MMEarth [42] | 1.2 M | S1, S2 | 10 | 1.7 | SAR (8), MSI (12) | 2017–2020 | L1C, L2A |
| SatlasPretrain [3] | 770 K | S1, S2, L9 | 10–100 | 25 | SAR (2), MSI (9), Thermal (2) | 2022 | L1C |
| SpectralEarth [5] | 415 K | EnMAP | 30 | 14.7 | HSI (202) | 2022–2024 | L2A |

Table 1. Pretraining Datasets. fMoW refers to the combined fMoW-full and fMoW-Sentinel datasets. Locations refers to the number of unique footprints, where footprint sizes are shown in square kilometers. The parentheses around modalities indicate the number of channels for that modality. MSI = multispectral imager, HSI = hyperspectral imager, QB = QuickBird-1, GE = GeoEye-2, WV = WorldView, S1 = Sentinel-1, S2 = Sentinel-2, and L9 = Landsat 9.

DINOv2, we set $n ^ { l }   =   4 ,   n ^ { g }   =   2 ,   H ^ { l }   =   W ^ { l }   =   9 6 ,$ and $H ^ { g } = W ^ { g } = 2 2 4$ . To be able to fit a full S2 image into a global crop, we set $C _ { \mathrm { h i g h } } ^ { g } = 1 3$ and for maximum flexibility, we set $C _ { \mathrm { l o w } } ^ { l }   =   1$ . Finally, we set $C _ { \mathrm { h i g h } } ^ { l }   =   C _ { \mathrm { l o w } } ^ { g }   =   4 ,$ , c.f. Section G.

## 3.3. Attention over channels as patch embedding

Let $x \in \mathbb { R } ^ { C \times H \times W }$ be an input image to the patch embedding (PE) where C is the number of channels, H the height, and W the width. Since C varies across images, we cannot use the standard 2D convolution of ViTs [14] as PE. Instead, as in Bao et al. [2], we first patchify and embed each channel with the same 2D convolution to $x _ { p } \in \mathbb { R } ^ { L \times C \times D }$ , where L is the number of patches and D the embedding dimension of the backbone. Similarly to Nguyen et al. [43], we then reduce C within each patch by a cross attention with a single learned query $q \in \mathbb { R } ^ { 1 \times D }$ and $x _ { p }$ as keys and values to $\tilde { x } _ { \mathrm { o u t } }   \in   \mathbb { R } ^ { L \times \tilde { D } }$ . We perform the cross attention on a dimension $D _ { \mathrm { a t m } } ,$ , i.e., query, key, and value projections as well as the final projection of the cross attention map to and, respectively, from $D _ { \mathrm { a t n } }$ . We find that $D _ { \mathrm { a t t n } } \: > \: D$ increases performance, c.f. Section 5.

To ingest spectral information about the individual channels, we add the following embeddings to $x _ { p } .$ . For optical channels, we choose the standard positional encoding [65] at the position equal to the central wavelength λ of the channel in nanometers (e.g., 664 nm for red), i.e.

$$
\mathrm{PE} _ {(\text {pos}, 2 i)} = \sin \left(\omega_ {i} \lambda\right),\tag{1}
$$

$$
\mathrm{PE} _ {(\text {pos}, 2 i + 1)} = \cos \left(\omega_ {i} \lambda\right),\tag{2}
$$

where $\omega _ { i } = 1 / 1 0 0 0 0 ^ { 2 i / D }$

We categorize SAR channels into 12 possible categories based on their 3 possible orbit direction (i.e. ascending, descending, unknown/both) and 4 possible transmit and receive polarizations (i.e. VV, VH, HH, HV), providing 12 possible polarization-orbit combinations. We learn embeddings of dimension $D / 3$ for each transmit, receiver and orbit variables, and concatenate them to the full embedding

vector. This follows known results that SAR data generated across polarizations [11, 63] and orbits [37, 64] are distinct. Note that we only infuse spectral information but no information on GSD, time, or processing levels.

## 3.4. Spectral progressive pre-training

We found that naively training Panopticon with diverse sensors yields poor performance. Inspired by progressive pre-training [41, 74], we pre-train in two stages with reduced spectral diversity in the first stage, followed by maximizing spectral diversity in the second. In Section 5, we show that this substantially increases performance.

## 3.5. Implementation details

**Datasets** We use the datasets presented in Table 1 with the following rationale:

• **fMoW**, comprising of fMoW [9] and fMoW-Sentinel [12], combines popular optical wavelengths (RGB, S2) and commercial MS sensors - WorldView (WV) 2/3, with high variation in GSD and very high resolution RGB images.

• **SatlasPretrain** [3] pairs S1 SAR, S2 optical, and Landsat 9 optical and thermal sensors across globally diverse footprints.

• **MMEarth** [42] provides additional SAR configurations—ascending and descending orbits with VV, VH, HV, and HH polarizations not found in SatlasPretrain.

• **SpectralEarth** [5] is a hyperspectral dataset with 202 channels, thus provides great spectral diversity over optical wavelengths.

Each dataset is standardized with mean and standard deviations computed over its training split. Additional details are available in Section C.

**Model weights and training** We use a ViT-B as backbone with $D \quad = \quad 7 6 8$ and load the published DINOv2 weights. Since weights for classification and iBOT heads

<!-- page 5 of 20 -->

<table><tbody><tr><td rowspan="2"></td><td colspan="3">Commercial Sensors</td><td colspan="2">Synthesized Sensors</td><td colspan="3">OOD Sensors</td></tr><tr><td>SpaceNet1WV7 miJacc (%) ↑</td><td>fMoW-10%WV8 Acc (%) ↑</td><td>Corine SuperDove mAP (%) ↑</td><td>Corine MODIS mAP (%)↑</td><td>Hyperview SuperDove MSE ↓</td><td>Hyperview MODIS MSE ↓</td><td>TC-10%GOES-16 MSE ↓</td><td>DT-10%Himawari MSE ↓</td></tr><tr><td>DINOv2-PE</td><td>90.0 / 90.6*</td><td>29.2 / 47.0*</td><td>66.8 / 80.6*</td><td>63.6 / 80.7</td><td>1.043 / 0.340*</td><td>0.639 / 0.335*</td><td>0.635</td><td>0.502</td></tr><tr><td>Croma-PE</td><td>89.4</td><td>33.7</td><td>75.1</td><td>78.4</td><td>0.343</td><td>0.519</td><td>0.394</td><td>0.925</td></tr><tr><td>Softcon-PE</td><td>90.1</td><td>32.8</td><td>77.3</td><td>80.2</td><td>0.353</td><td>0.533</td><td>0.471</td><td>0.810</td></tr><tr><td>Anysat-PE</td><td>87.7</td><td>26.2</td><td>73.4</td><td>76.6</td><td>0.326</td><td>0.519</td><td>0.449</td><td>1.406</td></tr><tr><td>SenPa-MAE</td><td>86.2</td><td>16.9</td><td>64.2</td><td>62.7</td><td>0.366</td><td>0.355</td><td>-</td><td>-</td></tr><tr><td>DOFA</td><td>89.9</td><td>42.0</td><td>81.0</td><td>80.2</td><td>0.334</td><td>0.338</td><td>0.385</td><td>1.032</td></tr><tr><td>Panopticon</td><td>90.3</td><td>50.3</td><td>85.8</td><td>86.2</td><td>0.313</td><td>0.321</td><td>0.315</td><td>0.963</td></tr></tbody></table>

Table 2. Spectral Generalization Performance. Synthesized sensor channels are generated from EnMAP source bands using spectral convolution to indicated target sensors, and tested on the SpectralEarth Corine dataset. Similarly, the Intuition-1 HS sensor is transformed to target bands for the Hyperview dataset. Target bands are generated through spectral convolution for Planet SuperDove’s (SD) 8 bands, or MODIS Terra’s 16 bands. Out-of-distribution (OOD) sensors refer to weather satellite channels with central wavelengths of 10.4 µm. ‘-’ indicates the inability of that model to adapt to the dataset. Best results in bold, second-best underlined. \*Trained re-initialized PE / frozen pre-trained PE using only RGB channels. DT = DigitalTyphoon, TC = TropicalCyclone, miJacc = micro Jaccard index

<table><tr><td rowspan="2"></td><td colspan="6">Classification (top-1 Accuracy %) ↑</td><td colspan="4">Segmentation (miJacc %) ↑</td></tr><tr><td>reBEN-10% S2</td><td>S1</td><td>m-eurosat S2</td><td>m-forestnet L8</td><td>m-so2sat S2</td><td>RESISC45 RGB</td><td>m-sacrop S2</td><td>m-cashew S2</td><td>m-nzcattle RGB</td><td>m-pv4ger RGB</td></tr><tr><td>DINOv2</td><td>80.1</td><td>-</td><td>95.5</td><td>53.5</td><td>60.8</td><td>94.0</td><td>51.2</td><td>65.9</td><td>92.7</td><td>96.9</td></tr><tr><td>CROMA</td><td>79.4</td><td>70.3</td><td>91.1</td><td>-</td><td>53.5</td><td>-</td><td>48.4</td><td>44.3</td><td>-</td><td>-</td></tr><tr><td>SoftCon</td><td>84.3</td><td>80.0</td><td>92.2</td><td>-</td><td>52.1</td><td>-</td><td>51.3</td><td>54.5</td><td>-</td><td>-</td></tr><tr><td>AnySat</td><td>76.8</td><td>64.4</td><td>87.6</td><td>50.9</td><td>42.5</td><td>65.5</td><td>39.5</td><td>38.8</td><td>92.5</td><td>92.2</td></tr><tr><td>Galileo</td><td>76.5</td><td>70.3</td><td>88.6</td><td>-</td><td>54.2</td><td>-</td><td>39.5</td><td>40.4</td><td>-</td><td>-</td></tr><tr><td>SenPa-MAE</td><td>63.8</td><td>-</td><td>77.5</td><td>33.5</td><td>33.7</td><td>28.2</td><td>39.3</td><td>40.7</td><td>89.5</td><td>78.3</td></tr><tr><td>DOFA</td><td>78.8</td><td>72.0</td><td>92.9</td><td>53.2</td><td>54.2</td><td>92.0</td><td>51.3</td><td>56.4</td><td>92.8</td><td>96.3</td></tr><tr><td>Panopticon</td><td>83.9</td><td>78.4</td><td>96.4</td><td>56.3</td><td>61.7</td><td>90.9</td><td>52.6</td><td>59.3</td><td>92.6</td><td>95.2</td></tr></table>

Table 3. Classification and segmentation on datasets consisting of well-known sensors from GEO-Bench, reBEN, and RESISC45. ‘- indicates the inability of that model to adapt to the dataset. As baseline we report the vanilla pre-trained DINOv2 weights only using RGB channels from the respective dataset, constituting a very strong baseline. Panopticon is in the top 2 models for all but 2 tasks. Best results in bold and second-best underlined. L8 = Landsat 8, miJacc = micro Jaccard index

are not provided and training is unstable when randomly initializing them, we pre-train the full DINOv2 weights with the standard patch embedding on fMoW using only the RGB channels and use those head weights as initialization for Panopticon. Finally, we use the attention over channels as PE with $D _ { \mathrm { a t n } } = 2 3 0 4$ resulting in 98.1 M parameters for the student, 12.9 M of which belong to the patch embedding.

We train our model in two stages using only fMoW in the first stage and all datasets in the second stage, c.f. Section 5. Following DINOv2, we artificially define an epoch as 1250 iterations and train for 87.5K iterations or 70 epochs on 16 A100 40GB GPUs with an effective batch size of 1200 for each stage. Further details can be found in the Section F.

## 4. Evaluation

We evaluate our model against DINOv2, fixed-sensor models, and any-sensor models in linear probing and kNN settings for 11 classification, 7 segmentation, and 4 regression downstream tasks. We evaluate sensor generalization in two specific ways: a) the ability to generalize to unseen sensor configurations, and b) the ability to generate representations that are scale and spectrally invariant. Moreover, we test performance on commonly utilized EO benchmarks such as GEO-Bench [32]. All evaluation is conducted on ViT-Base backbones. Additional details on dataset processing and evaluation protocols used throughout this section can be found in Table 1 and Section F.

<!-- page 6 of 20 -->

![Image block](images/page006-block01.png)

Figure 3. Representation stability under reduction of information. For m-eurosat, m-brick-kiln, and RESISC45, we report top-1 accuracy (acc) from kNN classification with k = 20 and for EnMAP-Corine, we report mean average precision (mAP) from linear probing. Panopticon is shown as a dark blue line in all plots. Top: Learning stable representations under spectral subsampling with 5 different subsets each and corresponding ±1 standard deviation bands, represented by shaded regions. Bottom: Learning scale-invariant representations by down- and up-sampling both train and test data.

## 4.1. Generalization to unique sensors

We evaluate Panopticon against state-of-the-art (SOTA) fixed-sensor models on datasets with sensor configurations that were not explicitly encountered during pre-training. Given the limited availability of open-access data and benchmarks from commercial sensors [54], we employ spectral convolution over HS bands to simulate existing sensors with different spectral characteristics; additional details in the Section E. Using the SpectralEarth-Corine dataset [5], we transform the HS bands to correspond to Planet’s Super-Dove sensor (8 bands) [48, 62] and MODIS Terra sensor (16 bands) [55], creating the 1 Corine-SD and 2 Corine-MODIS datasets, respectively. We apply the same transformation to the Hyperview [31] HS dataset, mapping it to 3 SuperDove and 4 MODIS bands. Finally, we employ the wind speed estimation tasks 5 TropicalCylone [39] and 6 DigitalTyphoon [30], useful for their extreme spectral distribution shifts with wavelengths of 10.4 µm. To adapt fixed-sensor models to new sensors, we re-train a 2D convolution as patch embedding while keeping the backbone frozen, denoted by the suffix “-PE”. This approach has shown to be a parameter-efficient domain adaptation strategy for EO foundation models [24]. This allows an otherwise fixed-sensor model to adapt to datasets originating from other sensors.

Additionally, we include benchmark datasets that con-

tain two commercial sensors: 7 SpaceNet 1 [15] which uses the WV2 sensor, and the 23 fMoW dataset that includes only the WV2+WV3 sensors (10%).

Table 2 summarizes the results. While the domainadapted fixed-sensor models work well and often outperform previous any-sensor models, Panopticon almost uniformly excels at these tasks, in most cases by a substantial margin.

## 4.2. Spectral invariance

Inspired by Reed et al. [51], we evaluate spectral invariance by progressively reducing the number of available channels as illustrated in Figure 3 (top). We employ k-nearest neighbors (kNN) classification on the datasets 8 EuroSAT [23] and 9 Brick Kiln [33], both with their GEO-Bench modifications [32] and use uniformly subsampled channels. For the HS 10 EnMAP-Corine dataset [5], we use linear probing (LP) and perform binned channel subsampling. For each setting, we sample 5 subsets and report mean and standard deviation of the task metric. Note that sampled channels are identical for all models. We only evaluate on any-sensor models since we are interested in representation quality without adjusting the backbone.

Panopticon retains the ability to generate high quality representations across almost the entire range of spectral subsampling, exhibiting high spectral invariance within

<!-- page 7 of 20 -->

these tasks. Of particular note is Panopticon’s reduced variance across subsampled channels indicating a higher level of channel, and therefore spectral invariance. Both DOFA and SenPa-MAE tend to retain stable performance across the range (albeit with higher variance), suggesting that the pre-training paradigms of any-sensor models likely contribute to this behavior. Additionally, we evaluate Panopticon for its cross-sensor generalization ability and the results can be found in Section D.1.

## 4.3. Scale invariance

We evaluate scale invariance through kNN classification across two multispectral and one RGB dataset, as presented in Figure 3 (bottom). The MS datasets include 8 EuroSAT [24] and 9 Brick Kiln [33], both modified according to GEO-Bench specifications [32]. For RGB assessment, we utilize the 11 RESISC45 [8] dataset since it exhibits a large variance in GSDs natively, which we further augment. The MS datasets feature fixed GSDs of 10 m. Similarly to the protocol established by Reed et al. [51], we downsample both train and test sets to progressively coarser GSDs to assess model robustness to spatial resolution degradation.

For the MS datasets, Panopticon’s representations remain very stable across the GSD range and perform better than the competition.

## 4.4. Performance on popular sensors

We compare existing fixed- and any-sensor models on the 6 classification and 6 segmentation datasets from GEO-Bench. We substitute 12 BigEarthNet-S2 and 13 BigEarthNet-S1 for reBEN, as it is upgraded with more robust splits and lower label noise [10]. These benchmark tasks represent a wide range of real-world downstream tasks on S1, S2, Landsat 8, RGB, and RGBN sensors. We include 11 RESISC45 [8] as an additional RGB land cover classification task, as well as an additional 9 tasks from 22 GEO-Bench [32]. An abridged summary of results is shown in Table 3. The complete GEO-Bench tables, along with the descriptions of the tasks, can be found in Section D.

Panopticon exhibits SOTA results on most tasks, and is competitive on the rest, showing that the model can also perform well on popular open access sensors.

## 5. Ablation studies

We perform ablation studies on both pre-training stages following the configuration of Section 3.5 but reducing the epochs to 30 with an effective batch size of 300. To evaluate performance over MS, SAR, and non-standard sensors with a smaller compute budget than in Section 4, we define the following metrics: i) $\mathbf { M S } _ { \mathrm { a c c } } \mathbf { : }$ average kNN accuracy on m-eurosat with and without RGB channels, ii) $\mathrm { S A R } _ { \mathrm { a c c } } ;$ average accuracy of kNN and linear probing on Eurosat-

sar [68], iii) $\mathrm{Sim}_{\mathrm{mAP}}:$ average mAP across two sensors simulated from HS Corine data by channel subsampling, and iv) Avg: average across all individual tasks. Details on metrics can be found in Section F and additional ablations studies in Section G. Our key findings are that spectral progressive pretraining, a larger dimensions in channel attention relative to the backbone, and spectral positional embeddings are crucial for good performance.

**Progressive pre-training** We ablate the dataset composition of our pre-training in Table 4. We can see that training with multiple datasets and, thus, more spectrally diverse inputs leads to suboptimal results. Note that, e.g., training with fMoW and MMEarth performs worse on all metrics than training on only MMEarth. This motivates our twostage pre-training approach. Following Table 4, we only train on fMoW in the first stage and increase spectral diversity by training on all datasets in the second stage. While possible, we found no benefits of using more than these two stages.

**Channel attention architecture** We sweep different number of heads $n _ { h }$ and embedding dimensions $D _ { \mathrm { a t m } }$ in the channel attention of our PE on stage 1 using two learning rates with 15 epochs each. Results can be seen in Table 5. Most notably, using $D _ { \mathrm { a t n } }   \geq   1 5 3 6$ improves performance significantly compared to $D _ { \mathrm { a t n } } \: = \: 7 6 8$ . The influence of $n _ { h }$ is less profound. Note that increasing $D _ { \mathrm { a t m } }$ comes with substantially increasing the number of parameters as can be seen in the second row of Table 5.

**Sensor views as augmentations** When generating a view of a given footprint, Panopticon samples a sensor image from all available images of that footprint as augmentation. In Table 4, we present an ablation (‘Single’) were we generate all views of a given footprint from a single image. This substantially reduces performance.

**Spectral embedding** In Table 4, we compare our proposed embedding of encoding individual wavelengths and SAR configurations (‘Fine-PE’) against a coarse embedding that only differentiates between SAR and optical channels (‘Coarse’), no embedding at all (‘None’), and an embedding that additionally encodes the standard deviation of the SRF of optical sensors (‘Fine-std’), see Section E for details. For optical sensors, having fine-grained spectral information substantially improves performance while adding information about the standard deviation does not seem to provide a benefit. For SAR, the coarse embedding may be sufficient. However, note that none of the downstream datasets include SAR data other than a single transmit and

<!-- page 8 of 20 -->

|  | Avg ↑ | MS<sub>acc</sub> ↑ | SAR<sub>acc</sub> ↑ | Sim<sub>mAP</sub> ↑ |
| --- | --- | --- | --- | --- |
| Stage 1 Default fMoW | 83.5 | 92.2 | 75.8 | 79.6 |
| Datasets MME | 81.7 | 86.4 | 81.7 | 80.8 |
| SP | 81.2 | 84.4 | 77.5 | 82.1 |
| fMoW, MME | 78.3 | 82.0 | 80.1 | 79.2 |
| fMoW, SP, MME, SE | 73.9 | 79.1 | 74.4 | 76.8 |
| Stage 2 Default Fine PE, multi-view | 86.8 | 93.1 | 82.2 | 83.4 |
| PE None | 84.8 | 91.5 | 81.2 | 80.7 |
| Coarse | 85.1 | 91.0 | 82.1 | 80.8 |
| Fine-std | 85.9 | 91.7 | 82.9 | 80.4 |
| Views Single | 82.2 | 84.2 | 78.8 | 83.0 |

Table 4. Ablations of major design decisions of Panopticon by pretraining stages. The metrics are averages in $\%$ over several tasks indicating performance on MS, SAR, and non-standard sensors simulated by channel subsampling. Best results within each stage in bold. SP = SatlasPretrain, SE = SpectralEarth, and MME = MMEarth.

<table><tr><td> $D_{\text{attn}}$ </td><td></td><td>768</td><td>1536</td><td>2304</td><td>3072</td></tr><tr><td># params (in M)</td><td></td><td>1.9</td><td>6.2</td><td>12.9</td><td>21.9</td></tr><tr><td rowspan="4"> $n_h$ </td><td>12</td><td>77.7</td><td>82.4</td><td>83.9</td><td>-</td></tr><tr><td>16</td><td>77.8</td><td>84.9</td><td>85.8</td><td>81.5</td></tr><tr><td>24</td><td>-</td><td>83.8</td><td>85.5</td><td>84.4</td></tr><tr><td>32</td><td>-</td><td>-</td><td>-</td><td>82.1</td></tr></table>

Table 5. Ablation of the channel attention architecture. $\mathrm { M S _ { a c c } }$ metric reported when changing the number of heads $n _ { h }$ in the rows and embedding dimension $D _ { \mathrm { a t m } }$ in the columns. The number of parameters of the channel attention module is shown in millions in the second row. Best results in bold, second-best underlined.

receiver polarization (VV+VH), limiting our ability to validate this outcome. For additional insights into design decisions, please refer to Table 7 and Section D.3.

## 6. Discussion and conclusion

In this work, we introduced Panopticon, a flexible anysensor foundation model that processes data from arbitrary sensor configurations without sensor-specific adaptations. By extending DINOv2 with multi-sensor view generation, spectral subsampling, and cross-attention over channels, our approach achieves state-of-the-art performance on standard benchmarks while demonstrating superior generalization across sensor configurations. Our rigorous evaluation tested the model’s sensor-agnostic capabilities through reduced spatial and spectral information tests and cross-sensor generalization. Our findings show that Panopticon consistently outperforms existing fixed-sensor and anysensor models in developing truly sensor-agnostic representations. Tests on datasets outside typical training dis tributions revealed that most models can adapt to novel

sensor modalities, though with varying success rates. Notably, DINOv2 with appropriate domain adaptation techniques proved a surprisingly strong baseline for RGB applications and certain MS use-cases.

Our evaluation across 23 diverse tasks represents significant progress in validating any-sensor capabilities, though limitations remain. We did not explore temporal invariance or equivariance—a critical direction for future work—nor fully investigate incorporating channel bandwidth information and using spectral convolution as augmentation strategies. Additionally, the absence of evaluation tasks utilizing diverse SAR channel combinations limited our ability to comprehensively test the model’s SAR generalization capabilities.

While Panopticon explicitly models channel-level sensor characteristics, it relies on view augmentations to learn other important EO invariances related to capture time, processing levels, and ground sampling distance. Though we specifically tested and validated the latter, the other properties proved challenging to isolate and evaluate. Our results suggest that combining explicit spectral modeling with diverse augmentation strategies offers a promising path toward truly sensor-agnostic Earth observation.

The development of robust any-sensor models is constrained by the scarcity of machine learning-ready data from diverse satellite platforms, impeding equitable planetary monitoring [54]. As remote sensing platforms proliferate, frameworks like Panopticon that generalize across sensor configurations will become increasingly valuable for maximizing scientific and societal benefits. Future work should incorporate additional sensor-specific inductive biases, expand temporal modeling capabilities, and develop comprehensive cross-sensor evaluation benchmarks.

<!-- page 9 of 20 -->

## Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No. DGE- 2125913. The authors gratefully acknowledge the computing time provided on the high-performance computer HoreKa by the National High-Performance Computing Center at KIT (NHR@KIT). This center is jointly supported by the Federal Ministry of Education and Research and the Ministry of Science, Research and the Arts of Baden-Wurttemberg, as part of the National High-Performance ¨ Computing (NHR) joint funding program<sup>1</sup>. HoreKa is partly funded by the German Research Foundation (DFG). The authors are also grateful to Prof. Joshua Blumenstock, and the Berkeley High Performance Computing (Savio) department for the provisioning of computing infrastructure.

## References

[1] Guillaume Astruc, Nicolas Gonthier, Clement Mallet, and Loic Landrieu. AnySat: An Earth observation model for any resolutions, scales, and modalities. arXiv preprint arXiv:2412.14123, 2024. 1, 2

[2] Yujia Bao, Srinivasan Sivanandan, and Theofanis Karaletsos. Channel vision transformers: An image is worth c x 16 x 16 words. arXiv preprint arXiv:2309.16108, 2023. 4, 6

[3] Favyen Bastani, Piper Wolters, Ritwik Gupta, Joe Ferdinando, and Aniruddha Kembhavi. Satlaspretrain: A largescale dataset for remote sensing image understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16772–16782, 2023. 4

[4] Jeremy Bentham. Panopticon or the inspection house. Bentham, Jeremy, 1791. 2

[5] Nassim Ait Ali Braham, Conrad M Albrecht, Julien Mairal, Jocelyn Chanussot, Yi Wang, and Xiao Xiang Zhu. Spectralearth: Training hyperspectral foundation models at scale. arXiv preprint arXiv:2408.08447, 2024. 2, 4, 6, 3

[6] Olivier Burggraaff. Biases from incorrect reflectance convolution. Optics express, 28(9):13801–13816, 2020. 2, 5

[7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, ´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging Properties in Self-Supervised Vision Transformers, 2021. arXiv:2104.14294 [cs]. 3

[8] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):1865–1883, 2017. 7

[9] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6172–6180, 2018. 4, 2

[10] Kai Norman Clasen, Leonard Hackel, Tom Burgert, Gencer Sumbul, Begum Demir, and Volker Markl. reben: Refined ¨ bigearthnet dataset for remote sensing image analysis. arXiv preprint arXiv:2407.03653, 2024. 7, 4

[11] Shane R Cloude and Eric Pottier. A review of target decomposition theorems in radar polarimetry. IEEE transactions on geoscience and remote sensing, 34(2):498–518, 1996. 4

[12] Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, and Stefano Ermon. SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In Advances in Neural Information Processing Systems, 2022. 1, 2, 4

[13] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. [https : / / github . com / open - mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 2020. 7

[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. 4

[15] Van Etten. Spacenet: A remote sensing dataset and challenge series. (No Title), 2018. 6

[16] Michel Foucault. Panopticism. In The information society reader, pages 302–312. Routledge, 2020. 2

[17] Anthony Fuller, Koreen Millard, and James Green. Croma: Remote sensing representations with contrastive radaroptical masked autoencoders. Advances in Neural Information Processing Systems, 36:5506–5538, 2023. 1, 2

[18] Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noord- ´ huis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour, 2018. 7

[19] Luis Guanter, Hermann Kaufmann, Karl Segl, Saskia Foerster, Christian Rogass, Sabine Chabrillat, Theres Kuester, Andre Hollstein, Godela Rossner, Christian Chlebek, et al. ´ The enmap spaceborne imaging spectroscopy mission for earth observation. Remote Sensing, 7(7):8830–8857, 2015. 3

[20] David Ha, Andrew Dai, and Quoc V. Le. Hypernetworks, 2016. 2

[21] Boran Han, Shuai Zhang, Xingjian Shi, and Markus Reichstein. Bridging remote sensors with multisensor geospatial foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27852–27862, 2024. 2

[22] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable ´ vision learners, 2021. 2

[23] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019. 6

[24] Chia-Yu Hsu, Wenwen Li, and Sizhe Wang. Geospatial foundation models for image analysis: Evaluating and enhancing nasa-ibm prithvi’s domain adaptability. International Journal of Geographical Information Science, pages 1–30, 2024. 2, 6, 7

[25] Jeremy Irvin, Lucas Tao, Joanne Zhou, Yuntao Ma, Langston Nashold, Benjamin Liu, and Andrew Y Ng. Usat: A unified

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>[https://www.nhr-verein.de/en/our-partners](https://www.nhr-verein.de/en/our-partners)</span></small>

<!-- page 10 of 20 -->

self-supervised encoder for multi-sensor satellite imagery. arXiv preprint arXiv:2312.02199, 2023. 2

[26] Johannes Jakubik, Sujit Roy, CE Phillips, Paolo Fraccaro, Denys Godwin, Bianca Zadrozny, Daniela Szwarcman, Carlos Gomes, Gabby Nyirjesy, Blair Edwards, et al. Foundation models for generalist geospatial artificial intelligence. arXiv preprint arXiv:2310.18660, 2023. 1

[27] Johannes Jakubik, Sujit Roy, C. E. Phillips, Paolo Fraccaro, Denys Godwin, Bianca Zadrozny, Daniela Szwarcman, Carlos Gomes, Gabby Nyirjesy, Blair Edwards, Daiki Kimura, Naomi Simumba, Linsong Chu, S. Karthik Mukkavilli, Devyani Lambhate, Kamal Das, Ranjini Bangalore, Dario Oliveira, Michal Muszynski, Kumar Ankur, Muthukumaran Ramasubramanian, Iksha Gurung, Sam Khallaghi, Hanxi, Li, Michael Cecil, Maryam Ahmadi, Fatemeh Kordi, Hamed Alemohammad, Manil Maskey, Raghu Ganti, Kommy Weldemariam, and Rahul Ramachandran. Foundation Models for Generalist Geospatial Artificial Intelligence, 2023. arXiv:2310.18660 [cs]. 2

[28] Hannah Kerner, Snehal Chaudhari, Aninda Ghosh, Caleb Robinson, Adeel Ahmad, Eddie Choi, Nathan Jacobs, Chris Holmes, Matthias Mohr, Rahul Dodhia, et al. Fields of the World: A machine learning benchmark dataset for global agricultural field boundary segmentation. arXiv preprint arXiv:2409.16252, 2024. 1

[29] Ethan King, Jaime Rodriguez, Diego Llanes, Timothy Doster, Tegan Emerson, and James Koch. Stars: Sensoragnostic transformer architecture for remote sensing. arXiv preprint arXiv:2411.05714, 2024. 5

[30] Asanobu Kitamoto, Jared Hwang, Bastien Vuillod, Lucas Gautier, Yingtao Tian, and Tarin Clanuwat. Digital typhoon: Long-term satellite image dataset for the spatio-temporal modeling of tropical cyclones. Advances in Neural Information Processing Systems, 36:40623–40636, 2023. 6

[31] Rıdvan Salih Kuzu, Frauke Albrecht, Caroline Arnold, Roshni Kamath, and Kai Konen. Predicting soil properties from hyperspectral satellite images. In 2022 IEEE International Conference on Image Processing (ICIP), pages 4296–4300, 2022. 6

[32] Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Bjorn L ¨ utjens, Jeremy Irvin, David ¨ Dao, Hamed Alemohammad, Alexandre Drouin, et al. Geobench: Toward foundation models for earth monitoring. Advances in Neural Information Processing Systems, 36, 2024. 2, 5, 6, 7, 3

[33] Jihyeon Lee, Nina R Brooks, Fahim Tajwar, Marshall Burke, Stefano Ermon, David B Lobell, Debashish Biswas, and Stephen P Luby. Scalable deep learning to identify brick kilns and aid regulatory capacity. Proceedings of the National Academy of Sciences, 118(17):e2018863118, 2021. 6, 7

[34] Xuyang Li, Danfeng Hong, and Jocelyn Chanussot. S2mae: A spatial-spectral pretraining foundation model for spectral remote sensing data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24088–24097, 2024. 1

[35] Xuyang Li, Danfeng Hong, and Jocelyn Chanussot. S2MAE: A spatial-spectral pretraining foundation model for spectral

remote sensing data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24088–24097, 2024. 2

[36] Lei Ma, Yu Liu, Xueliang Zhang, Yuanxin Ye, Gaofei Yin, and Brian Alan Johnson. Deep learning in remote sensing applications: A meta-analysis and review. ISPRS journal of photogrammetry and remote sensing, 152:166–177, 2019. 1

[37] Sahel Mahdavi, Meisam Amani, and Yasser Maghsoudi. The effects of orbit type on synthetic aperture radar (sar) backscatter. Remote sensing letters, 10(2):120–128, 2019. 4

[38] Utkarsh Mall, Bharath Hariharan, and Kavita Bala. Change-Aware Sampling and Contrastive Learning for Satellite Images. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5261–5270, Vancouver, BC, Canada, 2023. IEEE. 3

[39] Manil Maskey, Rahul Ramachandran, Iksha Gurung, Brian Freitag, Muthukumaran Ramasubramanian, and Jeffrey Miller. Tropical cyclone wind estimation competition dataset. Version 1.0, Radiant MLHub, 2022. 6

[40] Oscar Manas, Alexandre Lacoste, Xavier Giro-i Nieto, ˜ David Vazquez, and Pau Rodriguez. Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data, 2021. arXiv:2103.16607 [cs]. 3

[41] Mat´ıas Mendieta, Boran Han, Xingjian Shi, Yi Zhu, and Chen Chen. Towards geospatial foundation models via continual pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16806–16816, 2023. 4

[42] Vishal Nedungadi, Ankit Kariryaa, Stefan Oehmcke, Serge Belongie, Christian Igel, and Nico Lang. Mmearth: Exploring multi-modal pretext tasks for geospatial representation learning. arXiv preprint arXiv:2405.02771, 2024. 4, 3

[43] Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. Climax: A foundation model for weather and climate. arXiv preprint arXiv:2301.10343, 2023. 4

[44] Nicki Skafte Detlefsen, Jiri Borovec, Justus Schock, Ananya Harsh, Teddy Koker, Luca Di Liello, Daniel Stancl, Changsheng Quan, Maxim Grechkin, and William Falcon. Torch-Metrics - Measuring Reproducibility in PyTorch, 2022. 8, 9

[45] Mubashir Noman, Muzammal Naseer, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, and Fahad Shahbaz Khan. Rethinking transformers pre-training for multi-spectral satellite imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27811–27819, 2024. 1

[46] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1, 3

[47] Fernando Paolo, Tsu-ting Tim Lin, Ritwik Gupta, Bryce Goodman, Nirav Patel, Daniel Kuster, David Kroodsma, and Jared Dunnmon. xView3-SAR: Detecting dark fishing activity using synthetic aperture radar imagery. Advances in Neu-

<!-- page 11 of 20 -->

ral Information Processing Systems, 35:37604–37616, 2022. 1

[48] Planet Labs PBC. Understanding planetscope instruments, 2025. [https://developers.planet.com/docs/apis/data/sensors/](https://developers.planet.com/docs/apis/data/sensors/). 6

[49] Jonathan Prexl and Michael Schmitt. Multi-Modal Multi-Objective Contrastive Learning for Sentinel-1/2 Imagery. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Vancouver, BC, Canada, 2023. IEEE. 2

[50] Jonathan Prexl and Michael Schmitt. Senpa-mae: Sensor parameter aware masked autoencoder for multi-satellite self-supervised pretraining. arXiv preprint arXiv:2408.11000, 2024. 1, 3, 4

[51] Colorado J Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, and Trevor Darrell. Scale-mae: A scale-aware masked autoencoder for multiscale geospatial representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4088–4099, 2023. 1, 6, 7

[52] Felix Rembold, Clement Atzberger, Igor Savin, and Oscar Rojas. Using low resolution satellite imagery for yield pre-diction and yield anomaly detection. Remote Sensing, 5(4): 1704–1733, 2013. 1

[53] Esther Rolf, Konstantin Klemmer, Caleb Robinson, and Hannah Kerner. Mission critical–satellite data is a distinct modality in machine learning. arXiv preprint arXiv:2402.01444, 2024. 1

[54] Philippe Rufin, Patrick Meyfroidt, Felicia O Akinyemi, Lyndon Estes, Esther Shupel Ibrahim, Meha Jain, Hannah Kerner, Sa Nogueira Lisboa, David Lobell, Catherine ´ Nakalembe, et al. To enhance sustainable development goal research, open up commercial satellite image archives. Proceedings of the National Academy of Sciences, 122(7): e2410246122, 2025. 6, 8

[55] A Savtchenko, D Ouzounov, S Ahmad, J Acker, G Leptoukh, J Koziana, and D Nickless. Terra and aqua modis products available from nasa ges daac. Advances in Space Research, 34(4):710–714, 2004. 6

[56] Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ali Braham, Shradha Sehgal, Caleb Robinson, and Arindam Banerjee. SSL4EO-L: Datasets and foundation models for Landsat imagery. Advances in Neural Information Processing Systems, 36:59787–59807, 2023. 1

[57] Adam J Stewart, Caleb Robinson, Isaac A Corley, Anthony Ortiz, Juan M Lavista Ferres, and Arindam Banerjee. Torch-Geo: deep learning with geospatial data. In Proceedings of the 30th international conference on advances in geographic information systems, pages 1–12, 2022. 2, 3

[58] Merugu Suresh and Kamal Jain. Change detection and estimation of illegal mining using satellite images. In Proceedings of 2nd International conference of Innovation in Electronics and communication Engineering (ICIECE-2013), 2013. 1

[59] G Tseng, R Cartuyvels, I Zvonkov, M Purohit, D Rolnick, and H Kerner. Lightweight, pre-trained transform-

ers for remote sensing timeseries. arxiv. arXiv preprint arXiv:2304.14065, 2023. 1

[60] Gabriel Tseng, Anthony Fuller, Marlena Reil, Henry Herzog, Patrick Beukema, Favyen Bastani, James R Green, Evan Shelhamer, Hannah Kerner, and David Rolnick. Galileo: Learning global and local features in pretrained remote sensing models. arXiv preprint arXiv:2502.09356, 2025. 2

[61] Arsenios Tsokas, Maciej Rysz, Panos M Pardalos, and Kathleen Dipple. Sar data applications in earth observation: An overview. Expert Systems with Applications, 205:117342, 2022. 1

[62] Yu-Hsuan Tu, Kasper Johansen, Bruno Aragon, Marcel M El Hajj, and Matthew F McCabe. The radiometric accuracy of the 8-band multi-spectral surface reflectance from the planet superdove constellation. International Journal of Ap plied Earth Observation and Geoinformation, 114:103035, 2022. 6

[63] Fawwaz T Ulaby, Daniel Held, Myron C Donson, Kyle C McDonald, and Thomas BA Senior. Relating polaization phase difference of sar signals to scene properties. IEEE Transactions on Geoscience and Remote Sensing, GE-25(1): 83–92, 1987. 4

[64] Susan L Ustin and Elizabeth M Middleton. Current and nearterm advances in earth observation for ecological applications. Ecological Processes, 10(1):1, 2021. 1, 4

[65] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 4

[66] Yi Wang, Conrad M. Albrecht, and Xiao Xiang Zhu. Self-supervised Vision Transformers for Joint SAR-optical Representation Learning, 2022. arXiv:2204.05381 [cs]. 3

[67] Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong, Chenying Liu, Conrad M Albrecht, and Xiao Xiang Zhu. SSL4EO-S12: A large-scale multimodal, multitemporal dataset for self-supervised learning in Earth observation. IEEE Geoscience and Remote Sensing Magazine, 11(3):98–106, 2023. 1

[68] Yi Wang, Hugo Hernandez Hern ´ andez, Conrad M Albrecht, ´ and Xiao Xiang Zhu. Feature guided masked autoencoder for self-supervised learning in remote sensing. arXiv preprint arXiv:2310.18653, 2023. 7, 8

[69] Yi Wang, Conrad M. Albrecht, Nassim Ait Ali Braham, Chenying Liu, Zhitong Xiong, and Xiao Xiang Zhu. Decoupling Common and Unique Representations for Multimodal Self-supervised Learning, 2024. arXiv:2309.05300 [cs]. 2

[70] Yi Wang, Conrad M Albrecht, and Xiao Xiang Zhu. Multi-label guided soft contrastive learning for efficient earth observation pretraining. arXiv preprint arXiv:2405.20462, 2024. 1

[71] Yi Wang, Zhitong Xiong, Chenying Liu, Adam J. Stewart, Thomas Dujardin, Nikolaos Ioannis Bountos, Angelos Zavras, Franziska Gerken, Ioannis Papoutsis, Laura Leal-Taixe,´ and Xiao Xiang Zhu. Towards a unified copernicus foundation model for earth vision, 2025. 3

[72] Xinye Wanyan, Sachith Seneviratne, Shuchang Shen, and Michael Kirley. Dino-mc: Self-supervised contrastive learning for remote sensing imagery with multi-sized local crops. arXiv preprint arXiv:2303.06670, 2023. 1

<!-- page 12 of 20 -->

[73] Xinye Wanyan, Sachith Seneviratne, Shuchang Shen, and Michael Kirley. Extending global-local view alignment for self-supervised learning with remote sensing imagery, 2024. arXiv:2303.06670 [cs]. 3

[74] Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J Stewart, Joelle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le ¨ Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural plasticity-inspired foundation model for observing the earth crossing modalities. arXiv preprint arXiv:2403.15356, 2024. 1, 2, 4

[75] Zhitong Xiong, Yi Wang, Fahong Zhang, and Xiao Xiang Zhu. One for all: Toward unified foundation models for Earth vision. In IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium, pages 2734–2738. IEEE, 2024. 1

[76] Lixian Zhang, Yi Zhao, Runmin Dong, Jinxiao Zhang, Shuai Yuan, Shilei Cao, Mengxuan Chen, Juepeng Zheng, Weijia Li, Wei Liu, Wayne Zhang, Litong Feng, and Haohuan Fu. A\$ˆ{2}\$-MAE: A spatial-temporal-spectral unified remote sensing pre-training method based on anchor-aware masked autoencoder, 2024. arXiv:2406.08079 [cs] version: 2. 3

[77] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT Pre-Training with Online Tokenizer, 2022. arXiv:2111.07832 [cs]. 3

<!-- page 13 of 20 -->

# Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation Supplementary Material

## A. Why “Panopticon”?

The idea of the panopticon was first suggested by the philosopher Jeremy Bentham [4] as an ideal model for efficient prison design, where a single person could watch over an entire prison. Michel Foucault later reinterpreted it as a powerful metaphor for repressive systems of power, control and surveillance in modern societies [16], something that is perhaps even more relevant today with the proliferation of digital surveillance technologies.

We are well aware of the term’s loaded history and controversial connotations; so why choose it? We want to co-opt this term and flip its meaning, using it as a metaphor for systems that can keep a watchful and benevolent eye over our planet. Instead of surveilling people, Panopticon(s) can observe Earth itself—its changing landscapes, ecosystems, and climate patterns.

The beauty of our model is that, like the original panopticon concept, it provides comprehensive visibility from a single vantage point. But unlike Bentham’s prison design, our goal is not control and fear, but rather understanding and monitoring. There is also a technical parallel that we find fitting: the original panopticon was designed to see everything from a central position, regardless of where attention was directed. Similarly, our model can “look” through any sensor configuration without needing specific adaptations— a sensor-agnostic vision that mirrors the all-seeing nature of the conceptual panopticon, but repurposed for planetary good.

## B. Code and data availability

All data loaders and model code will be contributed to the TorchGeo library [57] for reproducibility and ease of future experimentation.

## C. Datasets

## C.1. Pre-training datasets

We organize the four pretraining datasets as shown in Table 1 of the main text by geographical footprint, where a footprint is defined by images having an exact geo-reference match. During pretraining, from each footprint, we randomly sample a number of “snapshots” as shown in Fig. 1.

**fMoW** fMoW [9] consists of data from 4 MS satellite sensors, QuickBird-1 and GeoEye-2 having 4 channels (RGB, NIR), while WorldView-2 and -3 have 8 channels, with GSDs typically ranging from 1–2 m. Additionally, the dataset also includes pansharpened versions of these same

![Image block](images/page013-block12.png)

Supp. Fig. 1. Snapshots: examples of snapshots taken from distinct footprints. Our pretraining dataset consists of various sensor modalities, channels, GSDs, scales, and timesteps acquired from across the Earth’s surface. Different channels, GSDs, and footprint sizes provide information about different attributes of the geospatial objects. Note: some images are shown in false colors to enable mapping from non-visible spectra.

images in RGB which have a GSD < 1 m. This dataset was chosen for its global spatial coverage, wide spectral coverage<sup>2</sup>and very low GSD values, along with extensive functional coverage of human modified land cover types. Moreover, this dataset also represents a large variance in time of capture, off-nadir angles, both of which affects illumination of targets. Since spatial footprints were available for all images, we generate an geographically indexed version of this dataset, which will be released with the rest of the code. Additionally, we remove images greater than 1024 px, which are typically the pansharpened RGB images, to reduce memory overheads.

**fMoW-Sentinel** fMoW-Sentinel [12] was created to be an exact copy of the locations captured by fMoW, but with Sentinel-2 imagery. We created a combined dataset from fMoW and fMoW-Sentinel by indexing by footprint and sensor type. Together, these two datasets capture surface properties from five separate sensor platforms between 2002 and 2022, providing a lot of natural variation for a given footprint. Finally, the footprints of each image vary

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>including “non-standard” bands, i.e. those with spectral coverage outside those of the popular open-data sources such as Landsat and Sentinel series.</span></small>

<!-- page 14 of 20 -->

SpectralEarth Corine: SD x S-2 x MODIS

![Chart block](images/page014-block02.png)

Supp. Fig. 2. Cross-sensor invariance. In addition to train/val/test splits, we also split datasets across sensors to explicitly test sensor invariance. y-axes on the heatmap represent training sensors and x-axes, test sensors. The diagonals represent same-sensor for train and test, and are grayed-out, while off-diagonal elements represent cross-sensor results with values expressed as percentage differences from the diagonals. This enables visualization of the change in cross-sensor performance relative to same sensor, expressed in (negative) percentages. Left: splits are across synthesized sensors from the HS EnMAP sensor using spectral convolution - MODIS, Sentinel-2 and Planet SuperDove, for any-sensor models. Right: reBEN; splits are implemented across Sentinel-1 and Sentinel-2 sensors, for all any-sensor and many-sensor models, except SenPaMAE which cannot process SAR imagery. Panopticon consistently outperforms other any-sensor models. Note that the value ranges differ for the two sub-figures.

tremendously, from 0.2 to 25 km<sup>2</sup>, providing a large range of features at different scales. This combined dataset consists of 89,666 unique footprints, where each footprint can have dozens of images across these sensors.

**MMEarth** Multi-modal Earth (MMEarth) [42] was released as a paired dataset of multiple modalities that include Sentinel-1 SAR, Sentinel-2 MS, elevation, and other paired modalities. Most importantly, the Sentinel-2 data included multiple processing levels (L1C and L2A), and Sentinel-1 data was captured in all 4 polarization combinations (VV, VH, HH, HV) and in both orbits (ascending, descending). This data was primarily included to model the effects of polarization and orbit for SAR data and processing levels for optical. Moreover, this extensive dataset comprises of 1,239,937 unique footprints equitably distributed according to land cover types, each of which providing a pair of S1 and S2 images.

**SpectralEarth** SpectralEarth [5] is the largest open source hyperspectral dataset available at the time of writing comprising of 450 K patches sampled globally by the EnMAP satellite [19], made available by the German Aerospace Center (DLR). This dataset additionally provides four downstream benchmark tasks using data from the same sensor, but utilizing non-overlapping patches, separate from

pretraining. This dataset was included for its rich spectral diversity, enabling the model to learn HS characteristics and simulate any arbitrary multispectral band. SpectralEarth provides 415 K unique footprints, each of which provides a single HS image of 202 bands.

**SatlasPretrain** SatlasPretrain consists of 30 TB of imagery across Sentinel-1, Sentinel-2, Landsat-9, and NAIP sensors. We utilize only the first three, as NAIP geographic coverage is limited to the United States and spectrally consists of only RGB and NIR bands. We created a unified indexed dataset comprising of 768,800 unique footprints, where each footprint can contain up to 3 sensor images taken across 2022. It is also the only large pretraining dataset to contain thermal images from the Landsat 9-TIRS sensor.

## C.2. Evaluation datasets

The utilize the following benchmark task and datasets. Where possible, we utilized existing Python libraries such as TorchGeo [57] and GEO-Bench [32]. For a complete list of datasets, please consult Tab. 6. Most datasets are implemented using the TorchGeo library [57], when available.

In the following, we outline any modifications we make to standard datasets:

<!-- page 15 of 20 -->

**SpaceNet** We utilize the SpaceNet 1 dataset from Torch-Geo, which is a building footprint segmentation task over the city of Rio de Janeiro with 8 band MS images and 3 band pansharpened RGB images captured by WorldView 2. We utilize only the 8-band images. Since the original dataset is only available with labels for the training set, we randomly split the dataset into training, validation and test splits with a 80:10:10 ratio.

## D. Additional Results

## D.1. Sensor invariance

We explicitly test for a model’s ability to generate stable representations regardless of the sensor input. To do this, we implement an additional split dimension on datasets that have paired sensors, splitting on the sensor. We use the 10 EnMAP-Corine [5] dataset where we employ spectral convolution to simulate MODIS, Sentinel-2, and Planet SuperDove sensors (Fig. 2 (left)) and the 12 reBEN [10] dataset that has paired imagery from Sentinel-1 and Sentinel-2 (Supp. Fig. 2 (right)). Models are trained on the training split of the dataset using a single sensor, while being validated and tested on a corresponding split of the dataset, using data from a different sensor. We then plot the relative difference to the samesensor setting as shown on the off-diagonal cells. This allows us to validate how close representations generated from one sensor are to ones generated from a different sensor on the same dataset and prediction objective. Ideally, the off-diagonal cell values are close to 0%, i.e. similar to the same-sensor setting. This is the case for Panopticon for the Corine dataset in Fig. 2 (left), where other any-sensor models struggle to generalize, especially when training on fewer bands (SD with 8 bands) and testing on sensors with more bands (MODIS with 16 bands). This effect is less pronounced in Fig. 2 (right), where the domain shift is very strong going from optical to SAR and vice-versa. However, even in this setting Panopticon outperforms all any-sensor and many-sensor models by 18% and 14%, for the S1→S2 and S2→S1 settings on average, respectively.

Absolute values for these experiments along with additional results on the fMoW dataset split according to three included sensors, can be seen in Supp. Fig. 3.

## D.2. Geo-Bench

We present the full results on the Geo-Bench classification datasets in Table 10 and on the segmentation datasets in Table 11, Table 12, and Table 13 reporting different metrics.

## D.3. Influence of individual design decisions

We evaluate the contributions of individual design decisions in Table 7. We identify spectral progressive pre-training and the sample sensor augmentation as the most influen-

tial design decisions. Interestingly, removing the DINOv2 weights and training from scratch only results in a performance decrease of 6.1 percentage points in the Avg metric.

## E. Utilizing complete spectral information

In the field of any-sensor models, DOFA [74] uses a hypernetwork and the mean of the SRF to learn spectral encodings per channel, while SenPaMAE [50] utilizes the full SRF. We experimented with the following mechanisms to incorporate SRF and/or bandwidth information per channel, in addition to the mean wavelength.

**Spectral integrated positional encoding** To achieve sensor agnosticism, we introduce a novel spectral integrated positional encoding (sIPE) that leverages known sensor characteristics. Building on the channel-wise attention mechanism from Nguyen et al. [43], we model each sensor’s SRF as an un-normalized Gaussian kernel characterized by its mean, $\mu$ and standard deviation, $\sigma .$ This was inspired by noticing that such a Gaussian kernel provides a relatively good fit for SRFs, such as Sentinel-2A and Landsat-9 OLI sensors as shown in Fig. 4. We also experimented with Epanechkinov kernel fitting, which provided better $R ^ { 2 }$ results, but since Gaussians are well understood and have closed-form solutions, they tend to be easier to work with. Hence, we model the SRF of a channel with the parameters $\{ \mu , \sigma \}$ of a Gaussian fit. We then extend ab solute positional embeddings [65] to incorporate this spectral information through integration against sinusoidal basis functions

$$
\left| \begin{array}{c}\mathrm{PE} _ {\mathrm{spectral}} (\mu , \sigma , 2 i) = \int e ^ {- \frac {(\mu - \lambda) ^ {2}}{2 \sigma^ {2}}} \cdot \sin (\omega_ {i} \lambda) \cdot \mathrm{d} \lambda , \\ \mathrm{PE} _ {\mathrm{spectral}} (\mu , \sigma , 2 i + 1) = \int e ^ {- \frac {(\mu - \lambda) ^ {2}}{2 \sigma^ {2}}} \cdot \cos (\omega_ {i} \lambda) \cdot \mathrm{d} \lambda , \end{array} \right|
$$

where $\begin{array} { r } { \omega = \frac { 1 } { 1 0 0 0 0 ^ { \frac { 2 i } { D } } } } \end{array}$ and $i \in ( 0 , D ]$

(3)

We derive the closed-form solution for Eq. (3) as follows:

$$
\begin{array}{r l} & {\mathrm{PE} _ {\mathrm{spectral}} (\mu , \sigma , 2 i) = \sigma \sqrt {2 \pi} \cdot e ^ {- \frac {\omega_ {i} ^ {2} \sigma^ {2}}{2}} \cdot \sin (\omega_ {i} \mu),} \\ & {\mathrm{PE} _ {\mathrm{spectral}} (\mu , \sigma , 2 i) = \sigma \sqrt {2 \pi} \cdot e ^ {- \frac {\omega_ {i} ^ {2} \sigma^ {2}}{2}} \cdot \cos (\omega_ {i} \mu).} \end{array}\tag{4}
$$

Eq. (4) allows us to efficiently generate spectral PEs that are added to channel tokens. Our hypothesis was that while the query learns how best to fuse spectral tokens across channels, it can only do so based on the extracted low-level features within those patches. Adding the sIPE to the patch tokens provides a spectral inductive bias to each token relative to its central wavelength and bandwidth, effectively grounding the patch token to its physical capture attributes.

<!-- page 16 of 20 -->

| Index | Name | Name used in this paper | Modifications |
| --- | --- | --- | --- |
| 1 | Corine [SuperDove] |  | Spectral convolution from EnMap to Planet Superdove |
| 2 | Corine [MODIS] |  | Spectral convolution from EnMap to MODIS |
| 3 | Hyperview [SuperDove] |  | Spectral convolution from Intuition to SuperDove |
| 4 | Hyperview [MODIS] |  | Spectral convolution from Intuition to MODIS |
| 5 | TropicalCyclone | TC | TorchGeo, we use 10% of train |
| 6 | DigitalTyphoon | DT | TorchGeo, we use 10% of train |
| 7 | SpaceNet 1 | SpaceNet 1 | Randomly split the train set into train, val, and test (80:10:10) |
| 8 | EuroSAT | m-eurosat | GEO-Bench |
| 9 | BrickKiln | m-brick-kiln | GEO-Bench |
| 10 | EnMAP-Corine | Corine | Original 202 band dataset |
| 11 | RESISC45 | RESISC | GEO-Bench |
| 12 | reBEN-S2 |  |  |
| 13 | reBEN-S1 |  |  |
| 14 | ForestNet | m-forestnet | GEO-Bench |
| 15 | So2Sat | m-so2sat | GEO-Bench |
| 16 | PV4Ger (cls.) | m-pv4ger | GEO-Bench |
| 17 | PV4Ger (segm.) | m-pv4ger-seg | GEO-Bench |
| 18 | Cheasapeake Landcover | m-chesapeake-landcover | GEO-Bench |
| 19 | Cashew Plantation | m-cashew-plantation | GEO-Bench |
| 20 | SA Crop Type | m-SA-crop-type | GEO-Bench |
| 21 | NZ Cattle | m-nz-cattle | GEO-Bench |
| 22 | NEON Tree | m-NeonTree | GEO-Bench |
| 23 | fMoW |  | Subset to WV2+WV3 sensors |

Table 6. Summary of all evaluation datasets and the modifications made to them.

| Individual contributions | A | vg ↑ |
| --- | --- | --- |
| Panopticon | 83.5 |  |
| - Spectral Progr. Pre-Training | 72.2 | -11.3 |
| - Sample sensor augm. | 73.2 | -10.3 |
| - Chn. attn. | 76.7 | -6.8 |
| - DINOv2 weights | 77.4 | -6.1 |
| - D<sub>attn</sub> to 2304 | 79.9 | -3.6 |
| - Spectral cropping | 81.3 | -2.2 |

Table 7. Impact of individual design choices. We take Panopticon, remove individual design choices, and observe the decrease in performance in the Avg metric. Progr.=progressive, augm.=augmentation, Chn. attn. = channel attention

**Spectral convolution** Inspired by King et al. [29], we employ spectral convolution [6] as a spectral augmentation for HS inputs. Given a source HS image comprised of i channels, $x ^ { s } ( \lambda )$ , and its spectral response function $\mathrm { S R F } _ { s }$ , a spectral convolution R is defined as the integral of the product of $x ^ { s }$ and $\mathrm { S R F } _ { s } ,$ , normalized by the integral of its SRF. To generate arbitrary MS channels with unknown SRFs, we model their SRF, $\mathrm { S R F } _ { t }$ as un-normalized Gaussian kernel with a

mean wavelength, $\lambda _ { t } ,$ and standard deviation, $\sigma _ { t } .$ Through this process, we are able to generate novel multi-spectral views from any HS source, extensively expanding the spectral augmentation capabilities of this framework. We hypothesize that this combination may prove to add useful spectral inductive biases to the model.

**Evaluation** To evaluate these design choices, we run the evaluation suite on a model trained according to the specifications laid out in Section 3.5, i.e. identical to the model described in the main paper. We call this model Panopticon-IPE. The results are shown in Tab. 8.

Comparing these results to Tab 2 & 3, we see that Panopticon-IPE is relatively close in performance to Panopticon, and in some cases (TropicalCyclone) even excels. However, on average this model performs worse than our default settings, and as a result, we did not include these methods in the main paper. We leave these findings for the benefit of future researchers.

<!-- page 17 of 20 -->

![Image block](images/page017-block01.png)

Supp. Fig. 3. Cross-sensor invariance (absolute values). In addition to train/val/test splits, we also split datasets across sensors to explicitly test sensor invariance. y-axes on the heatmap represent training sensors and x-axes, test sensors. The diagonals represent same-sensor for train and test, while off-diagonal elements represent cross-sensor results with all values expressed as absolute performance values as percentages. Left: splits are across QuickBird-2 (QB2) / GeoEye-1 (GE1) which are RGB-NIR. Right: splits are across synthesized sensors from the HS EnMAP sensor using spectral convolution - MODIS, Sentinel-2 and Planet SuperDove, for any-sensor models. Bottom: reBEN; splits are implemented across Sentinel-1 and Sentinel-2 senors, for all any-sensor and many-sensor models, except SenPaMAE which cannot process SAR imagery. Panopticon consistently outperforms other models in both settings. Note that the value ranges differ for each sub-figure.

![Chart block](images/page017-block03.png)

![Chart block](images/page017-block04.png)

Supp. Fig. 4. Spectral response function (SRF) fitting for Sentinel-2 (left) and Landsat 9 (right).

## F. Technical details

## F.1. Pre-training

**Implementation of Channel Attention** We implement the initial shared 2D convolution of the PE with a $1 \times p \times p$ 3D convolution [2], where p is the patch size. Note that images within a batch can originate from different sensors in our pipeline and, thus, the number of channels is not consistent within a batch. To efficiently compute the cross attention for batched inputs, we hence employ padding and masking.

**Hyperparameters** We follow most of the DINOv2 configuration with the following changes: We multiply the learning rate of the ViT blocks in the backbone by 0.2, reduce the iBOT loss weight to 0.1, and remove the KoLeo regularizer. We performed early off-the-record ablations on these choices. Apart from that, and for both stages 1 and 2, we train for 70 artificial epochs with 1250 iterations each, with the AdamW optimizer, a base learning rate of 5e-4, standard learning rate scaling lr·bsz/256, and a linear learning rate warmup for 5 epochs followed by a cosine decay to a minimum 1e-6 learning rate.

<!-- page 18 of 20 -->

Table 8. Performance metrics for Panopticon-IPE across various datasets. miJacc = micro Jaccard index

<table><tr><td>Dataset</td><td>Score</td></tr><tr><td colspan="2">Classification (Accuracy %)</td></tr><tr><td>m-eurosat</td><td>96.1</td></tr><tr><td>m-brick-kiln</td><td>95.8</td></tr><tr><td>m-pv4ger-cls</td><td>95.8</td></tr><tr><td>RESISC45</td><td>91.2</td></tr><tr><td>m-forestnet</td><td>54.0</td></tr><tr><td colspan="2">Segmentation (miJacc %)</td></tr><tr><td>m-pv4ger</td><td>95.4</td></tr><tr><td>m-nzcattle</td><td>92.8</td></tr><tr><td>spacenet1</td><td>90.3</td></tr><tr><td>m-neontree</td><td>79.6</td></tr><tr><td>m-chesapeake</td><td>78.0</td></tr><tr><td>m-cashew</td><td>59.1</td></tr><tr><td>m-sacrop</td><td>52.3</td></tr><tr><td colspan="2">Multi-Label Classification (mAP %)</td></tr><tr><td>Corine-MODIS</td><td>80.0</td></tr><tr><td>Corine-SuperDove</td><td>79.8</td></tr><tr><td colspan="2">Regression (MSE)</td></tr><tr><td>DigitalTyphoon</td><td>0.93</td></tr><tr><td>Hyperview-MODIS</td><td>0.35</td></tr><tr><td>Hyperview-SuperDove</td><td>0.35</td></tr><tr><td>TropicalCyclone</td><td>0.28</td></tr></table>

**Metrics** For details on the average metrics defined in the ablations, see Tab. 9.

## F.2. Evaluation

All tasks were executed on either a 40 GB A100 or a 48 GB L40S GPU. The batch size is optimized for each task and model to maximize GPU memory and the base learning rate lr is scaled by the batch size according to the linear scaling rule lr · bsz/256 [18]. In our evaluations, we use the following tasks types.

k**NN** k-nearest neighbors (kNN) with k = 20 and temperature 0.07 following Reed et al. [51].

**Linear probing** We mainly follow the implementation of DINOv2 and sweep multiple extraction methods. In particular, we extract the tokens from the last one or four transformer blocks and aggregate them by concatenating the cls tokens, average pooling, or the default aggregation suggested by the specific model at hand. For each aggregation, we sweep the 13 base learning rates 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, and 10, resulting in 2 · 3 · 13 = 78 runs. Note that we use an extension of the DINOv2 implementation that computes all these different configurations simultaneously from a single forward pass of the backbone. We train for 50 epochs with a 0.9 momentum Stochastic Gradient Descent optimizer. We also add

a trainable batch normalization before the linear head. For the cross-sensor evaluations in Fig. 2 and Fig. 3, we only extract tokens from the last transformer block to ease compatibility issues across models that use different encoders for SAR and optical modalities.

**Linear probing with re-initialized patch embedding** We replace the patch embedding of the model with a randomly initialized 2D convolution layer with the correct number of channels of the dataset at hand. We add a trainable batch norm and linear head to the encoder, unfreeze the new patch embedding and freeze the remaining encoder. We fix the feature aggregation to the default one suggested by the model authors and sweep the base learning rates 0.01, 0.001, 0.0005, and 0.0001 with 50 epochs each, stochastic gradient descent optimizer and 5 epochs of learning rate warmup.

AnySat and Galileo presented unique challenges due to the way they implement modality specific encoders. For AnySat, we pick a specific modality, and retrain the 2D convolution layer of that branch; we picked the NAIP encoder since it does not implement temporal attention, which consumes significant memory and compute. For Galileo, this was not possible since they use a channel grouping where each group produces a set of independent tokens, unlike other models where the channel dimension is collapsed. To replace this, we would have to create a new group with a custom number of channels which would break how Galileo processes tokens in groups. Furthermore, Galileo employs FlexiPatchEmbed, which is trained by randomizing the patch size during pretraining. To properly train this module, we would need to mimic that during evaluation, which was beyond the scope of the evaluation phase of this paper. Therefore, we omitted evaluating Galileo in tests that required retraining the patch embed module for domain adaptation.

**Semantic segmentation** We freeze the backbone and add trainable standard modules from the MMSegmentation library [13]. In particular, we use a Feature2Pyramid as neck, a UPerNet decoder and an auxiliary FCNHead. We sweep the base learning rates 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6 and utilize the AdamW optimizer for 50 epochs with no learning rate warmup.

**Model adaptations** During evaluation, we picked the image size that the model was natively trained on to maximize that model’s ability to produce representations.

## G. Additional ablations

**Spectral augmentation** In line with the DINO ablation on scales in random resized crops, we ablate our spectral

<!-- page 19 of 20 -->

<table><tr><td>Agg.</td><td>Id</td><td>Dataset</td><td>Task details</td><td>Metric</td></tr><tr><td rowspan="2"> $MS_{acc}$ </td><td>1</td><td>m-eurosat</td><td>kNN, k = 20, T = 0.07</td><td>top-1 micro accuracy</td></tr><tr><td>2</td><td>m-eurosat without RGB channels</td><td>kNN, k = 20, T = 0.07</td><td>top-1 micro accuracy</td></tr><tr><td rowspan="2"> $SAR_{acc}$ </td><td>3</td><td>m-eurosat-SAR [68]</td><td>kNN, k = 20, T = 0.07</td><td>top-1 micro accuracy</td></tr><tr><td>4</td><td>m-eurosat-SAR</td><td>linear probing for 10 epochs</td><td>top-1 micro accuracy</td></tr><tr><td rowspan="2"> $Sim_{mAP}$ </td><td>5</td><td>Corine-4</td><td>linear probing on 10% subset for 10 epochs</td><td>top-1 micro multilabel average precision</td></tr><tr><td>6</td><td>Corine-12</td><td>linear probing on 10% subset for 10 epochs</td><td>top-1 micro multilabel average precision</td></tr><tr><td rowspan="3">Avg</td><td>1–6</td><td></td><td></td><td></td></tr><tr><td>7</td><td>RESISC45</td><td>kNN, k = 20, T = 0.07</td><td>top-1 micro accuracy</td></tr><tr><td>8</td><td>m-eurosat only RGB channels</td><td>kNN, k = 20, T = 0.07</td><td>top-1 micro accuracy</td></tr></table>

Table 9. Metrics used to inform design decisions and reported in the ablation section of the main text. The aggregation metric is computed as the average of all its metrics. Corine-n denotes the Corine dataset subsampled to n fixed randomly-selected channels.

|  | m-brick-kiln(S2) | m-eurosat(S2) | m-forestnet(L8) | m-pv4ger(RGB) | m-so2sat(S2) | reBEN(S2) |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2 | 97.5 | 95.5 | 53.5 | 97.5 | 60.8 | 80.1 |
| CROMA | 94.5 | 91.1 | - | - | 53.5 | 79.4 |
| SoftCon | 94.9 | 92.2 | - | - | 52.1 | 80.6 |
| AnySat | 90.3 | 87.6 | 50.9 | 92.8 | 42.5 | 76.8 |
| Galileo | 93.1 | 88.6 | - | - | 54.2 | 76.5 |
| SenPaMAE | 83.9 | 77.5 | 33.5 | 87.1 | 33.7 | 63.8 |
| DOFA | 95.8 | 92.9 | 53.2 | 97.4 | 54.2 | 78.8 |
| Panopticon | 96.7 | 96.4 | 56.3 | 96.4 | 61.7 | 83.9 |

Table 10. Linear probing on GEO-Bench classification datasets and reBEN. We report micro accuracy for single-label and mean average precision for multi-label datasets in percentages.

|  | m-cashew(S2) | m-chesapeake (RGBN) | m-neontree (RGB) | m-nzcattle (RGB) | m-pv4ger(RGB) | m-sacrop(S2) |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2 | 65.9 | 78.5 | 80.9 | 92.7 | 96.9 | 51.2 |
| CROMA | 44.3 | - | - | - | - | 48.4 |
| SoftCon | 54.5 | - | - | - | - | 51.3 |
| AnySat | 38.8 | 75.9 | 79.6 | 92.5 | 92.2 | 39.5 |
| Galileo | 40.4 | - | - | - | - | 39.5 |
| SenPaMAE | 40.7 | 59.9 | 79.5 | 89.5 | 78.3 | 39.3 |
| DOFA | 56.4 | 78.2 | 80.4 | 92.8 | 96.3 | 51.3 |
| Panopticon | 59.3 | 78.1 | 79.6 | 92.6 | 95.2 | 52.6 |

Table 11. GEO-Bench segmentation performance. We report tje micro Jaccard index in percentage computed by torchmetrics.classification.JaccardIndex of the torchmetrics package with version x.y.z [44].

$$
(1, s), (s, 1 3)
$$

$$
s \in \{2, 4, 8, 1 3 \}
$$

see that s = 4 performs best.

<!-- page 20 of 20 -->

|  | m-cashew(S2) | m-chesapeake (RGBN) | m-neontree (RGB) | m-nzcattle (RGB) | m-pv4ger(RGB) | m-sacrop(S2) |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2 | 53.4 | 64.0 | 59.1 | 76.9 | 95.2 | 32.0 |
| CROMA | 34.3 | - | - | - | - | 34.4 |
| SoftCon | 39.9 | - | - | - | - | 33.7 |
| AnySat | 21.8 | 61.7 | 49.9 | 67.6 | 89.2 | 23.6 |
| Galileo | 30.4 | - | - | - | - | 28.4 |
| SenPaMAE | 17.8 | 46.9 | 48.3 | 49.6 | 80.4 | 18.4 |
| DOFA | 41.9 | 59.2 | 55.4 | 74.3 | 94.6 | 32.5 |
| Panopticon | 45.1 | 60.8 | 50.4 | 73.9 | 94.1 | 35.7 |

Table 12. GEO-Bench segmentation performance. We report the macro Jaccard index in percentage computed by torchmetrics.classification.JaccardIndex of the torchmetrics package with version 1.6.1 [44]

|  | m-cashew(S2) | m-chesapeake (RGBN) | m-neontree (RGB) | m-nzcattle (RGB) | m-pv4ger(RGB) | m-sacrop(S2) |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2 | 52.7 | 26.8 | 55.8 | 77.4 | 92.7 | 13.9 |
| CROMA | 33.8 | - | - | - | - | 14.2 |
| SoftCon | 39.2 | - | - | - | - | 14.2 |
| AnySat | 20.9 | 25.5 | 47.4 | 67.9 | 86.0 | 10.0 |
| Galileo | 30.1 | - | - | - | - | 11.9 |
| SenPaMAE | 17.6 | 19.7 | 46.3 | 49.0 | 69.4 | 9.1 |
| DOFA | 41.2 | 26.6 | 51.5 | 74.6 | 89.7 | 13.9 |
| Panopticon | 44.7 | 26.6 | 47.8 | 73.0 | 88.2 | 14.9 |

Table 13. GEO-Bench segmentation performance. We report mIoU averaged over images in percentage computed by torchmetrics.segmentation.MeanIoU of the torchmetrics package with version 1.6.1 [44]

| s | 2 | 4 | 8 | 13 |
| --- | --- | --- | --- | --- |
| MS<sub>acc</sub> | 81.2 | 85.3 | 83.6 | 81.2 |

Table 14. Ablation of non-overlapping spectral crop size.