<!-- page 1 of 14 -->

Scientific Reports

ARTICLE | Article in Press

# Reducing cost of fluorescent microscopy through dye exclusion, distillation techniques, and deep learning

**Adriana Borowa  · Bartosz Zieliński · Dawid Rymarczyk**

Received: 29 November 2024 / Accepted: 27 August 2026

© The Author(s) 2026

## Abstract

High Content Screening (HCS) via the Cell Painting assay is a powerful drug discovery tool, but the requirement for five fluorescent channels significantly increases the cost and complexity of image acquisition. Current deep learning models for HCS are trained and evaluated on all five channels, and it remains unclear whether robust biological representations can be maintained when channels are removed at inference time. This work introduces Channel-Reduced DINO (CR-DINO), a self-distillation method built on the DINO framework that generates rich HCS image representations from a reduced number of channels. CR-DINO employs a teacherstudent architecture in which the teacher retains full five-channel visibility while the student receives progressively fewer channels following a curriculum-inspired schedule. This scheduled reduction, moving from five channels down to two over the course of training, encourages the student to learn cross-channel dependencies rather than relying on information directly available in its input. Experiments on the Bray dataset, validated through Mode of Action prediction, biological activity property prediction, and image–structure retrieval tasks, show that using only two channels (DNA and Mito) provides results on par with those using the full set, and that CR-DINO recovers performance even for the least informative channel pair (ER and AGP). These results highlight the potential of our method to significantly reduce the reagent and acquisition costs of consecutive HCS experiments without sacrificing the depth of biological insights.

## Introduction

## High content screening for drug discovery

High Content Screening (HCS) captures changes in a cell morphology using fluorescent microscopy to advance drug discovery. In recent years, the prevalent protocol for such screenings was Cell Painting , which comprises six dyes marking eight cellular components resulting in 5-channel images. The introduction of such protocol contributed to the standardization of HCS experiments which was further supported by the generation of massive public datasets, e.g. JUMP-CP <sup>2</sup>, RxRx <sup>3</sup>, and Bray <sup>4</sup>. A combination of those factors facilitated the development of increasingly sophisticated deep-learning methods for the analysis of HCS data.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">This Article in Press is shared early to give you faster access to new research. It is citable and carries a permanent DOI. The final edited version will replace it automatically.</span></small>

Check for updates

Scientific Reports

[https://doi.org/10.1038/s41598-026-69424-3](https://doi.org/10.1038/s41598-026-69424-3)

Published online: 09 September 2026

scientific reports

<!-- page 2 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

## Deep learning methods for phenotypic drug discovery

Initially, the main approach to analyzing this data consisted of classical techniques, among others, cell counts or size change measurement 5. Algorithms quantifying cell morphology were encompassed within many proprietary applications, mostly introduced by hardware manufacturers, e.g., Harmony introduced by PerkinElmer (now Revvity) or CellPathFinder (Yokogawa), as well as open-source software, e.g., CellProfiler . Lately, some manufacturers introduced Machine Learning and Deep Learning methods in their software <sup>7</sup>.

As the natural application to HCS images, researchers used Convolutional Neural Networks with supervised tasks such as prediction of Mode of Action <sup>8,9</sup>, biological activity properties <sup>10</sup>, or predefined clusters of phenotypes <sup>11</sup>. A variety of approaches was used to solve those tasks, including single-cell approach <sup>11</sup>, downsampling images <sup>9</sup>, and processing channels separately <sup>8</sup>. In parallel, semi-supervised <sup>12</sup> and unsupervised methods have been developed. Among unsupervised approaches, generative methods included techniques such as paired cell inpainting <sup>13</sup>, variational autoencoders <sup>14</sup> and generative adversarial networks <sup>15</sup>. Self-supervised methods include archetypal analysis <sup>16</sup>, weakly-supervised classification of cells in multiple instance learning setup <sup>17</sup>, and Unsupervised Multi-scale Mode-of-action Discovery <sup>18</sup>. Additionally, auxiliary tasks, often regarded as unsupervised <sup>19</sup>, e.g. compound matching <sup>20</sup>, have been explored.

With the advent of Vision Transformer <sup>21</sup>, which outperformed methods using CNNs, it was subsequently applied to High Content Screening images for learning self-supervised representations <sup>22</sup>. Later, this concept was expanded to carry information about the channel with its token <sup>23</sup>. Eventually, self-distillation with no-labels DINO <sup>24</sup> was applied to HCS images <sup>25,26</sup>, including weak supervision <sup>27</sup> and improvements meant to mitigate batch effects through Cross-Domain Consistency Learning 2

## Reducing number of channels while preserving information

The majority of the aforementioned methods were trained using Cell Painting data, resulting in numerous models tailored to this protocol. However, running this assay may not always be the most cost-effective approach, particularly when the research question can be addressed using a specific channel <sup>29</sup>. We present an estimated cost of each dye in the Cell Painting protocol in Table 1 which shows that the lowest cost is caused by nuclei staining. Moreover, research by Araceli Biosciences has demonstrated that high-throughput screening can be significantly accelerated using a specialized microscope that captures only four channels , which also helps to reduce the overall cost of staining. We question whether it is possible to further reduce the number of channels and if knowledge can be inferred between channels. However, to the best of our knowledge, there have been no previous attempts to reduce the number of channels during the inference of the Deep Learning model trained on Cell Painting data. In other imaging domains, such as MRI <sup>31</sup> and natural image <sup>32,33</sup> analysis, research has already demonstrated that techniques like channel dropout and scheduling force networks to learn more robust representations, i.e., one that maintains downstream task performance despite reduced input information.

Table 1 Estimated percentage of the cost of each dye based on ThermoFisher product catalog [https://www.thermofisher.com/](https://www.thermofisher.com/).

<table><tr><td>Channel</td><td>Dye</td><td>Cost (%)</td></tr><tr><td>0 DNA</td><td>Hoechst 33342</td><td>4</td></tr><tr><td>1 ER</td><td>Concanavalin A</td><td>9</td></tr><tr><td>2 RNA</td><td>SYTO 14</td><td>19</td></tr><tr><td rowspan="2">3 AGP</td><td>Phalloidin (Alexa Fluor 568)</td><td>23</td></tr><tr><td>Wheat Germ Agglutinin</td><td>12</td></tr><tr><td>4 Mito</td><td>MitoTracker Deep Red</td><td>33</td></tr></table>

Two dyes are used for the AGP channel, accounting for 35% of the cost.

scientific reports

Scientific Reports

<!-- page 3 of 14 -->

https://doi.org/10.1038/s41598-026-69424-3

Article in Press

## Scheduled self-distillation for cost-effective HCS

To fill this gap, we propose Channel-Reduced DINO (CR-DINO), a knowledge self-distillation method that generates robust image representation,which encodes relevant biological information. Image representation is a vector obtained by passing an image through Deep Learning model $f _ { i }$

$$
p _ {i} = \text {SoftMax} (f (x _ {i}))\tag{1}
$$

where $p _ { i } \in R ^ { d }$ is a d-dimensional image representation, SoftMax is the softmax function, and $x _ { i }$ is an input image. In CR-DINO, the student network learns to extract information with limited channel access, while the teacher network maintains visibility of all channels. Our method progressively decreases the number of channels used during training, to gently guide the model while it obtains less and less information. Despite utilizing entire Cell Painting images during training, our approach enables the model to operate effectively with fewer channels. This capability allows cost savings in consecutive screening experiments, as fewer dyes are needed, reducing overall expenditure but not the performance of the model as presented in Fig. 1.

We compare this representation to two baselines: the standard DINO training scheme and DINO training with the removal of random channels without any scheduling. We use two standard target tasks to test our representation: Mode of Action <sup>8,9</sup> and biological activity properties <sup>10</sup> prediction using public labels sourced from ChEMBL database <sup>34</sup>. We showcase that up to 3 channels can be removed from the image with no or little drop in the performance. Additionally, we test our representation using a retrieval task where an image is retrieved using chemical structure features and vice versa, again showing that channel removal does not decrease the performance. Our main contributions are as follows:

● We showcase that some channels can be omitted when generating a robust High Content Screening image representation.

● We introduce Channel-Reduced DINO (CR-DINO) that learns dependencies between channels and can lower the cost of consecutive experiments by reducing the number of dyes used to stain cells.

● We demonstrate the efficacy of our representation in scenarios such as retrieval and classification, opening a new research avenue of cost-effective High-Content Screening.

Fig. 1 The cost of screening increases with the number of dyes used in the experiment. Channel-Reduced DINO (CR-DINO) aims to reduce that cost while maintaining high-quality features and prediction performance.

![Image block](images/page003-block12.png)

Scientific Reports

scientific reports

<!-- page 4 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

## Results

## CR-DINO maintains the performance of the model while using a reduced number of channels

We compare a model pre-trained on JUMP-CP dataset (JUMP) as well as three approaches described above and the results are presented in Table 2. Our model trained on the JUMP-CP dataset achieves the lowest results as the additional fine-tuning on the Bray dataset boosts the results. This can be expected as the model is more fitted to the dataset. Other than that, we do not observe significant differences not only between methods but also we do not see a decrease in performance whilst removing consecutive channels. This might suggest that either the majority of the information from the Cell Painting image is carried by the DNA (0) and Mito (4) channels or all channels carry similar information. $\mathbf { E . g . }$ , when analyzed with human eye channels, Mito and ER look very similar.

In Table 3, we present the results of all methods with the backbone of ResNet-50. Once again, we see no significant difference across methods or channel sets. Here, the baseline method was trained on the Bray dataset as well, so we do not see the boost as for the ViT results. The results are lower in comparison to ViT results, showing that Transformer architecture learns a more expressive representation.

## CR-DINO distills information non-directly available at ER and AGP channels.

As there is no significant difference between sets of channels presented in the previous subsection, we test representation generated by all possible sets of two channels using the method All. The results, presented in Table 4, are consistent across all pairs except for one: channels 1 (ER) and 3 (AGP). We can infer from the results that those two channels most probably carry less information and do not have a sufficient overlap with other channels. Furthermore, we test if other fine-tuning methods can distill knowledge to make up for the missing information. Table 5 presents those results. For Mode of Action tasks, the boost of Random and Schedule fine-tuning is small. However, for Property tasks, we see a significant boost when using the Schedule method, which allows us to reach the results of other channel pairs. This shows that using a Schedule fine-tuning helps the network learn connections between channels for at least part of tasks.

Table 2 Mode of action and property prediction tasks using Vision Transformer.

<table><tr><td colspan="2">Fine-tuning</td><td colspan="4">Mode of Action</td><td colspan="4">Property</td></tr><tr><td>Scheme</td><td>Channels</td><td>Avg</td><td>≥ 0.7</td><td>≥ 0.8</td><td>≥ 0.9</td><td>Avg</td><td>≥ 0.7</td><td>≥ 0.8</td><td>≥ 0.9</td></tr><tr><td rowspan="4">None</td><td>all</td><td> $0.52 \pm 0.01\dagger$ </td><td> $4.40 \pm 1.08$ </td><td> $1.60 \pm 0.68$ </td><td> $0.20 \pm 0.20\dagger$ </td><td> $0.54 \pm 0.03$ </td><td> $4.20 \pm 1.59$ </td><td> $2.40 \pm 1.03$ </td><td> $0.80 \pm 0.80$ </td></tr><tr><td>0134</td><td> $0.53 \pm 0.02$ </td><td> $4.20 \pm 1.02$ </td><td> $1.40 \pm 0.68$ </td><td> $0.60 \pm 0.40$ </td><td> $0.56 \pm 0.03$ </td><td> $5.00 \pm 1.52$ </td><td> $1.60 \pm 0.93$ </td><td> $0.80 \pm 0.80$ </td></tr><tr><td>034</td><td> $0.54 \pm 0.02$ </td><td> $4.60 \pm 1.33$ </td><td> $1.80 \pm 0.58$ </td><td> $0.60 \pm 0.40$ </td><td> $0.55 \pm 0.03$ </td><td> $5.20 \pm 1.59$ </td><td> $2.20 \pm 1.07$ </td><td> $1.00 \pm 1.00$ </td></tr><tr><td>04</td><td> $0.53 \pm 0.01\dagger$ </td><td> $4.60 \pm 1.17$ </td><td> $1.80 \pm 0.73$ </td><td> $0.60 \pm 0.40$ </td><td> $0.56 \pm 0.03$ </td><td> $5.00 \pm 1.52$ </td><td> $2.60 \pm 1.21$ </td><td> $1.20 \pm 0.97$ </td></tr><tr><td rowspan="4">All</td><td>all</td><td> $0.57 \pm 0.02$ </td><td> $5.20 \pm 1.32$ </td><td> $2.80 \pm 0.92$ </td><td> $0.80 \pm 0.37$ </td><td> $0.58 \pm 0.02$ </td><td> $4.80 \pm 1.07$ </td><td> $3.00 \pm 1.05$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>0134</td><td> $0.57 \pm 0.02$ </td><td> $5.20 \pm 1.32$ </td><td> $2.60 \pm 0.98$ </td><td> $0.80 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $4.80 \pm 1.07$ </td><td> $3.00 \pm 1.05$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>034</td><td> $0.57 \pm 0.02$ </td><td> $5.20 \pm 1.32$ </td><td> $2.60 \pm 0.98$ </td><td> $0.80 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $4.80 \pm 1.07$ </td><td> $2.80 \pm 1.16$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>04</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.47$ </td><td> $2.80 \pm 0.92$ </td><td> $0.80 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $4.60 \pm 0.98$ </td><td> $2.80 \pm 0.97$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td rowspan="4">Random</td><td>all</td><td> $0.57 \pm 0.02$ </td><td> $6.20 \pm 1.50$ </td><td> $3.00 \pm 0.95$ </td><td> $1.20 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $5.60 \pm 1.17$ </td><td> $3.20 \pm 1.16$ </td><td> $1.40 \pm 0.75$ </td></tr><tr><td>0134</td><td> $0.57 \pm 0.02$ </td><td> $6.20 \pm 1.50$ </td><td> $3.20 \pm 0.86$ </td><td> $1.20 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.29$ </td><td> $3.40 \pm 1.03$ </td><td> $1.40 \pm 0.75$ </td></tr><tr><td>034</td><td> $0.58 \pm 0.02$ </td><td> $6.40 \pm 1.44$ </td><td> $3.40 \pm 0.87$ </td><td> $1.20 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.44$ </td><td> $3.40 \pm 1.03$ </td><td> $1.20 \pm 0.80$ </td></tr><tr><td>04</td><td> $0.58 \pm 0.02$ </td><td> $6.20 \pm 1.50$ </td><td> $3.20 \pm 0.86$ </td><td> $1.20 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.03$ </td><td> $3.20 \pm 1.16$ </td><td> $1.40 \pm 0.75$ </td></tr><tr><td rowspan="4">Schedule</td><td>all</td><td> $0.57 \pm 0.02$ </td><td> $5.60 \pm 1.36$ </td><td> $2.40 \pm 0.75$ </td><td> $1.20 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $4.20 \pm 1.11$ </td><td> $2.40 \pm 1.03$ </td><td> $1.00 \pm 0.77$ </td></tr><tr><td>0134</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.29$ </td><td> $2.60 \pm 0.75$ </td><td> $1.20 \pm 0.37$ </td><td> $0.56 \pm 0.02$ </td><td> $4.20 \pm 1.11$ </td><td> $2.40 \pm 1.03$ </td><td> $1.00 \pm 0.77$ </td></tr><tr><td>034</td><td> $0.57 \pm 0.02$ </td><td> $5.60 \pm 1.36$ </td><td> $2.60 \pm 0.81$ </td><td> $1.00 \pm 0.32$ </td><td> $0.57 \pm 0.02$ </td><td> $4.60 \pm 1.33$ </td><td> $2.60 \pm 1.08$ </td><td> $1.20 \pm 0.80$ </td></tr><tr><td>04</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.29$ </td><td> $2.60 \pm 0.75$ </td><td> $1.20 \pm 0.37$ </td><td> $0.56 \pm 0.02$ </td><td> $4.20 \pm 1.11$ </td><td> $2.40 \pm 1.03$ </td><td> $1.00 \pm 0.77$ </td></tr></table>

ViT obtained using DINO trained on JUMP-CP dataset (None) achieves the worst results. However, when it is fine-tuned to the Bray dataset (remaining fine-tuning schemes) the results are improved, especially for Random approach. Removing channels does not decrease the performance across all methods. We compare all methods to the results of the model fine-tuned using All scheme with representation generated using all channels. Red annotates higher values, while blue lower values than the average. † denotes values significantly lower according to Z-score analysis with 95% confidence interval.

scientific reports

Scientific Reports

<!-- page 5 of 14 -->

https://doi.org/10.1038/s41598-026-69424-3

Article in Press

Table 3 Mode of action and property prediction tasks using ResNet.

<table><tr><td colspan="2">Fine-tuning</td><td colspan="4">Mode of Action</td><td colspan="4">Property</td></tr><tr><td>Scheme</td><td>Channels</td><td>Avg</td><td>≥0.7</td><td>≥0.8</td><td>≥0.9</td><td>Avg</td><td>≥0.7</td><td>≥0.8</td><td>≥0.9</td></tr><tr><td rowspan="4">None</td><td>all</td><td> $0.54 \pm 0.02$ </td><td> $5.20 \pm 1.46$ </td><td> $2.20 \pm 0.49$ </td><td> $0.40 \pm 0.24$ </td><td> $0.56 \pm 0.02$ </td><td> $4.00 \pm 0.89$ </td><td> $2.60 \pm 0.81$ </td><td> $1.40 \pm 0.68$ </td></tr><tr><td>0134</td><td> $0.54 \pm 0.02$ </td><td> $4.60 \pm 1.36$ </td><td> $2.20 \pm 0.58$ </td><td> $0.20 \pm 0.20$ </td><td> $0.56 \pm 0.02$ </td><td> $4.00 \pm 1.14$ </td><td> $2.40 \pm 1.12$ </td><td> $1.20 \pm 0.73$ </td></tr><tr><td>034</td><td> $0.53 \pm 0.02$ </td><td> $4.60 \pm 1.12$ </td><td> $1.80 \pm 0.58$ </td><td> $0.60 \pm 0.40$ </td><td> $0.56 \pm 0.02$ </td><td> $3.80 \pm 1.16$ </td><td> $2.20 \pm 0.97$ </td><td> $0.80 \pm 0.58$ </td></tr><tr><td>04</td><td> $0.54 \pm 0.01$ </td><td> $4.60 \pm 1.12$ </td><td> $2.00 \pm 0.55$ </td><td> $0.60 \pm 0.24$ </td><td> $0.56 \pm 0.02$ </td><td> $4.20 \pm 1.20$ </td><td> $2.60 \pm 0.81$ </td><td> $0.60 \pm 0.40$ </td></tr><tr><td rowspan="4">All</td><td>all</td><td> $0.56 \pm 0.02$ </td><td> $4.80 \pm 1.16$ </td><td> $1.80 \pm 0.86$ </td><td> $0.80 \pm 0.20$ </td><td> $0.58 \pm 0.02$ </td><td> $5.00 \pm 1.00$ </td><td> $2.80 \pm 1.07$ </td><td> $2.00 \pm 1.05$ </td></tr><tr><td>0134</td><td> $0.54 \pm 0.02$ </td><td> $4.80 \pm 1.32$ </td><td> $2.60 \pm 1.17$ </td><td> $1.00 \pm 0.55$ </td><td> $0.56 \pm 0.02$ </td><td> $4.80 \pm 1.39$ </td><td> $2.20 \pm 1.46$ </td><td> $1.40 \pm 0.68$ </td></tr><tr><td>034</td><td> $0.54 \pm 0.02$ </td><td> $4.60 \pm 0.98$ </td><td> $1.80 \pm 0.58$ </td><td> $0.60 \pm 0.40$ </td><td> $0.56 \pm 0.02$ </td><td> $4.40 \pm 1.36$ </td><td> $1.80 \pm 1.07$ </td><td> $1.40 \pm 0.68$ </td></tr><tr><td>04</td><td> $0.53 \pm 0.03$ </td><td> $4.60 \pm 1.21$ </td><td> $1.40 \pm 0.75$ </td><td> $0.60 \pm 0.40$ </td><td> $0.55 \pm 0.02$ </td><td> $4.40 \pm 1.50$ </td><td> $2.00 \pm 0.84$ </td><td> $1.20 \pm 0.58$ </td></tr><tr><td rowspan="4">Random</td><td>all</td><td> $0.52 \pm 0.01\dagger$ </td><td> $4.40 \pm 0.81$ </td><td> $1.60 \pm 0.51$ </td><td> $0.40 \pm 0.24$ </td><td> $0.54 \pm 0.02$ </td><td> $4.60 \pm 1.21$ </td><td> $2.00 \pm 1.30$ </td><td> $0.60 \pm 0.60$ </td></tr><tr><td>0134</td><td> $0.53 \pm 0.02$ </td><td> $3.80 \pm 0.97$ </td><td> $2.20 \pm 0.73$ </td><td> $0.40 \pm 0.24$ </td><td> $0.54 \pm 0.01$ </td><td> $3.80 \pm 1.16$ </td><td> $2.00 \pm 1.14$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>034</td><td> $0.54 \pm 0.02$ </td><td> $4.20 \pm 1.16$ </td><td> $2.00 \pm 0.45$ </td><td> $0.40 \pm 0.24$ </td><td> $0.55 \pm 0.03$ </td><td> $4.60 \pm 1.36$ </td><td> $2.60 \pm 1.47$ </td><td> $1.20 \pm 0.97$ </td></tr><tr><td>04</td><td> $0.53 \pm 0.02$ </td><td> $4.00 \pm 1.00$ </td><td> $1.00 \pm 0.55$ </td><td> $0.20 \pm 0.20$ </td><td> $0.56 \pm 0.02$ </td><td> $4.40 \pm 1.63$ </td><td> $1.80 \pm 0.58$ </td><td> $0.80 \pm 0.37$ </td></tr><tr><td rowspan="4">Schedule</td><td>all</td><td> $0.54 \pm 0.02$ </td><td> $4.00 \pm 1.29$ </td><td> $1.25 \pm 0.25$ </td><td> $0.50 \pm 0.29$ </td><td> $0.56 \pm 0.01$ </td><td> $4.50 \pm 1.26$ </td><td> $2.00 \pm 0.41$ </td><td> $1.00 \pm 0.00$ </td></tr><tr><td>0134</td><td> $0.55 \pm 0.02$ </td><td> $5.60 \pm 1.44$ </td><td> $2.40 \pm 0.93$ </td><td> $0.80 \pm 0.58$ </td><td> $0.56 \pm 0.02$ </td><td> $4.80 \pm 0.97$ </td><td> $2.00 \pm 1.05$ </td><td> $1.40 \pm 0.68$ </td></tr><tr><td>034</td><td> $0.54 \pm 0.02$ </td><td> $4.60 \pm 1.29$ </td><td> $1.80 \pm 0.66$ </td><td> $0.60 \pm 0.40$ </td><td> $0.55 \pm 0.02$ </td><td> $4.40 \pm 1.12$ </td><td> $1.80 \pm 0.86$ </td><td> $1.40 \pm 0.68$ </td></tr><tr><td>04</td><td> $0.53 \pm 0.02$ </td><td> $4.40 \pm 0.87$ </td><td> $1.60 \pm 0.51$ </td><td> $0.60 \pm 0.40$ </td><td> $0.56 \pm 0.02$ </td><td> $4.80 \pm 1.32$ </td><td> $2.00 \pm 0.84$ </td><td> $1.20 \pm 0.73$ </td></tr></table>

Removing the channel does not decrease the performance of ResNet-based representation, but the results are worse than those obtained by Vision Transformer. We compare all methods to the results of the model fine-tuned using All scheme with representation generated using all channels. Red annotates higher than the average values. † denotes values significantly lower according to Z-score analysis with 95% confidence interval.

Table 4 Mode of action and property prediction tasks using 2 channels and Vision Transformer.

<table><tr><td rowspan="2">Channels</td><td colspan="4">Mode of Action</td><td colspan="4">Property</td></tr><tr><td>Avg</td><td>≥ 0.7</td><td>≥ 0.8</td><td>≥ 0.9</td><td>Avg</td><td>≥ 0.7</td><td>≥ 0.8</td><td>≥ 0.9</td></tr><tr><td>all</td><td> $0.57 \pm 0.02$ </td><td> $5.20 \pm 1.32$ </td><td> $2.80 \pm 0.92$ </td><td> $0.80 \pm 0.37$ </td><td> $0.58 \pm 0.02$ </td><td> $4.80 \pm 1.07$ </td><td> $3.00 \pm 1.05$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>01</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.36$ </td><td> $2.60 \pm 0.81$ </td><td> $1.20 \pm 0.20$ </td><td> $0.57 \pm 0.02$ </td><td> $4.80 \pm 0.97$ </td><td> $3.20 \pm 0.86$ </td><td> $1.00 \pm 0.77$ </td></tr><tr><td>02</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.36$ </td><td> $2.40 \pm 0.75$ </td><td> $1.40 \pm 0.40$ </td><td> $0.57 \pm 0.02$ </td><td> $4.80 \pm 0.97$ </td><td> $2.60 \pm 0.93$ </td><td> $1.20 \pm 0.73$ </td></tr><tr><td>03</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.47$ </td><td> $2.60 \pm 0.81$ </td><td> $1.00 \pm 0.45$ </td><td> $0.58 \pm 0.02$ </td><td> $4.80 \pm 0.97$ </td><td> $3.20 \pm 0.97$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>04</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.47$ </td><td> $2.80 \pm 0.92$ </td><td> $0.80 \pm 0.37$ </td><td> $0.57 \pm 0.02$ </td><td> $4.60 \pm 0.98$ </td><td> $2.80 \pm 0.97$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>12</td><td> $0.56 \pm 0.02$ </td><td> $5.40 \pm 1.50$ </td><td> $2.80 \pm 0.58$ </td><td> $1.40 \pm 0.51$ </td><td> $0.57 \pm 0.02$ </td><td> $4.80 \pm 1.11$ </td><td> $2.60 \pm 0.93$ </td><td> $0.80 \pm 0.58$ </td></tr><tr><td>13</td><td> $0.52 \pm 0.02\dagger$ </td><td> $3.00 \pm 0.55$ </td><td> $1.20 \pm 0.49$ </td><td> $0.40 \pm 0.24\dagger$ </td><td> $0.53 \pm 0.02$ </td><td> $3.60 \pm 1.08$ </td><td> $1.00 \pm 0.77$ </td><td> $0.80 \pm 0.80$ </td></tr><tr><td>14</td><td> $0.57 \pm 0.02$ </td><td> $4.60 \pm 1.12$ </td><td> $3.00 \pm 0.71$ </td><td> $1.40 \pm 0.51$ </td><td> $0.58 \pm 0.02$ </td><td> $5.20 \pm 1.24$ </td><td> $3.20 \pm 1.20$ </td><td> $1.00 \pm 0.55$ </td></tr><tr><td>23</td><td> $0.56 \pm 0.02$ </td><td> $4.60 \pm 1.17$ </td><td> $2.20 \pm 0.73$ </td><td> $1.20 \pm 0.49$ </td><td> $0.56 \pm 0.02$ </td><td> $4.20 \pm 1.16$ </td><td> $2.60 \pm 0.87$ </td><td> $1.00 \pm 0.77$ </td></tr><tr><td>24</td><td> $0.57 \pm 0.02$ </td><td> $5.40 \pm 1.44$ </td><td> $2.40 \pm 0.75$ </td><td> $1.20 \pm 0.20$ </td><td> $0.57 \pm 0.02$ </td><td> $4.60 \pm 1.03$ </td><td> $2.80 \pm 1.07$ </td><td> $1.20 \pm 0.73$ </td></tr><tr><td>34</td><td> $0.58 \pm 0.02$ </td><td> $5.20 \pm 1.24$ </td><td> $2.80 \pm 0.66$ </td><td> $1.00 \pm 0.32$ </td><td> $0.57 \pm 0.02$ </td><td> $5.00 \pm 1.14$ </td><td> $2.80 \pm 1.20$ </td><td> $0.80 \pm 0.58$ </td></tr></table>

Removing channels 0, 2, 4 decreases the performance significantly. We compare all methods to the results of the model fine-tuned using All scheme with representation generated using all channels. Red annotates higher values, while blue lower values than the average. † denotes values significantly lower according to Z-score analysis with 95% confidence interval..

Table 5 Mode of action and property prediction tasks for channels 1 and 3 using Vision Transformer.

<table><tr><td rowspan="2">Fine-tuning scheme</td><td colspan="4">Mode of Action</td><td colspan="4">Property</td></tr><tr><td>Avg</td><td>≥ 0.7</td><td>≥ 0.8</td><td>≥ 0.9</td><td>Avg</td><td>≥ 0.7</td><td>≥ 0.8</td><td>≥ 0.9</td></tr><tr><td>All</td><td> $0.52 \pm 0.02$ </td><td> $3.00 \pm 0.55$ </td><td> $1.20 \pm 0.49$ </td><td> $0.40 \pm 0.24$ </td><td> $0.53 \pm 0.02$ </td><td> $3.60 \pm 1.08$ </td><td> $1.00 \pm 0.77$ </td><td> $0.80 \pm 0.80$ </td></tr><tr><td>Random</td><td> $0.53 \pm 0.01$ </td><td> $4.20 \pm 0.66$ </td><td> $2.00 \pm 0.55$ </td><td> $0.80 \pm 0.37$ </td><td> $0.53 \pm 0.02$ </td><td> $3.80 \pm 1.02$ </td><td> $1.60 \pm 0.87$ </td><td> $0.60 \pm 0.40$ </td></tr><tr><td>Schedule</td><td> $0.54 \pm 0.02$ </td><td> $4.60 \pm 1.36$ </td><td> $2.20 \pm 0.58$ </td><td> $0.40 \pm 0.24$ </td><td> $0.57 \pm 0.03$ </td><td> $5.40 \pm 1.17$ </td><td> $2.60 \pm 0.93$ </td><td> $1.00 \pm 0.77$ </td></tr></table>

CR-DINO (Schedule) improves the performance of representation obtained from 2 worse performing channels. Here, red annotates higher values than fine-tuning method All and blue lower ones.

Scientific Reports

scientific reports

<!-- page 6 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

## CR-DINO is successful in other tasks

The retrieval task results are presented in the Table 6. We can expect a drop in performance compared to CLOOME <sup>35</sup> as it was trained to create aligned image and structure representations, and other methods were fine-tuned. We can observe that in the structure retrieval Top-1 metric, however, for Top-5 and Top-10 retrieved structures this advantage disappears. For image retrieval, we do not observe any difference between methods. In summary, fewer channels can be used for task retrieval without a significant drop in performance.

## Discussion

In this study, we introduce Channel-Reduced DINO, a method for generating representations of High Content Screening images using two to five channels. This capability allows for cost-effective HCS experiments by minimizing the dyes needed.

Results show that up to three channels can be reduced without a decrease in performance, which was evaluated using Mode of Action and property prediction tasks. We compared CR-DINO to two baselines where all and random channels were used, as well as to a representation model trained on the JUMP-CP dataset. Fine-tuning on the Bray dataset improved the ROC AUC by an average of 0.04, and up to two more tasks achieved ROC AUC over 0.7. Moreover, we show that Vision Transformers generate a more robust representation as results obtained by the ResNet network were lower across all methods. We acknowledge that the ViT and ResNet architectures produce representations of different dimensionality (768 and 512, respectively). However, comparing a small number of architectures with their native embedding sizes is standard practice in this research area $^ { 2 7 , 3 6 , 3 7 }$ Our primary objective is to evaluate channel reduction within each architecture rather than to rank architectures against each other. The consistent observation that channel reduction does not degrade performance holds across both architectures, strengthening the generality of our findings.

While the absolute AUC scores (0.52–0.58 range) may appear modest, they are consistent with results reported in prior work on comparable datasets and evaluation protocols. Previous works <sup>38</sup> report similar performance

Table 6 Retrieval task results (ResNet).

<table><tr><td rowspan="2">Fine-tuning scheme</td><td rowspan="2">Channels</td><td colspan="3">Structure</td><td colspan="3">Image</td></tr><tr><td>Top-1 ↑</td><td>Top-5 ↑</td><td>Top-10 ↑</td><td>Top-1 ↑</td><td>Top-5 ↑</td><td>Top-10 ↑</td></tr><tr><td rowspan="4">None</td><td>all</td><td> $5.69 \pm 0.34*$ </td><td> $8.35 \pm 0.69$ </td><td> $10.23 \pm 0.77$ </td><td> $1.22 \pm 0.04$ </td><td> $5.05 \pm 0.13$ </td><td> $9.79 \pm 0.14$ </td></tr><tr><td>0134</td><td> $5.43 \pm 0.37$ </td><td> $7.96 \pm 0.65$ </td><td> $9.73 \pm 0.68$ </td><td> $1.07 \pm 0.09$ </td><td> $5.17 \pm 0.23$ </td><td> $9.80 \pm 0.25$ </td></tr><tr><td>034</td><td> $6.08 \pm 0.30*$ </td><td> $8.40 \pm 0.61$ </td><td> $10.13 \pm 0.76$ </td><td> $0.96 \pm 0.10$ </td><td> $5.31 \pm 0.27$ </td><td> $10.10 \pm 0.18$ </td></tr><tr><td>04</td><td> $5.54 \pm 0.19*$ </td><td> $7.87 \pm 0.49$ </td><td> $9.65 \pm 0.76$ </td><td> $1.16 \pm 0.10$ </td><td> $5.19 \pm 0.21$ </td><td> $10.12 \pm 0.31$ </td></tr><tr><td rowspan="4">All</td><td>all</td><td> $4.08 \pm 0.21\dagger$ </td><td> $7.26 \pm 0.57$ </td><td> $10.37 \pm 0.61$ </td><td> $1.14 \pm 0.10$ </td><td> $4.96 \pm 0.21$ </td><td> $9.96 \pm 0.21$ </td></tr><tr><td>0134</td><td> $4.17 \pm 0.49$ </td><td> $7.83 \pm 0.60$ </td><td> $10.89 \pm 0.58$ </td><td> $1.20 \pm 0.06$ </td><td> $4.86 \pm 0.19$ </td><td> $9.74 \pm 0.17$ </td></tr><tr><td>034</td><td> $4.61 \pm 0.41$ </td><td> $8.09 \pm 0.61$ </td><td> $11.34 \pm 0.52$ </td><td> $1.07 \pm 0.11$ </td><td> $4.65 \pm 0.12\dagger$ </td><td> $10.04 \pm 0.27$ </td></tr><tr><td>04</td><td> $4.57 \pm 0.64$ </td><td> $7.90 \pm 0.84$ </td><td> $11.11 \pm 0.75$ </td><td> $1.20 \pm 0.11$ </td><td> $5.34 \pm 0.32$ </td><td> $9.72 \pm 0.26$ </td></tr><tr><td rowspan="4">Random</td><td>all</td><td> $5.52 \pm 0.40$ </td><td> $8.08 \pm 0.51$ </td><td> $10.16 \pm 0.40$ </td><td> $1.08 \pm 0.11$ </td><td> $4.90 \pm 0.13$ </td><td> $9.64 \pm 0.30$ </td></tr><tr><td>0134</td><td> $4.74 \pm 0.45$ </td><td> $7.72 \pm 0.52$ </td><td> $10.18 \pm 0.27$ </td><td> $1.03 \pm 0.10$ </td><td> $4.92 \pm 0.09$ </td><td> $9.74 \pm 0.36$ </td></tr><tr><td>034</td><td> $4.69 \pm 0.38$ </td><td> $7.78 \pm 0.60$ </td><td> $10.25 \pm 0.55$ </td><td> $0.85 \pm 0.09\dagger$ </td><td> $5.16 \pm 0.18$ </td><td> $9.86 \pm 0.27$ </td></tr><tr><td>04</td><td> $5.43 \pm 0.49$ </td><td> $8.92 \pm 0.49$ </td><td> $11.16 \pm 0.76$ </td><td> $1.06 \pm 0.10$ </td><td> $4.79 \pm 0.06$ </td><td> $9.81 \pm 0.19$ </td></tr><tr><td rowspan="4">Schedule</td><td>all</td><td> $4.16 \pm 0.23$ </td><td> $7.84 \pm 0.51$ </td><td> $10.80 \pm 0.79$ </td><td> $0.91 \pm 0.13$ </td><td> $5.05 \pm 0.34$ </td><td> $9.99 \pm 0.10$ </td></tr><tr><td>0134</td><td> $4.33 \pm 0.37$ </td><td> $7.76 \pm 0.56$ </td><td> $10.76 \pm 0.70$ </td><td> $1.11 \pm 0.07$ </td><td> $4.73 \pm 0.11$ </td><td> $9.80 \pm 0.21$ </td></tr><tr><td>034</td><td> $4.48 \pm 0.57$ </td><td> $7.95 \pm 0.70$ </td><td> $10.90 \pm 0.66$ </td><td> $1.14 \pm 0.18$ </td><td> $5.02 \pm 0.20$ </td><td> $9.69 \pm 0.25$ </td></tr><tr><td>04</td><td> $4.17 \pm 0.43$ </td><td> $7.80 \pm 0.69$ </td><td> $10.75 \pm 0.72$ </td><td> $1.26 \pm 0.14$ </td><td> $4.92 \pm 0.10$ </td><td> $10.10 \pm 0.13$ </td></tr></table>

While there is a drop in performance for top-1 structure retrieval, other metrics do not record a decrease while using our method (Schedule) and baselines (All, Random). We compare all methods to the results of the model fine-tuned using All scheme with representation generated using all channels. Red annotates higher values, while blue lower values than the average. ∗ denotes values significantly higher and † significantly lower according to Z-score analysis with 95% confidence interval.

scientific reports

Scientific Reports

<!-- page 7 of 14 -->

https://doi.org/10.1038/s41598-026-69424-3

Article in Press

levels on overlapping tasks using the same dataset, and established evaluation frameworks <sup>10,39</sup> demonstrate that multi-task prediction over heterogeneous biological activity labels inherently yields lower average AUC values. This is partly because some perturbations do not produce detectable phenotypic changes, and the corresponding tasks effectively dilute the average. To account for this, we also report the number of tasks exceeding specific AUC thresholds (0.7, 0.8, 0.9), following the standard practice established in prior work. Importantly, for assessing the effect of channel reduction, the relative comparison between channel configurations is more informative than the absolute performance level. Moreover, we report results for tasks exceeding specific AUC thresholds rather than analyzing the full distribution, as tasks with AUC near or below 0.5 likely represent perturbations that do not induce detectable phenotypic changes and would typically be discarded in practical screening applications.

We tested the limits of the methods by evaluating the representation obtained from all possible combinations of two channels. A combination of two channels, in particular, ER and AGP, achieved the lowest result when using a baseline method to generate the representation. Our method, CR-DINO, removes this limitation, and we show that using a Schedule fine-tuning scheme boosts these results, for the property prediction task, this channel combination obtained results on par with the rest of the pairs. Furthermore, we tested the representation using image and structure retrieval tasks. Once again, up to three channels can be removed without influencing the performance.

As we used public labels, our study has limitations, particularly in the number of tasks used for evaluation. Specifically, the publicly available labels from the ChEMBL database may not encompass the full range of tasks and complexities present in real-world biological scenarios. The limited number of tasks and the potentially sparse or uneven distribution of labels in these datasets could restrict the generalizability of our findings. To fully study the efficacy of representation generated with CR-DINO, it is crucial to evaluate it across a broader spectrum of tasks, each with more densely populated and varied labels. Such comprehensive evaluation might reveal specific tasks where performance could be enhanced or compromised by different channel configurations.

## Methods

## Materials

High Content Screening (HCS) is a method of capturing phenotypic changes in cells influenced, i.e., by chemical or genetic perturbation, and is commonly used in the Drug Discovery process. Data is generated by inducing cells with a perturbation and marking them with fluorescent dyes that attach to different parts of the cell. Images considered in this work were generated using Cell Painting protocol <sup>1</sup> and are using the following 5 channels with organelle or cellular component in parentheses (their numbers will be used to describe channel setup in experiments):

0 DNA (Nucleus)

1 ER (Endoplasmic reticulum)

2 RNA (Nucleoli, cytoplasmic RNA)

3 AGP (F-actin cytoskeleton, Golgi, plasma membrane)

4 Mito (Mitochondria)

## JUMP-CP dataset

JUMP-Cell Painting consortium, consisting of ten pharmaceutical companies and two non-profits, generated a first-of-its-kind massive HCS dataset <sup>2</sup>. The dataset was built using U2OS cells and around 120,000 compounds, which were shared among companies to create multiple replicates, 13,000 overexpressed genes, and 8,000 genes knocked down by CRISPR-Cas9. While we do not explicitly use this dataset in our work, it was utilized to pretrain

Scientific Reports

scientific reports

<!-- page 8 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

a Vision Transformer, which served as the foundation for DINO training as we used DINO4cells weights <sup>25</sup> to initialize our models.

## Bray dataset

The Bray dataset <sup>4</sup> contains images of U2OS cells perturbed with around 30,000 small molecules. It was also generated using the Cell Painting protocol. This dataset was first split into the distillation and evaluation datasets. The evaluation dataset contains all compounds that have a corresponding label in ChEMBL database <sup>34</sup> to describe Modes of Action (MoA, e.g. Vanilloid receptor, Inositol phosphatase, Arachidonate lipoxygenase) and properties (e.g. CHEMBL203: Epidermal growth factor receptor erbB1, CHEMBL205: Carbonic anhydrase II, CHEMBL206: Estrogen receptor alpha) of compounds. Here, we use compounds that have at least 25 positive and 25 negative labels to allow for further test/train split. The rest of the compounds that do not have any labels are used to train a representation model through a distillation method. The distillation dataset contains 18570 compounds and the evaluation dataset contains 12043 compounds. The evaluation dataset was further split into train/test datasets using 5-fold splits stratified for compound structure to avoid data leaks by testing on structures similar to the ones used in the training. Refer to<sup>40</sup> for detailed numbers of this split and list of Mode of Actions and properties.

## Channels

We arbitrarily set an order to channel importance starting with the DNA channel, which presents a nucleus of the cell and is used in the majority of screenings for its usefulness in assessing cell count or cell cycle state <sup>41</sup>. Next, we select the Mito channel displaying mitochondria whose morphology is related to the cell health <sup>42,43</sup> and AGP channel, which contains information about three cellular components (such as actin cytoskeleton, plasma membrane, and Golgi apparatus) that reflect cellular shape. The last two channels are ER and RNA, due to their limited contribution to the morphological and phenotypic characterization of cells, as well as substantial overlap in their wavelength spectrum . The conclusion that the ER and AGP channels carry less information is supported by the results in Table 4, which show significantly lower performance when only these two channels (channels 1 and 3) are used. Running experiments for all possible channel orderings would be computationally prohibitive, so we selected a biologically motivated ordering based on established understanding of each channel’s contribution to cellular morphology characterization, supplemented by empirical validation from our two-channel experiments.

We do not evaluate single-channel configurations in our experiments. Preliminary tests using only the DNA channel (nucleus) showed substantially degraded performance, as this single channel cannot capture the cytoplasmic and organellar morphology that carries much of the biologically relevant information. Furthermore, the DNA dye (Hoechst 33342) accounts for only approximately 4% of the total reagent cost (Table 1), offering negligible savings from its exclusion. We therefore focus our analysis on the more practically meaningful reduction from five to two channels.

## Channel-reduced distillation

We aim to train a model capable of generating an effective representation while not using all channels of a Cell Painting image. We propose a self-distillation method wherein the student network is trained to distill information without direct access to all channels while the teacher network retains full-channel visibility. Our approach involves a gradual reduction in the number of channels utilized during training, beginning with all five channels and reducing their number with time down to two. Our models are validated through the downstream prediction and retrieval tasks detailed in the Evaluation of Representation section. Our analysis focuses on comparing channel configurations within the DINO framework to evaluate internal scaling properties. We note that alternative

scientific reports

Scientific Reports

<!-- page 9 of 14 -->

https://doi.org/10.1038/s41598-026-69424-3

Article in Press

architectures might yield different absolute performance; however, the consistent trends across both ViT and ResNet architectures (Table 2, 3) support the generality of our channel reduction findings.

We follow a training regime of self-distillation with no labels $( \mathrm { D I N O } ) ^ { 2 4 }$ presented in Fig. 2. Firstly, image x is augmented, creating two views $x _ { 1 }$ and $x _ { 2 } ,$ which are then passed through the student and teacher networks, respectively. The output of the teacher network is centered, i.e., the exponential moving average of the teacher outputs is subtracted, preventing representation collapse to a uniform distribution <sup>24</sup>. Next, softmax is applied to both outputs to obtain an image representation. Finally, representations are passed to DINO loss which aims to align them. Both networks have the same architecture, however the gradient is passed only through the student network and the teacher is updated using Exponential Moving Average (EMA). The augmentation of the image includes using local and global views. The local view is a crop of size 96 × 96 pixels, and we use eight of those crops, while the global crop is of size 224 × 224 pixels, and there are two global views. All views are passed through the student network, and only global views are passed through the teacher network. Our method introduces additional augmentation of the channel selection to the student network.

In line with established methodologies in self-supervised learning <sup>24,44</sup>, the student and teacher networks share an identical architecture to ensure feature alignment. While the optimal depth can in principle be determined empirically by identifying the point of diminishing returns, we follow established practice <sup>45–47</sup> and maintain identical architectures for teacher and student throughout our experiments. Since the architectures are shared, the total number of trainable parameters remains constant across all channel configurations. The computational difference arising from processing fewer input channels affects only the initial embedding layers and amounts to less than 1% variation in FLOPs. This design choice ensures that observed performance differences are attributable to the information content of the input channels, not to differences in model capacity.

We propose to alter the method by removing part of the information from the student network, which here is information about selected channels. We compare three fine-tuning methods depicted in the Fig. 2:

● All: All channels are used in the training. This is the basic DINO training.

![Image block](images/page009-block08.png)

Fig. 2 The overview of the method. We follow a well-established DINO framework with three possible approaches to the channel selection for the student network: All (all channels are used), Random (channels are sampled randomly), Schedule (number of sampled channels is decreasing from 5 to 2 channels during training). $p _ { 1 }$ and $p _ { 2 }$ denote the softmax-normalized representations (Eq. 1) of the student and teacher outputs, respectively, EMA - Exponential Moving Average, sg - stop gradient..

Scientific Reports

scientific reports

<!-- page 10 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

Fig. 3 Evaluation of representation. Firstly, the representation of an image is generated by passing selected channels through the teacher network. To evaluate the retrieval task, representations of images and structures are compared to find the closest ones. To evaluate the prediction task, an MLP is trained and then tested. The MLP model transitions from the input features through hidden layers of sizes 1536, 512, 256, and 128. ReLU activation and dropout (p = 0.2) are applied after each hidden transformation to enhance generalization.

![Image block](images/page010-block04.png)

● Random: Channels are randomly selected during the training. The student network sees only those channels, and the teacher sees all channels. The number of randomly selected channels is also randomized.

● Schedule: Channels are selected using a schedule. We start with all channels and then reduce the number of channels every quarter of training epochs, meaning that in the last set of epochs, we use only 2 channels in the training. Again, the student sees a reduced number of channels, which are sampled randomly.

The first two methods, All and Random, are our baselines. To remove channels, we either mask their encodings (ViT) or channels themselves (ResNet) by imputing 0. When using ViT, we initialize the student and teacher using weights trained on the JUMP-CP dataset. We initialize ResNet-based models using CLOOME weights. We also evaluate both pre-trained models, annotated as None in the results. There might be a data leak when using CLOOME weights because it was pretrained on the Bray dataset and the exact splits might not be the same as the ones used in this work. However, the data leak is not likely to occur, because CLOOME training does not use any labels related to Mode of Action or property.

The scheduled channel reduction follows the principles of Curriculum Learning <sup>33</sup>, which demonstrates that progressively increasing task difficulty during training prevents convergence to poor local optima and can lead to improved generalization. In the context of our work, the gradual removal of channels represents an increasing difficulty for the student network. This principle has been validated in multimodal medical imaging settings, where scheduled modality dropout forces networks to learn more robust cross-modal representations<sup>31</sup>. Similarly, Scheduled DropHead <sup>32</sup> has shown benefits of progressive regularization in transformer architectures. We note, however, that in our experiments the advantage of Schedule over Random is modest and primarily evident for the most challenging channel configurations (Table 5), which is consistent with curriculum learning benefits being most pronounced under difficult learning conditions.

## Evaluation of representation

To evaluate the model, we generate the representation of images in the evaluation dataset, as presented in Fig. 3. Following the DINO framework <sup>24</sup>, we use the teacher network weights for generating representations at evaluation time. The teacher model’s quality is assessed through its downstream performance on the prediction and retrieval tasks described below, which serve as the benchmark for evaluating whether the learned representations capture biologically meaningful information. We generate the representation of various sets of channels to see the

scientific reports

Scientific Reports

<!-- page 11 of 14 -->

https://doi.org/10.1038/s41598-026-69424-3

Article in Press

difference in performance. Then, a Multi Layer Perceptron (MLP) is trained using this representation to predict an array of tasks corresponding to compounds’ properties (total: 121 tasks) and Modes of Action (total: 30 tasks). We use the same metrics as in the previous works <sup>10,39,40</sup>:

● Avg: average ROC AUC across all tasks, either Mode of Action or Property,

● ≥ 0.7: number of tasks with ROC AUC equal to or over 0.7,

● ≥ 0.8: number of tasks with ROC AUC equal to or over 0.8,

● ≥ 0.9: number of tasks with ROC AUC equal to or over 0.9,

For each metric, we provide mean and median absolute deviation (MAD) values. To evaluate significance of the results, we applied the Z-score analysis with 95% confidence interval. MLP is built from five linear layers, ReLU activation, and dropout 𝑝𝑝 𝑝 𝑝𝑝𝑝 which was selected following architecture from<sup>40</sup>. The dimensions of representation vectors are 768 for ViT and 512 for ResNet. The choice of a five-layer MLP for evaluation follows our previous benchmark <sup>40</sup>, which systematically compared MLP architectures for this task and found that deeper networks were necessary to capture the non-linear relationships between image representations and biological activity labels. This is consistent with findings in related domains, where non-linear evaluation heads outperform linear probes for fine-grained biological distinctions <sup>48,49</sup>. We adopt the same architecture across all channel configurations to ensure that observed differences reflect only the effect of channel reduction, not the evaluation setup.

Additionally, we test the representation using the retrieval task, where the image is retrieved from the chemical structure and vice versa. Representation successful in this task has higher potential to actually describe biologically relevant changes in the cell, rather than just capturing visual noise or technical batch effects. <sup>35</sup> Images are encoded using ResNet backbone to allow use of code provided by authors of CLOOME. Chemical structures are firstly described using SMILES <sup>50</sup>, then encoded using 1024-bit Morgan fingerprints <sup>51</sup>, and finally processed using MLP initialized with CLOOME weights. CLOOME model is trained to increase the similarity of image-structure pairs and decrease the similarity of not pairs. Firstly, we analyze if removing selected channels decreases the models’ ability to match images and structures, and then, we measure the performance of the finetuned image model in this task calculated using accuracy for Top-k results, where 𝑘𝑘 ∈ {1, 5, 10}.

**Author contributions** A.B. prepared the data, conceived and conducted the experiments, A.B. and D.R. validated the results. D.R. and B.Z. obtained the funding. All authors analyzed the results, wrote and reviewed the manuscript.

**Funding** The research was carried out within the research project “Interpretable and Interactive Multimodal Retrieval in the Drug Discovery” (grant no. FENG.02.02-IP.05-0040/23) within the First Team FENG program of the Foundation for Polish Science co-financed by the European Union under the Funds for a Modern Economy.

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2023/016834.

Moreover, some experiments were performed on servers purchased with funds from a grant from the Priority Research Area (Artificial Intelligence Computing Center Core Facility) under the Strategic Programme Excellence Initiative at Jagiellonian University.

**Data availability** The dataset analyzed during the current study is available at The Cell Image Library repository as Plates 24277-26796, [https://github.com/gigascience/paper-bray2017](https://github.com/gigascience/paper-bray2017).

## Declarations

**Code availability** Code is available at [https://github.com/gmum/cr-dino](https://github.com/gmum/cr-dino).

**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your

Scientific Reports

scientific reports

<!-- page 12 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/).

## References

1. Bray, M.-A. et al. Cell painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nat Protoc 11, 1757–1774 (2016).

2. Chandrasekaran, S. N. et al. Jump cell painting dataset: Morphological impact of 136,000 chemical and genetic perturbations. bioRxiv [https://doi.org/10.1101/2023.03.23.534023](https://doi.org/10.1101/2023.03.23.534023) (2023).

3. Sypetkowski, M. Rxrx1: A dataset for evaluating experimental batch correction methods. In Proceedings of the IEEE/ CVF Conference on Computer Vision and Pattern Recognition, pp. 4284–4293 (2023).

4. Bray, M.-A. et al. A dataset of images and morphological profiles of 30 000 small-molecule treatments using the cell painting assay. Gigascience 6, giw014 (2017).

5. Haney, S. A., LaPan, P., Pan, J. & Zhang, J. High-content screening moves to the front of the line. Drug Discov Today 11, 889–894 (2006).

6. Stirling, D. R. et al. Cellprofiler 4: Improvements in speed, utility and usability. BMC Bioinform 22, 1–11 (2021).

7. Jamali, N. et al. Assessing the performance of the cell painting assay across different imaging systems. Biorxiv 2023–02 (2023).

8. Ando, D. M., McLean, C. Y. & Berndl, M. Improving phenotypic measurements in high-content imaging screens. BioRxiv 161422 (2017).

9. Kensert, A., Harrison, P. & Spjuth, O. Transfer learning with deep convolutional neural networks for classifying cellular morphological changes. SLAS DISCOV: Adv Life SciR&D 24, 247255521881875. [https://doi.org/10.1177/2472555218818756](https://doi.org/10.1177/2472555218818756) (2019).

10. Hofmarcher, M., Rumetshofer, E., Clevert, D.-A., Hochreiter, S. & Klambauer, G. Accurate prediction of biological assays with high-throughput microscopy images and convolutional networks. J Chem Inf Model 59, 1163–1171 (2019).

11. Dürr, O. & Sick, B. Single-cell phenotype classification using deep convolutional neural networks. J Biomol Screen 21, 998–1003 (2016).

12. Masud, U., Cohen, E., Bendidi, I., Bollot, G. & Genovesio, A. Comparison of semi-supervised learning methods for high content screening quality control. In European Conference on Computer Vision, pp. 395–405. Springer, (2022).

13. Lu, A. X., Kraus, O. Z., Cooper, S. & Moses, A. M. Learning unsupervised feature representations for single cell microscopy images with paired cell inpainting. PLoS Comput Biol 15, e1007348 (2019).

14. Lafarge, M. W. et al. Capturing single-cell phenotypic variation via unsupervised representation learning. In International Conference on Medical Imaging with Deep Learning, pp. 315–325 PMLR, (2019).

15. Goldsborough, P., Pawlowski, N., Caicedo, J. C., Singh, S. & Carpenter, A. E. Cytogan: Generative modeling of cell images. BioRxiv 227645 (2017).

16. Siegismund, D., Wieser, M., Heyse, S. & Steigele, S. Self-supervised representation learning for high-content screening. In Konukoglu, E. et al. (eds.) In Proceedings of The 5th International Conference on Medical Imaging with Deep Learning, 172 of Proceedings of Machine Learning Research, pp. 1108–1124 (PMLR), (2022).

17. Borowa, A., Kruczek, S., Tabor, J. & Zieliński, B. Weakly-supervised cell classification for effective high content screening. In International Conference on Computational Science, pp. 318–330. Springer, (2022).

18. Janssens, R., Zhang, X., Kauffmann, A., de Weck, A. & Durand, E. Y. Fully unsupervised deep mode of action learning for phenotyping high-content cellular images. Bioinformatics 37, 4548–4555. [https://doi.org/10.1093/bioinformatics/btab497](https://doi.org/10.1093/bioinformatics/btab497) (2021).

19. Shi, B., Hoffman, J., Saenko, K., Darrell, T. & Xu, H. Auxiliary task reweighting for minimum-data learning. Adv Neural Inform Process Syst 33, 7148–7160 (2020).

20. Caicedo, J. C., McQuin, C., Goodman, A., Singh, S. & Carpenter, A. E. Weakly supervised learning of single-cell feature embeddings. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9309–9318 (2018).

21. Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR (2021).

22. Sivanandan, S. et al. A pooled cell painting crispr screening platform enables de novo inference of gene function by self-supervised deep learning. bioRxiv [https://doi.org/10.1101/2023.08.13.553051](https://doi.org/10.1101/2023.08.13.553051) (2023).

23. Bao, Y., Sivanandan, S. & Karaletsos, T. Channel vision transformers: An image is worth 1 x 16 x 16 words. In The Twelfth International Conference on Learning Representations (2024).

24. Caron, M. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9650–9660 (2021).

25. Doron, M. et al. Unbiased single-cell morphology with self-supervised vision transformers. bioRxiv 2023–06 (2023).

scientific reports

Scientific Reports

<!-- page 13 of 14 -->

https://doi.org/10.1038/s41598-026-69424-3

Article in Press

26. Kim, V., Adaloglou, N., Osterland, M., Morelli, F. & Zapata, P. A. M. Self-supervision advances morphological profiling by unlocking powerful image representations. bioRxiv 2023–04 (2023).

27. Cross-Zamirski, J. O. Self-supervised learning of phenotypic representations from cell images with weak labels (2022) [arXiv:2209.07819](http://arxiv.org/abs/2209.07819) arXiv preprint.

28. Haslum, J. F., Matsoukas, C., Leuchowius, K.-J., Müllers, E. & Smith, K. Metadata-guided consistency learning for high content images. In Medical Imaging with Deep Learning, 918–936 (PMLR), (2024).

29. Crowley, L. C., Marfell, B. J. & Waterhouse, N. J. Analyzing cell death by nuclear staining with hoechst 33342. Cold spring Harbor protocols 2016, pdb–prot087205 (2016).

30. Biosciences, A. Araceli endeavor high content analysis platform. [https://www.aracelibio.com/](https://www.aracelibio.com/) (2021). Accessed: 2024-0 7-02.

31. Havaei, M., Guizard, N., Chapados, N. & Bengio, Y. Hemis: Hetero-modal image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 469–477. Springer, (2016).

32. Zhou, W., Ge, T., Wei, F., Zhou, M. & Xu, K. Scheduled drophead: A regularization method for transformer models. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1971–1980 (2020).

33. Bengio, Y., Louradour, J., Collobert, R. & Weston, J. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 41–48 (2009).

34. Mendez, D. et al. Chembl: Towards direct deposition of bioassay data. Nucleic Acids Res 47, D930–D940 (2019).

35. Sanchez-Fernandez, A., Rumetshofer, E., Hochreiter, S. & Klambauer, G. Cloome: Contrastive learning unlocks bioimaging databases for queries with chemical structures. Nat Commun 14, 7339 (2023).

36. Li, B. et al. Phenoprofiler: Advancing phenotypic learning for image-based drug discovery. Nat Commun 17, 793 (2025).

37. Tang, Q. et al. Morphological profiling for drug discovery in the era of deep learning. Brief Bioinform 25, bbae284 (2024).

38. Moshkov, N. et al. Learning representations for image-based profiling of perturbations. Nat Commun 15, 1594 (2024).

39. Simm, J. et al. Repurposing high-throughput image assays enables biological activity prediction for drug discovery. Cell Chem Biol 25, 611–618 (2018).

40. Borowa, A. et al. Decoding phenotypic screening: A comparative analysis of image representations. Comput Struct Biotechnol J 23, 1181–1188 (2024).

41. Ferro, A. et al. Blue intensity matters for cell cycle profiling in fluorescence dapi-stained images. Lab Investig 97, 615–625 (2017).

42. Monzel, A. S., Enríquez, J. A. & Picard, M. Multifaceted mitochondria: Moving mitochondrial science beyond function and dysfunction. Nat Metab 5, 546–562 (2023).

43. Suomalainen, A. & Nunnari, J. Mitochondria at the crossroads of health and disease. Cell 187, 2601–2627 (2024).

44. Grill, J.-B. et al. Bootstrap your own latent: A new approach to self-supervised learning. Adv Neural Inform Process Systems 33, 21271–21284 (2020).

45. Yang, C., Xie, L., Qiao, S. & Yuille, A. Knowledge distillation in generations: More tolerant teachers educate better students. arXiv preprint [arXiv:1805.05551](http://arxiv.org/abs/1805.05551) (2018).

46. Cho, J. H. & Hariharan, B. On the efficacy of knowledge distillation. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4793–4801 IEEE, (2019).

47. Mirzadeh, S. I. et al. Improved knowledge distillation via teacher assistant. In In Proceedings of the AAAI Conference on Artificial Intelligence 34, 5191–5198 (2020).

48. Chossegros, M., Delhommeau, F., Stockholm, D. & Tannier, X. Improving the generalizability of white blood cell classification with few-shot domain adaptation. J Pathol Inform 15, 100405 (2024).

49. Li, J. Can we simplify slide-level fine-tuning of pathology foundation models? (2025) [arXiv:2502.20823](http://arxiv.org/abs/2502.20823) arXiv preprint.

50. Weininger, D. Smiles. A chemical language and information system. 1. introduction to methodology and encoding rules. J Chem inform Comput Sci 28, 31–36 (1988).

51. Morgan, H. L. The generation of a unique machine description for chemical structures: A technique developed at chemical abstracts service. J Chem Doc 5, 107–113 (1965).

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Scientific Reports

scientific reports

<!-- page 14 of 14 -->

Article in Press

https://doi.org/10.1038/s41598-026-69424-3

## Authors and Affiliations

**Adriana Borowa**<strong><sup>1,2,3</sup></strong>  **· Bartosz Zieliński**<strong><sup>1</sup></strong> **· Dawid Rymarczyk**<strong><sup>1,3</sup></strong>

Adriana Borowa [ada.borowa@student.uj.edu.pl](mailto:ada.borowa@student.uj.edu.pl)

1 Faculty of Mathematics and Computer Science, Jagiellonian University, ul. Lojasiewicza 6, 30-348 Krakow, Poland

Doctoral School of Exact and Natural Sciences, Jagiellonian University, ul. Lojasiewicza 11, 30-348 Krakow, Poland

3 Ardigen SA, ul. Sternbacha 1, 30-394 Krakow, Poland

scientific reports

Scientific Reports