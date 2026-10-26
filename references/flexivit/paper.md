<!-- page 1 of 29 -->

arXiv:2212.08013v2 [cs.CV] 23 Mar 2023

# FlexiViT: One Model for All Patch Sizes

Lucas Beyer?1 Pavel Izmailov?1,3 Alexander Kolesnikov?1 Mathilde Caron?2 Simon Kornblith?1 Xiaohua Zhai?1 Matthias Minderer?1 Michael Tschannen?1 Ibrahim Alabdulmohsin?1 Filip Pavetic?1

Google Research

![Image block](images/page001-block05.png)

Figure 1. FlexiViT is a standard ViT model that sees randomized patch sizes, hence sequence lengths, during training. The patch embedding weights are resized adaptively for each patch size and the model weights are shared as-is across all patch sizes.

## Abstract

Vision Transformers convert images to sequences by slicing them into patches. The size of these patches controls a speed/accuracy tradeoff, with smaller patches leading to higher accuracy at greater computational cost, but changing the patch size typically requires retraining the model. In this paper, we demonstrate that simply randomizing the patch size at training time leads to a single set of weights that performs well across a wide range of patch sizes, making it possible to tailor the model to different compute bud-

![Chart block](images/page001-block09.png)

Figure 2. FlexiViT results on ImageNet-1k. We train three Flexi-ViTs based on DeiT III on ImageNet-1k and show their speedaccuracy tradeoff when evaluated at various patch sizes. Each curve corresponds to a single model with a single set of weights run with different patch sizes. We also evaluate DeiT III at various patch sizes to show ViT’s natural inflexibility. EfficientNetv2 numbers from [55] and ResNet50 numbers from [5], the latter distills from an ImageNet-21k pretrained teacher. FlexiViT was trained for 1000 epochs, but runs for 600, 300, and 90 epochs shown as shaded curves indicate that long training mostly benefits the short-sequence setting and is not strictly necessary.

gets at deployment time. We extensively evaluate the resulting model, which we call FlexiViT, on a wide range of tasks, including classification, image-text retrieval, open-world detection, panoptic segmentation, and semantic segmentation, concluding that it usually matches, and sometimes outperforms, standard ViT models trained at a single patch size in an otherwise identical setup. Hence, FlexiViT training is a simple drop-in improvement for ViT that makes it easy to add compute-adaptive capabilities to most models relying on a ViT backbone architecture. Code and pre-trained models are available at github.com/googleresearch/big\_vision.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">?All authors made significant technical contributions. Lucas started and led the project.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">1 Google Research, Brain Team. 2 Google Research.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">3 work done at Google Brain, while being a PhD student at NYU.</span></small>

1

<!-- page 2 of 29 -->

## 1. Introduction

Vision Transformers (ViTs) cut images into non-overlapping patches and perform all computations on tokens created from these patches. This “patchification” procedure represents a significant shift away from the previously dominant convolutional neural network (CNN) approach [33], where an image is processed with small local and typically overlapping filters. Patchification has unlocked new capabilities, such as (random) dropping of image patch tokens [10, 21, 47, 56, 65], adding specialized tokens for new tasks [57, 59] or mixing image tokens with tokens from other modalities [1, 41, 68].

Despite the importance of patchification for ViT models, the role of the patch size has received little attention. While the original ViT paper [16] works with three patch sizes (32×32, 16×16, and 14×14 pixels), many follow-up works fix the patch size at 16×16 pixels [57, 58, 69]. In this work, we show that the patch size provides a simple and effective lever to change the compute and predictive performance of a model, without changing model parametrization. For example, a ViT-B/8 model achieves 85.6% top-1 accuracy on ImageNet1k with 156 GFLOPs and 85 M parameters, while a ViT-B/32 model achieves only 79.1% accuracy with 8.6 GFLOPs and 87 M parameters. Despite the major difference in performance and compute, these models have essentially the same parametrization. However, standard ViT models perform well only at the patch size that they have been trained at. Tuning the patch size therefore requires complete re-training of the model.

To overcome this limitation, we propose FlexiViT, a flexible ViT which matches or outperforms standard fixed-patch ViTs across a wide range of patch sizes with no added cost. To train FlexiViT, we randomize the patch size during training, and resize the positional and patch embedding parameters adaptively for each patch size, as shown in Figure 1. These simple modifications are already sufficient for strong performance, but we also propose a optimized resizing operation and a training procedure based on knowledge distillation which achieves even better results.

We demonstrate the efficiency of FlexiViT models in many downstream tasks, such as image classification, transfer learning, panoptic and semantic segmentation, imagetext retrieval and open-world recognition, and provide a general recipe for flexifying existing ViT-based training setups. Furthermore, we show that flexibility of the backbone, i.e. strong performance across patch sizes, is often preserved even after fine-tuning with a fixed patch size. We leverage this observation to perform resource-efficient transfer learning: we finetune the model cheaply with a large patch size, but then deploy it with a small patch size for strong downstream performance. We further show that flexible patch size can be used to accelerate pre-training.

To explain the effectiveness of FlexiViT, we analyze the

model’s representations. We find that the representations are often similar across different patch sizes, especially in the deeper layers. Finally, we show that FlexiViT outperforms alternative architectural ways of controlling the performance-compute trade-off in ViT models.

## 2. Related work

Several recent works explore improving ViT’s efficiency by exploiting patchification. Some suggest removing tokens, either in randomized [21] or structured [10] fashion throughout training. Others aim to quantify a token’s importance and remove the least important ones, during [47, 65] or after [56] training. [60] trained a cascade of Transformers using increasing number of tokens to allow early exiting during inference. Conversely, we always keep all tokens and do not discard any information. It may be possible to combine such approaches with FlexiViT in future work.

Another related line of work looks at changing input resolution during training, typically in order to speed-up pre-training [38, 55, 58, 62] or as data augmentation for self-supervised learning of ViTs [35]. We do not explore the data augmentation inpact of Flexi training to avoid an explosion in scope. The aforementioned models all work only at their single, final resolution, while our ai is to have one model that works well at all trained patch-sizes.

More similar to our approach, the Neural Architecture Search (NAS) field is converging towards training one “supernet” from which individual, differently-shaped “subnets” can be extracted [8, 19, 67]. Since these works aim for changes in most or all model dimensions, they usually involve multiple specialized architectural additions. Super-ViT [36] is most related to FlexiViT as it patchifies an image at multiple scales, passes all these patches to ViT, while dropping random tokens [21] to reduce the sequence length. In contrast to the aforementioned works, our sharpened focus on ViT’s patch size only, allows benefiting from existing pretrained models, future ViT improvements, and is an easy drop-in to any existing training procedure.

Matryoshka representation learning [32] proposes training models whose output vector contains meaningful subvectors. This can be seen as the complement of FlexiViT.

## 3. Making ViT flexible

In this section we show that standard ViT models are not flexible, and introduce the FlexiViT model and training procedure in the supervised image classification setting. We perform all experiments in this section on the public ImageNet-21k dataset [49]. We use the base (ViT-B) scale model and unregularized light2 setting from [53], and train the models for 90 epochs following [39].

2

<!-- page 3 of 29 -->

![Chart block](images/page003-block01.png)

Figure 3. Standard ViTs are not flexible in patch size. However, FlexiViT can train them to be flexible without loss of performance.

## 3.1. Background and notation

FlexiViT is based on the Vision Transformer (ViT) architecture [16]. Here, we briefly describe the ViT architecture and introduce the necessary notation.

Consider an image $x \; \in \; \mathbb { R } ^ { h \times w \times c }$ , where $( h , w , c )$ are the width, height and number of channels respectively. ViT first tokenizes the input image into a sequence of s patches $x _ { i } \in \mathbb { R } ^ { p \times p \times c }$ , where $i \; \in \; \{ 1 , \ldots , s \}$ We refer to this procedure as patchification and illustrate it in Figure 1. The sequence length $s \; = \; \lfloor h / p \rfloor   \cdot   \lfloor w / p \rfloor$ is the number of patches (or tokens) after patchification and controls the amount of compute used by the ViT: self-attention scales as $\mathcal { O } ( s ^ { 2 } )   =   \mathcal { O } ( h ^ { 4 } )   =   \mathcal { O } ( w ^ { 4 } )$ , i.e. quartically in terms of image height (or width).

Next, we compute patch embeddings $e _ { i }   =   ( e _ { i } ^ { k } ) _ { k = 1 } ^ { d }   \in$ $\mathbb { R } ^ { d }$ for each patch $x _ { i } \colon e _ { i } ^ { k } = \langle x _ { i } , \omega _ { k } \rangle = \operatorname { v e c } ( x _ { i } ) ^ { T } \widetilde { \operatorname { v e c } ( \omega _ { k } ) } ,$ where $\omega _ { k } \in \mathbb { R } ^ { p \times p \times c }$ are the patch embedding weights, $\langle \cdot , \cdot \rangle$ denotes the dot product, and vec is the operation flattening a multi-dimensional array to a vector. Finally, we add learned position embeddings $\pi _ { i } \in \mathbb { R } ^ { d }$ to the patch embeddings $t _ { i } =$ $e _ { i } + \pi _ { i }$ . We then pass the sequence of s tokens $t _ { i }$ as input to the Transformer encoder, as illustrated in Figure 1.

In summary, for a given image size $h \times w ,$ the patch size p determines the length s of the input sequence to the Transformer model: smaller patch sizes correspond to longer input sequences and slower, more expressive models. Following [16], we denote ViT models as $\operatorname { V i T - } { \mathcal { S } } / p ,$ where $\mathcal { S }   \in   \{ \mathrm { S } , \mathrm { M } , \mathrm { B } , \mathrm { L } , \dots \}$ is the model scale (small, medium, base, large, . . . ) and $p$ is the patch size. Note that there are only two parts of the model where the parameter vectors depend on the patch size: the patch embedding weights $\omega _ { k }$ and the position embedding π. In the following sections, we will develop a flexible ViT model which works simultaneously for any patch size.

```python
Algorithm 1 Minimal FlexiViT pseudo-implementation.
model = ViT(...)
for batch in data:
    ps = np.random.choice([8, 10, ..., 40, 48])
    logits = model(batch["images"], (ps, ps))
    # [...] backprop and optimize as usual
class ViT(nn.Module):
    def __call__(self, image, patchhw):
        # Patchify, flexibly:
        w = self.param("w_emb", (32, 32, 3, d))
        b = self.param("b_emb", d)
        w = resize(w, (*patchhw, 3, d))
        x = conv(image, w, strides=patchhw) + b
        # Add flexible position embeddings:
        pe = self.param("posemb", (7, 7, d))
        pe = resize(pe, (*x.shape[1:3], d))
        return TransformerEncoder(...)(x + pe)
```

Notes: Changes to existing code highlighted via violet background.

## 3.2. Standard ViTs are not flexible

We first show that evaluating a standard pre-trained ViT model at different patch sizes yields poor performance. In order to change the patch size, we simply resize the patch embedding weights ω and the position embeddings π with bilinear interpolation. <sup>2</sup>. For the position embeddings, this resize approach was already proposed in the original ViT paper [16] to fine-tune at higher resolution.

The result is shown in Figure 3, where we see that the performance of standard ViT models (dashed and dotted lines) rapidly degrades as the inference-time patch size departs from the one used during training.

## 3.3. Training flexible ViTs

In Figure 3 we also show the performance of our FlexiViT-B model (solid line), which matches both ViT-B/16 and ViT-B/30 when evaluated at their training patch sizes, and significantly outperforms them for all other patch sizes. This model was trained in the same setting as the ViT-B/16 and ViT-B/30 models, except that at each step of training, the patch size was chosen uniformly at random from a set of pre-defined patch sizes.<sup>3</sup>In order to do so, two small changes to the model and training code are necessary.

First, the model needs to define an underlying parameter shape for ω and $\pi .$ . The learnable parameters are of that shape, and resized on-the-fly as part of the model’s forward pass. We show in Appendix B that the exact shape of these underlying learnable parameters does not matter much, and we use an underlying size of $3 2 \times 3 2$ for patches and $7 \times 7$ for position embeddings in all experiments.

Second, to have a large variety of patch sizes that perfectly tile the image, we use an image resolution of $2 4 0 ^ { 2 } \mathrm { { \tt p x } }$

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>[tf.image.resize(input, res, method='bilinear')](https://www.tensorflow.org/api_docs/python/tf/image/resize)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>3</sup>We sample patch sizes uniformly in most experiments. Some early runs used a distribution which slightly favors intermediate patch sizes. Later experiments showed that the distribution makes little difference (Appendix C). We therefore did not re-run the early experiments.</span></small>

3

<!-- page 4 of 29 -->

which allows for patch sizes $\it { p } \in \{ 2 4 0 , 1 2 0 , 6 0 , 4 8 , 4 0 , 3 0 }$ 24, 20, 16, 15, 12, 10, 8, 6, 5, 4, 2, 1}, of which we use all between 48 and 8, inclusive.<sup>4</sup> At each iteration we sample p from the uniform distribution $\mathcal { P }$ over these patch sizes.

These are all the changes necessary to flexify an existing ViT training procedure. Algorithm 1 summarizes them.

Note that changing the patch size is related to, but not identical to, changing the image size. The patch size is purely a change to the model while changing the image size may drastically reduce the available information. This distinction is further explored in Section 3.4.

We explore two alternative ways to flexify ViTs in Section 7: flexible depth and flexible patch stride. Both of them have merits, but patch size works best.

## 3.4. How to resize patch embeddings

Consider a patch $x \in \mathbb { R } ^ { p \times p }$ of the input image, and the patch embedding weights $\omega   \in   \mathbb { R } ^ { p \times p }$ and let’s assume a simple scenario when we are dealing with non-negative values. If we resize both the patch and the embedding weights with bilinear interpolation, the magnitude of the resulting tokens will differ greatly; for example $\langle x , \omega \rangle$ ≈ ${ \textstyle { \frac { 1 } { 4 } } } { \check { \langle } } \mathrm { r e s i z e } _ { p } ^ { 2 p } ( x )$ , resize ${ \mathrm { e } } _ { p } ^ { 2 p } ( \omega ) \rangle$ . We hypothesize that this dramatic change in token norm is part of the reason of ViT’s inflexibility, and an inductive bias that hinders learning of a single FlexiViT. Ideally, as long as there is no loss of information during resizing, the patch embeddings $e _ { i }   =   \langle x , \omega \rangle$ after resizing both the input x and the embedding ω should remain the same.

One way to achieve this equality is to normalize the tokens right after their embedding, either explicitly or by using a LayerNorm [2] module. However, this approach requires changing the model architecture and is not compatible with existing pre-trained ViTs. Further, it does not exactly preserve the patch embeddings. As we will show, there is a more principled way of achieving this goal, which is compatible with existing pre-trained models and does not require any architectural change.

First, we note that the linear resize operation introduced in Section 3.2 can be represented by a linear transformation:

$$
\operatorname{resize} _ {p} ^ {p *} (o) = B _ {p} ^ {p *} \operatorname{vec} (o),\tag{1}
$$

where $o \; \in \; \mathbb { R } ^ { p \times p }$ is any input, and $B _ { p } ^ { p _ { * } } \: \in \: { \mathbb { R } ^ { { p _ { * } } ^ { 2 } \times p ^ { 2 } } }$ . We resize channels of multi-channel inputs o independently.

Intuitively, we would like to find a new set of patchembedding weights $\hat { \omega }$ such that the tokens of the resized patch match the tokens of the original patch. Formally, we want to solve the optimization problem:

$$
\hat {\omega} \in \arg \min _ {\hat {\omega}} \mathbb {E} _ {x \sim \mathcal {X}} \left[ (\langle x, \omega \rangle - \langle B x, \hat {\omega} \rangle) ^ {2} \right],\tag{2}
$$

![Chart block](images/page004-block13.png)

Figure 4. Various ways of “resizing” ViTs. We load a ViT-B/8 from [53] trained on $2 2 4 ^ { 2 }$ px, resize patch-embeddings and input images by the same factor, and compute validation accuracy. PI-resize is the only method that stays accurate when upscaling.

where $B ~ = ~ B _ { p } ^ { p _ { * } }$ and $\mathcal { X }$ is some distribution over the patches. In case when we are increasing the patch size, i.e. $p _ { * } \geq p ,$ we can use $\hat { \omega } = P \omega$ where $P = B ( B ^ { T } B ) ^ { - 1 }$ = $( B ^ { T } ) ^ { + }$ is the pseudoinverse of $B ^ { T }$ :

$$
\langle B x, \hat {\omega} \rangle = x ^ {T} B ^ {T} B (B ^ {T} B) ^ {- 1} w = x ^ {T} w = \langle x, w \rangle .\tag{3}
$$

This way we match the patch embeddings exactly for all $x .$

In the case of downsampling, i.e. when $p _ { * } < p ,$ the solution to the optimization problem in Eq. (2) will in general depend on the patch distribution $\mathcal { X } .$ In Appendix $\mathrm { A } . 2$ , we show that for $\mathcal { X } = \mathcal { N } ( 0 , I )$ , we recover the pseudoinverse $\hat { \omega } = P \omega = ( B ^ { T } ) ^ { + } \omega$ as the optimal solution5. To sum up, we define PI-resize (pseudoinverse resize) as:

$$
\mathrm{PI-resize} _ {p} ^ {p _ {*}} (w) = \left((B _ {p} ^ {p _ {*}}) ^ {T}\right) ^ {+} \operatorname{vec} (\omega) = P _ {p} ^ {p _ {*}} \operatorname{vec} (\omega),\tag{4}
$$

where $P _ { p } ^ { p _ { * } }   \in   \mathbb { R } ^ { { p _ { * } } ^ { 2 } \times p ^ { 2 } }$ is the matrix corresponding to the PI-resize transformation. The PI-resize operation resizes the patch embedding weights, serving as an inverse of the bilinear resize operation.

To experimentally validate the effectiveness of PI-resize and compare it to several alternative heuristics, including standard linear resize, we load a pre-trained ViT-B/8 model from [53] and evaluate it after resizing both the image and the model, thus preserving its sequence length $s =$ $( 2 2 4 / 8 ) ^ { 2 }   =   7 8 4$ . The results, shown in Figure $4 ,$ demonstrate that PI-resize maintains nearly constant performance when upsampled, and degrades gracefully when downsampling. None of the heuristics works as well as thoughtful PI-resize across the board.

For completeness, in Appendix A.1 we experimentally compare the remaining ways of dealing with variable patch

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>4</sup>Perfect tiling may not be strictly necessary, and it may be fine to use arbitrary patch sizes and ignore a small border of the image. For simplicity, we focus on the perfect tiling setting.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>5</sup>We can also target the patch distribution in the data in place of X, producing a resize operation which depends on the data. In our preliminary experiments, we did not observe significant benefits from this approach.</span></small>

4

<!-- page 5 of 29 -->

![Chart block](images/page005-block01.png)

Figure 5. The effect of initialization when distilling to FlexiViT.

sizes when one does not care about maintaining model compatibility. These methods include fixed normalization, LayerNorm, and learning separate parameters ω for each patch size. Adding a LayerNorm works best, but otherwise, PI-resize and bilinear resize are among the best techniques.

## 3.5. Connection to knowledge distillation

Knowledge distillation [24] is a popular technique, where a typically smaller student model is trained to mimic the predictions of a typically larger teacher model. This can significantly improve the performance of the student model compared to standard label-supervised training [5, 12, 64].

It was recently shown that knowledge distillation corresponds to a much more challenging optimization problem than standard supervised training [5, 52], and that initializing the student close to the teacher simplifies alleviates this [52]. Unfortunately, this solution is impractical since the teacher usually has a different (larger) architecture than the student [5]. However, with FlexiViT, we can initialize a student FlexiViT with the weights of a powerful ViT teacher and significantly improve distillation performance.

Unless otherwise stated, the model we use for the remaining experiments in this paper is a FlexiViT-B initialized and distilled from the powerful ViT-B/8 model of [53]. At initialization, we PI-resize the teacher’s patch embedding weights to $3 2   \times   3 2 ,$ and bilinearly resample its position embeddings $\mathrm { t o } ~ 7 \times 7$ . We then train the student model following the FunMatch [5] approach, minimizing the KL-divergence between the predictions of the teacher and the student FlexiViT with a randomized patch size:

$$
\mathbb {E} _ {x \in \mathcal {D}} \mathbb {E} _ {p \sim \mathcal {P}} \mathrm{KL} \left(f _ {\text {FlexiViT}} (x, p) | | f _ {\text {ViT - B/8}} (x)\right),\tag{5}
$$

where $f _ { \mathrm { F l e x i V i T } } ( x , p )$ is the distribution over classes for the FlexiViT model on an input x with patch size $p ,   f _ { \mathrm { V i T - B } / 8 } ( x )$ is the predictive distribution of the teacher on the exact same input, D is the training data distribution with random flips, crops, and mixup, and $\mathcal { P }$ is the distribution over patch sizes used for training the FlexiViT model.

![Image block](images/page005-block10.png)

Figure 6. t-SNE visualizations of intermediate representations of network layers across different grid sizes. Colors reflect different layers; dot sizes reflect different grid sizes.

Figure 5 compares the effect of distilling using teacher initialization to random initialization and to supervised training from labels. The comparison was performed for 90 epochs and shows considerable benefits of this unique initialization capability of FlexiViT. Since distillation needs patience [5, 57], we additionally run for 300 and 1000 epochs, shown as pale green curves in the figure. FlexiViT matches the teacher’s performance at small patch sizes, and teacher initialization provide a large improvement in accuracy at the largest patch sizes. In the following sections, we use the FlexiViT that was trained for 300 epochs and train two fixed ViT-B/30 and ViT-B/16 models in the same setting (including the initialization) as baselines.

## 3.6. FlexiViT’s internal representation

Does FlexiViT process inputs with different patch sizes in similar ways? We investigate this by analyzing the model’s internal representations. We apply minibatch centered kernel alignment (CKA) [14, 29, 42], a widely-used approach for comparing representations within and across neural networks. For visualization purposes, we apply an arccosine transform to transform CKA/cosine similarity to proper metrics [61] and then perform t-SNE.

Results are shown in Figure 6. Feature map representations are similar across grid sizes from the first layer until the MLP sublayer of block 6. At the MLP sublayer of block 6, layer representations diverge, before converging again at the final block. By contrast, CLS token representations remain aligned across grid sizes. Thus, although internal representations of a substantial portion of FlexiViT differ by grid size, output representations are generally aligned.

5

<!-- page 6 of 29 -->

![Chart block](images/page006-block01.png)

Transfer and evaluation setting

Figure 7. Using a pre-trained FlexiViT. We use the flexibly pre-trained FlexiViT-B model in a diverse set of downstream computer vision tasks at two patch sizes, and verify that it performs the same or better than a plain (inflexible) ViT model pre-trained at that patch size. These results indicate that flexibly pre-training a single ViT may be preferrable than pre-training several fixed ViTs.

## 4. Using pre-trained FlexiViTs

We have shown that ViTs can be trained flexibly without significant loss of upstream performance. Next, we verify that pre-trained FlexiViTs are still comparable to individual fixed patch-size ViTs when transferred to other tasks. We check this by transferring the single pre-trained FlexiViT with its patch size fixed to either $1 6 ^ { 2 }$ or to $3 0 ^ { 2 }$ during transfer. We compare FlexiViT to ViT-B/16 and a ViT-B/30 models that were pre-trained using the same distillation setup as FlexiViT (Section 3.5), but with a fixed patch size. We perform this transfer on the following set of diverse tasks.

For each task, we provide more details along with many more results, all with the same take-away, in Appendix E.

**Classification** We fine-tune on small- (Pet [44], Flowers [43]) and medium-scale (CIFAR10, CIFAR100 [31], Food101 [7], SUN397 [63]) classification datasets following the setup of [16] at $2 4 0 ^ { 2 }$ px resolution.

**Locked-image Tuning (LiT)** We follow [70] to train a text model contrastively [25, 46] for the frozen FlexiViT, which we evaluate in terms of 0-shot classification and retrieval.

**Open-vocabulary detection** We test the transferability of FlexiViT to object detection using OWL-ViT [40], an open-vocabulary object detector based on image-text models such as LiT or CLIP [46]. We evaluate its zero-shot open-vocabulary detection performance on LVIS [20].

**Panoptic segmentation** The Universal Vision Model (UViM) is a general-purpose modeling approach for vision [28]. We train UViM on the COCO panoptic segmentation dataset [26, 37] and use FlexiViT as initialization for the image encoder in UViM.

**Semantic segmentation** We transfer to semantic segmentation following Segmenter’s linear decoder setup [54]. We report mean IoU for single scale evaluation and evaluate on Cityscapes [13] and ADE-20k [71].

![Chart block](images/page006-block12.png)

Figure 8. Fast transfer. FlexiViT can be cheaply finetuned at small sequence length and used at test time with much longer sequence to achieve higher performance. (left) FlexiViT-B and ViT-B/30 models finetuned at grid size 8x8 (indicated by dots) and evaluated at other grid sizes. The standard ViT model’s accuracy quickly deteriorates, while FlexiViT demonstrates large performance boost with increased grid size. (right) A single FlexiViT-B model finetuned at three different grid sizes (indicated by dots) and evaluated at various grid sizes.

## 4.1. Results

The results of these transfer experiments are summarized in Figure 7. Across the diverse set of tasks, a single FlexiViT model roughly matches the two fixed ViT models, barely lagging behind at large patch size and leading to a small or significant improvement at smaller patch size.

These results confirm that there is no significant downside in using a pre-trained FlexiViT, as opposed to pre-training multiple ViTs for different patch sizes.

## 4.2. Resource-efficient transfer via flexibility

FlexiViT enables a new way of making transfer learning more resource efficient, saving accelerator memory and compute. This is possible because, surprisingly, flexibility is largely retained even after transfer at a fixed patch size. We can therefore perform transfer training cheaply with large

6

<!-- page 7 of 29 -->

![Chart block](images/page007-block01.png)

Figure 9. Flexified transfer of a flexible model works best, but even inflexible models can be flexified during transfer. The lines represent fixed transfer of fixed models, for reference.

![Chart block](images/page007-block03.png)

Figure 10. Flexified LiT transfer works the best, while fixed LiT transfer of FlexiViT with a single patch size (30 or 16) performs surprisingly well, similarly to Section 4.2.

input patches (small input grid), but later deploy the resulting model using small patch sizes (large input grid). We pre-form experiments by transferring a FlexiViT-B model (pre-trained on ImageNet-21k with distillation) to the ImageNet-1k dataset, and use a similarly pretrained fixed ViT-B/30 model as the baseline. The pretrained FlexiViT works well at larger grid sizes even after fixed-size transfer. For example, we can perform relatively cheap finetuning at 8 × 8 grid size. When evaluated at 8 × 8 grid size, the model achieves 81.8% accuracy, but when evaluated at the 24×24 grid size, it achieves 85.3% top-1 accuracy gaining 3.5% accuracy at no additional training cost (Figure 8). More details on the finetuning setup can be found in the Appendix D.

## 5. Flexifying existing training setups

So far, we have focused on flexifying models during pre-training. We now show that existing pre-trained models can also be flexified during transfer to downstream tasks. Below, we flexify a diverse set of existing training setups.

## 5.1. Transfer learning

We use the same set of 6 transfer datasets from Section 4, with the same settings. We again show the results for SUN397 in Figure 9 and all other datasets in Appendix E. The difference is that we now also randomize the patch size during transfer, and evaluate the single resulting model at different patch sizes (x-axis, three groups of bars).

Flexible transfer of FlexiViT works best, but flexifying a fixed model during transfer also works surprisingly well, considering the very short training and low learning rate used for transfer. The baseline of a fixed-size model transferred at a fixed patch size and evaluated at that same size is indicated by a small horizontal line.

![Chart block](images/page007-block11.png)

Figure 11. Flexified open-vocabulary detection. Image-text models are transferred to detection [40]. Dark green shows transfer of a fixed, pale green of a flexible model.

![Chart block](images/page007-block13.png)

Figure 12. Flexible patch sizes to accelerate pre-training. The ViT-B/16 models are trained for different epochs, (Flexi)ViT-B/16 are different curricula. Evaluation at patch size 16.

## 5.2. Multimodal image-text training

Next, we discuss two ways to flexify multimodal imagetext training: FlexiLiT and FlexiCLIP. In FlexiLiT, we train a text tower to produce text embeddings that align well with visual embeddings from various patch sizes (B/flexi). LiT baselines with direct use of either FlexiViT models at fixed resolutions, or ViT models are provided. Figure 10 shows zero-shot image to text retrieval results on the Flickr30k [45] dataset. FlexiLiT-B/flexi performs the best on average, while LiT with FlexiViT-B/30 and FlexiViT-B/16 both get very close results. Flexification additionally provides the possibility of fast transfer as discussed in Section 4.2. The LiT-ViT baselines shown in Figure 10 match FlexiLiT on the sequence length it has been trained for, but performance drops quickly when using a different sequence length during inference. We observe similar conclusions with a from-scratch image-text training setup, i.e. Flexi-CLIP (see Appendix G for more results).

## 5.3. Open-vocabulary detection

Beyond image-level tasks, we find that flexification also works for object detection training. We modify the training of OWL-ViT to introduce flexible patch sizes as described in Algorithm 1. Similar to classification, flexible OWL-ViT detection models perform close to or better than fixed-size models at any patch size during inference (Figure 11). In addition, we find that for detection, the optimal patch size is not necessarily the smallest. When evaluated on a set of 35 detection datasets [34], inference-time tuning of the patch size leads to improved results over evaluation at the smallest patch size (Appendix E). This makes flexification especially valuable for detection.

7

<!-- page 8 of 29 -->

## 5.4. Training times and flexification

Besides having a flexible model, one can use FlexiViT’s machinery to pre-train fixed ViTs faster. In this case, we specify a curriculum: a sequence $( p _ { k } ) _ { k = 1 } ^ { K }$ of probability distributions over the patch sizes along with a mapping $c   :   \mathbb { N }   \to   [ K ]$ that identifies which distribution $p _ { k }$ to use at training step t. For example, if the desired patch size is $1 6 \times 1 6 ,$ the last probability distribution in the sequence p<sub>K</sub> would place its entire mass on said patch size. A multitude of curricula can be designed, see Appendix H. In Figure 12 we show that in general training with a patch size curriculum leads to better performance per compute budget than standard training as we vary the training length.

## 6. Analyzing FlexiViTs

**Attention relevance patterns across scales** We find that decreasing the patch size results in attention relevance [9] to concentrate into a larger number of smaller areas throughout the image. In Figure 13 (top) we observe that attention can significantly change at different scales.

**Relation of token representations across scales** As we decrease FlexiViT’s patch size, each token “splits” into multiple tokens. A natural question is how token representations at larger patch sizes relate to token representations at smaller patch size. To answer this question, we measure co-sine similarity between the representation of a “seed” token at the center of a feature map at one patch size and representations of other tokens at the same and different patch sizes. As shown in Figure 13 (bottom), we are indeed able to find correspondences between tokens across scales.

**Ensembling** We explored whether it is possible to improve prediction accuracy by ensembling the predictions of the same FlexiViT at multiple scales. We find that, in terms of total compute spent, it is nearly always better to run a single FlexiViT at that compute budget than to ensemble multiple smaller ones. Full results are provided in Appendix J.

**Shape or texture bias** ViT’s bias towards using shape or texture features [18] has been shown to largely depend on its patch size [6]. In Appendix K, we show that FlexiViT evaluated at each patch size has a similar texture bias to a ViT trained and evaluated at that same patch size.

**Model and dataset size** Throughout the paper, we focus on FlexiViT models of the base size (-B) trained on 12 M images. In order to validate that neither of these two settings are required, we train FlexiViT-S,B,L models on ImageNet-1k (1.2 M images) using the ImageNet-1k DeiT III model [58] as teacher. We can see in Fig 2 that a single FlexiViT-L model matches or outperforms all three DeiT III models and EfficientNetV2. However, there is still a point at which it becomes more effective to change model width than patch size. Numerical results and evaluation on

Relevance map for the "fork" class

![Image block](images/page008-block10.png)

Similarity of center token representation

![Image block](images/page008-block12.png)

Figure 13. Analysis of FlexiViT attention and token representations across scales. Top: Attention relevance (as in [9]) can significantly change at different patch sizes. For example, FlexiViT-B/48 and FlexiViT-B/8 consider different areas of the input most relevant for class ’fork’. See Appendix L for more examples. Bottom: Cosine similarity between a seed token representation at the center of the feature map of FlexiViT-B at patch size 16 and representations of tokens at other patch sizes. Representations are taken from block 6 and averaged across our I21K validation set. See Appendix I for similar plots for other blocks and patch sizes.

ImageNet-ReaL/v2/A/R [3, 22, 23, 48] are in Appendix F, a version of Fig. 2 using GFLOPs [15] is in Appendix P.

## 7. Discussion of alternatives

Changing the input patch size is not the only way to trade off sequence length and compute in ViTs. We explore two alternatives in our core setup: distillation on ImageNet-21k.

**Varying patch embedding stride** One alternative is to fix the patch size and change its sampling stride, i.e. extract overlapping patches to increase sequence length. Intuitively, the advantage of this approach is that the intrinsic patch size is fixed and we avoid any special care when computing patch embeddings. Results in Figure 14 suggest varying the stride works almost as well, only slightly lagging behind our baseline.

**Varying model depth** Another alternative is adding flexibility in terms of depth, i.e. number of layers. Depth pruning has been explored in the context of NLP [17, 50] and more recently also for ViTs [66]. Depth pruning differs fundamentally from FlexiViT: it scales linearly in depth, uses a subset of parameters, and allows progresively refining a prediction. We randomize the depth by attaching the shared head to various intermediate layers. We also tried sepa-

8

<!-- page 9 of 29 -->

![Chart block](images/page009-block01.png)

Figure 14. Varying stride and depth as alternatives to FlexiViT

rate heads, which worked worse. The results in Figure 14 show that FlexiViT provides a significantly better computeaccuracy tradeoff than depth pruning.

## 8. Conclusion

FlexiViT is a simple and efficient way of trading off compute and predictive performance with a single model, enabled by the unique patch embedding strategy of ViTs. FlexiViT can be used to significantly reduce pre-training costs by only training a single model for all scales at once, and performs well at a variety of downstream tasks. There are many exciting directions for future work, and we hope that our results inspire the community to explore additional creative applications of patchification.

## 9. Acknowledgments

We thank Daniel Keysers for good feedback on a draft of the paper, Geoffrey Hinton for a nudge to pursue this project early on, and our respective teams at Google for encouraging creative and independent research.

We would also like to thank the following artists for making their photographs (used for visualizations) freely available through unsplash.com: Tanya Patrikeyeva, Markus Spiske, Matheus Bardemaker, Alexandru Sofronie, Chris Smith, Kajetan Sumila, Julee Juu, Mike Erskine, Piermanuele Sberni, Feyza Yıldırım and Sixteen Miles Out.

We thank the anonymous reviewers for good feedback that further improves our paper.

Finally, the raccoon picture used as background in Figure 31 is from publicdomainvectors.org.

## References

[1] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. VATT: transformers for multimodal self-supervised learning from raw video, audio and text. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, 2021. 2

[2] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016. 4, 13

[3] Lucas Beyer, Olivier J. Henaff, Alexander Kolesnikov, Xi- ´ aohua Zhai, and Aaron van den Oord. Are we done with ¨ imagenet? CoRR, abs/2006.07159, 2020. 8

[4] Lucas Beyer, Xiaohua Zhai, and Alexander Kolesnikov. Better plain vit baselines for imagenet-1k. CoRR, abs/2205.01580, 2022. 14

[5] Lucas Beyer, Xiaohua Zhai, Amelie Royer, Larisa Markeeva, ´ Rohan Anil, and Alexander Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10925–10934, 2022. 1, 5, 20

[6] Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, and Andreas Veit. Understanding robustness of transformers for image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10231–10241, 2021. 8

[7] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In European conference on computer vision, pages 446–461. Springer, 2014. 6

[8] Han Cai, Chuang Gan, and Song Han. Once for all: Train one network and specialize it for efficient deployment. CoRR, abs/1908.09791, 2019. 2

[9] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 782–791, June 2021. 8, 19, 23

[10] Wuyang Chen, Wei Huang, Xianzhi Du, Xiaodan Song, Zhangyang Wang, and Denny Zhou. Auto-scaling vision transformers without training. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 2

[11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. ´ Microsoft COCO captions: Data collection and evaluation server. CoRR, abs/1504.00325, 2015. 15

[12] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4794–4802, 2019. 5

[13] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. CoRR, abs/1604.01685, 2016. 6

[14] Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. The Journal of Machine Learning Research, 13:795–828, 2012. 5

[15] Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, and Yi Tay. The efficiency misnomer. CoRR, abs/2110.12894, 2021. 8, 20

[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at

9

<!-- page 10 of 29 -->

scale. In International Conference on Learning Representations, 2021. 2, 3, 6, 14, 15

[17] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. 8

[18] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231, 2018. 8, 19

[19] Chengyue Gong, Dilin Wang, Meng Li, Xinlei Chen, Zhicheng Yan, Yuandong Tian, Qiang Liu, and Vikas Chandra. Nasvit: Neural architecture search for efficient vision transformers with gradient conflict aware supernet training. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 2

[20] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In CVPR, June 2019. 6, 15

[21] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross B. Girshick. Masked autoencoders ´ are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 15979–15988. IEEE, 2022. 2

[22] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 8320–8329. IEEE, 2021. 8

[23] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 15262–15271. Computer Vision Foundation / IEEE, 2021. 8

[24] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 5

[25] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4904–4916. PMLR, 2021. 6

[26] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollar. Panoptic segmentation. In ´ Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. 6, 15

[27] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby.

Large scale learning of general visual representations for transfer. arXiv preprint arXiv:1912.11370, 2(8), 2019. 15

[28] Alexander Kolesnikov, Andre Susano Pinto, Lucas Beyer, ´ Xiaohua Zhai, Jeremiah Harmsen, and Neil Houlsby. Uvim: A unified modeling approach for vision with learned guiding codes. Advances in Neural Information Processing Systems, 2022. 6, 15

[29] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, pages 3519–3529. PMLR, 2019. 5

[30] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. 15

[31] Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, Univ. Toronto, 2009. 6

[32] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, and Ali Farhadi. Matryoshka representations for adaptive deployment. CoRR, abs/2205.13147, 2022. 2

[33] Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick ´ Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. 2

[34] Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng Gao. ELE-VATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models. In NeurIPS, 2022. 7, 19, 22

[35] Chunyuan Li, Jianwei Yang, Pengchuan Zhang, Mei Gao, Bin Xiao, Xiyang Dai, Lu Yuan, and Jianfeng Gao. Efficient self-supervised vision transformers for representation learning. CoRR, abs/2106.09785, 2021. 2

[36] Mingbao Lin, Mengzhao Chen, Yuxin Zhang, Ke Li, Yunhang Shen, Chunhua Shen, and Rongrong Ji. Super vision transformer. CoRR, abs/2205.11397, 2022. 2

[37] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, 2014. 6

[38] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin Transformer V2: Scaling Up Capacity and Resolution. CoRR, abs/2111.09883, 2021. 2

[39] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022. 2

[40] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil

10

<!-- page 11 of 29 -->

Houlsby. Simple open-vocabulary object detection with vision transformers. In European Conference on Computer Vision. ECCV, 2022. 6, 7, 15, 19

[41] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. arXiv preprint arXiv:2206.02770, 2022. 2

[42] Thao Nguyen, Maithra Raghu, and Simon Kornblith. Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth. In International Conference on Learning Representations, 2021. 5

[43] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, pages 722–729. IEEE, 2008. 6

[44] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pages 3498–3505. IEEE, 2012. 6

[45] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 2641–2649. IEEE Computer Society, 2015. 7, 15

[46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021. 6, 17

[47] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 13937–13949, 2021. 2

[48] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 5389–5400. PMLR, 2019. 8

[49] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large

scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015. 2, 15

[50] Tal Schuster, Adam Fisch, Tommi Jaakkola, and Regina Barzilay. Consistent accelerated inference via confident adaptive transformers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4962–4979, 2021. 8

[51] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A Large-Scale, High-Quality Dataset for Object Detection. In ICCV, pages 8429–8438, 2019. 15

[52] Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A Alemi, and Andrew G Wilson. Does knowledge distillation really work? Advances in Neural Information Processing Systems, 34:6906–6919, 2021. 5

[53] Andreas Peter Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers. Transactions on Machine Learning Research, 2022. 2, 4, 5, 13, 20

[54] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. In ICCV, 2021. 6, 15, 16

[55] Mingxing Tan and Quoc V. Le. Efficientnetv2: Smaller models and faster training. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 10096–10106. PMLR, 2021. 1, 2

[56] Yehui Tang, Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chao Xu, and Dacheng Tao. Patch slimming for efficient vision transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 12155–12164. IEEE, 2022. 2

[57] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, volume 139, pages 10347–10357, July 2021. 2, 5

[58] Hugo Touvron, Matthieu Cord, and Herve J ´ egou. Deit III: ´ revenge of the vit. CoRR, abs/2204.07118, 2022. 2, 8

[59] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herve J ´ egou. Going deeper with im- ´ age transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021. 2

[60] Yulin Wang, Rui Huang, Shiji Song, Zeyi Huang, and Gao Huang. Not all images are worth 16x16 words: Dynamic transformers for efficient image recognition. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 11960–11973, 2021. 2

[61] Alex H Williams, Erin Kunz, Simon Kornblith, and Scott Linderman. Generalized shape metrics on neural representa-

11

<!-- page 12 of 29 -->

derstanding of scenes through the ADE20K dataset. Int. J. Comput. Vis., 127(3):302–321, 2019. 6

tions. Advances in Neural Information Processing Systems, 34:4738–4750, 2021. 5

[62] Chao-Yuan Wu, Ross B. Girshick, Kaiming He, Christoph Feichtenhofer, and Philipp Krahenb ¨ uhl. A multigrid method ¨ for efficiently training video models. CoRR, abs/1912.00998, 2019. 2

[63] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010. 6

[64] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687–10698, 2020. 5

[65] Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 10799–10808. IEEE, 2022. 2

[66] Fang Yu, Kun Huang, Meng Wang, Yuan Cheng, Wei Chu, and Li Cui. Width & depth pruning for vision transformers. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 3143–3151. AAAI Press, 2022. 8

[67] Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans, Mingxing Tan, Thomas S. Huang, Xiaodan Song, Ruoming Pang, and Quoc Le. Bignas: Scaling up neural architecture search with big single-stage models. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part VII, volume 12352 of Lecture Notes in Computer Science, pages 702–717. Springer, 2020. 2

[68] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. MERLOT RESERVE: neural script knowledge through vision and language and sound. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 16354–16366. IEEE, 2022. 2

[69] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104–12113, 2022. 2, 15

[70] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 18102–18112. IEEE, 2022. 6, 15, 17, 19

[71] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic un-

12

<!-- page 13 of 29 -->

![Chart block](images/page013-block01.png)

Figure 15. Options for dealing with patch-embeddings. Supervised training from-scratch on ImageNet-21k. Vanilla (bilinear) is the simplest method and works well. PI-resize further improves the large-patch case and provides other advantages (see text). Untied means learning separate patch-embedding kernels for each size, and does not work too well. Normalize, Token-LN, and Image-LN all but require modifications to the model making it incompatible with standard ViT.

Besides providing more details and results on various sections as mentioned in the main paper, we also provide full numerical (tabular) results of all figures in $\mathbf { A p } -$ pendix R in order to facilitate reproduction/comparison in future work.

Finally, more details about the alternative ways of flexibility discussed in Section 7 are provided in Appendix O.

## A. More details on flexible patch-sizes

In this section, we further elaborate on many details of flexible patch-sizes. We provide results for alternative ways of dealing with flexible patch-sizes when one does not care about preserving model architecture in Appendix A.1. We provide a detailed derivation of PI-resize in Appendix A.2, and show some PI-resize matrices in Appendix A.3. We further show some visualizations of patch-embedding weights, both raw and resized, in Appendix A.4

## A.1. Alternatives for dealing with flexible patch-size

Besides bilinear resizing (called **Vanilla**) or **PI-resizing** the patch-embedding weights to deal with variable patchsizes, there are a few other alternatives which we discuss and compare here.

**Untied** weights for each size, i.e. having separate trainable parameter buffers for each patch-size.

**Untied, =init** is the same as above, but initializing all patch embedding weights to the same (PI-resized) values as a reference initialization. In this setting, the model is still compatible with standard ViT models at initialization time, how-

ever, the parameters can, and do, diverge during training, resulting in a non-standard ViT architecture.

**Normalize** simply l2-normalizes the tokens computed by the patch-embedding individually to unit-norm. This solves any norm-related issues in a simple, parameter-free way, but is incompatible with pre-trained standard ViT models.

**Token-LN and Image-LN** add a LayerNorm [2] right after the patch-embedding and differ only in which axis they perform the normalization. Again, this solves the norm-related issues, but does add learnable parameters and is incompatible with pre-trained standard ViT models.

A comparison of all these variants is performed in the label-supervised training setup on ImageNet-21k following [53], but training for 90 epochs. The result, presented in Figure 15, indicates that plain resizing and PI-resizing are among the best solutions, but have the added benefit of resulting in standard ViT models. Furthermore, not visible in this figure, both Untied variants displayed slightly unstable training curves in the first half of training, while all other variants train smoothly.

## A.2. PI-resize derivation

We can rewrite the objective function in Eq. (2) as follows:

$$
\begin{array}{l} \mathbb {E} _ {x \sim \mathcal {X}} \left[ (\langle x, \omega \rangle - \langle B x, \hat {\omega} \rangle) ^ {2} \right] = \\ \mathbb {E} _ {x \sim \mathcal {X}} \left[ (x ^ {T} (\omega - B ^ {T} \hat {\omega})) ^ {2} \right] = \\ \mathbb {E} _ {x \sim \mathcal {X}} \left[ ((\omega - B ^ {T} \hat {\omega}) ^ {T} x) (x ^ {T} (\omega - B ^ {T} \hat {\omega})) \right] = \\ (\omega - B ^ {T} \hat {\omega}) ^ {T} \mathbb {E} _ {x \sim \mathcal {X}} \left[ x x ^ {T} \right] (\omega - B ^ {T} \hat {\omega}) = \\ \| \omega - B ^ {T} \hat {\omega} \| _ {\Sigma} ^ {2}, \end{array}\tag{6}
$$

where $\| v \| _ { \Sigma } ^ { 2 }   =   v ^ { T } \Sigma v$ and $\Sigma   =   \mathbb { E } _ { x \sim \mathcal { X } } x x ^ { T }$ is the (uncentered) covariance matrix of X . In case when $\mathcal { X } = \mathcal { N } ( 0 , I )$ we recover the standard euclidean norm $\| \boldsymbol { \omega } - B ^ { T } \boldsymbol { \hat { \omega } } \| ^ { 2 }$

Finally, we note that the pseudoinverse matrix recovers a least squares solution to a linear system of equations:

$$
(B ^ {T}) ^ {+} \omega \in \arg \min _ {\hat {\omega}} \| \omega - B ^ {T} \hat {\omega} \| ^ {2}.\tag{7}
$$

We can also derive an analytic solution for an arbitrary $\Sigma ~ = ~ \mathbb { E } _ { x \sim \mathcal { X } } x x ^ { T }$ . Note that $\| v \| _ { \Sigma } ^ { 2 } \quad = \quad v ^ { T } \Sigma v \quad =$ $( \sqrt { \Sigma } v ) ^ { T } \sqrt { \Sigma } v = \| \sqrt { \Sigma } v \| ^ { 2 }$ . Then, we have

$$
\| \omega - B ^ {T} \hat {\omega} \| _ {\Sigma} ^ {2} = \| \sqrt {\Sigma} \omega - \sqrt {\Sigma} B ^ {T} \hat {\omega} \| ^ {2}.\tag{8}
$$

The optimal solution is then given by

$$
(\sqrt {\Sigma} B ^ {T}) ^ {+} \sqrt {\Sigma} \omega \in \arg \min _ {\hat {\omega}} \| \omega - B ^ {T} \hat {\omega} \| ^ {2}.\tag{9}
$$

## A.3. Visualization of some PI-resize matrices

We visualize an upscaling and a downscaling matrix for both bilinear and PI-resize operations for a visual comparison in Figure 16.

13

<!-- page 14 of 29 -->

![Image block](images/page014-block01.png)

Figure 16. Visualization of resize matrices. Each row corresponds to the weights that are used in the computation of one output pixel. Off-diagonals correspond to pixels below/above the current one. We can see that PI-resize does include negative weights, has a larger receptive field, and uses overall larger weights than bilinear resizing.

## A.4. Visualization of patch-embedding weights

PCA of patch embeddings (see [16]) in Figs. 33 to 37.

## B. The “underlying” parameter shapes

FlexiViT does introduce two new hyper-parameters which were not present in the original ViT architecture:

the size of the underlying patch-embedding weights and position-embeddings (i.e. learned params, before resize).

However, both of these parameters have (maybe surprisingly) little influence on the final performance of the model, as long as they are in a “reasonable” range. We verified these in two different settings.

**For the patch-size parameter,** since it is affected by the resize method used, we perform the ablation in the full Flexi-ViT setup. Figure 17 (b) shows that there is no notable difference across all evaluation sizes, and hence we stick to the (initially arbitrary) default of 32 across all experiments.

**For the position embedding parameter,** we ran an early experiment with ViT-S/16 trained from scratch on ImageNet-1k following [4]. Figure 17 (c) shows that in general, even for plain ViT training, this approach could be taken and training curves are mostly unaffected by in-graph bilinear resizing of position embeddings.

## C. Distribution of patch-size sampling

During roughly the first half of this project, we sampled patch-size from a distribution which is not uniform, but samples patch-sizes between 16 and 30 up to three times more than patch-sizes outside this range. This “triangular” distribution was based purely on gut-feeling, and once we verified that it is no better than uniform sampling (the experiment is shown in Figure 17 (a)), we decided to use a uniform distribution for simplicity. We further decided to avoid the costly re-running of all experiments we did so far, and thus some experiments in the main paper were done with the “triangular” distribution. However, in all direct comparisons presented throughout the paper, the curves being directly compared always were trained in the exact same way.

![Chart block](images/page014-block13.png)

![Chart block](images/page014-block14.png)

![Chart block](images/page014-block15.png)

Figure 17. Ablation of a few unimportant hyperparameters, since changing their value shows no noteworthy difference in the result. (a) Sampling distribution of patch sizes. Uniform is preferable for its simplicity, but we use “Triangular” (more weight on mid-sized patches, less weight on extremely large or small ones) sometimes for legacy reasons. (b) Varying the shape of the underlying patch embedding parameter in the full FlexiViT training setup. (c) Varying the shape of the underlying position embeddings in a smaller supervised training of a ViT-S/16 baseline following [4].

14

<!-- page 15 of 29 -->

![Chart block](images/page015-block01.png)

![Chart block](images/page015-block02.png)

![Chart block](images/page015-block03.png)

![Chart block](images/page015-block04.png)

![Chart block](images/page015-block05.png)

![Chart block](images/page015-block06.png)

Figure 18. More transfer results. The legend is the same as in Figure 7. The main take-away is that the results are qualitatively the same across a wide range of image classification tasks and it is safe to use a pre-trained FlexiViT model in lieu of a ViT model even when one only uses it at a fixed patch size after transfer.

## D. Detais on resource-efficient transfer

For finetuning FlexiViT models on the Imagenet-1k dataset we generally follow the transfer learning setup from [16]. We use SGD momentum optimizer, with the initial learning rate of 0.03 and cosine learning rate decay. We also reduce the learning rate for the pretrained parameters by a factor of 10. We optimize for 20000 steps with inception crop and flip left-right augmentation, using batch size 512 and input image size of $4 8 0 \times 4 8 0$

## E. Using pre-trained FlexiViT models: more details and results

In this section, we provide more details and full results for the scenario described in Section 4: using pre-trained FlexiViT models.

For more details on flexified training procedures discussed in Section 5, we redirect to Appendix M for flexified transfer learning, Appendix G for flexified contrastive image-text learning (LiT and CLIP), and Appendix N for flexified open-vocabulary detection (OWL-ViT), including results on ELEVATER.

## E.1. Using FlexiViT models for transfer

For transfer, we follow the simple BiT-HyperRule [27]. In short, we transfer for a relatively brief number of steps (500 for flowers and pets, 2500 for food101 and sun, and 10 000 for CIFAR), using the SGD optimizer with a momentum of 0.9, no weight decay, no dropout, and no other augmentations besides flips and random crops. We initialize the new classification layer to all-zeros and use a short learning-rate warmup, both of these having as effect to better preserve the pre-trained weights. The only setting which differs from [27] is that we do sweep the learning-rate across {0.03, 0.01, 0.003, 0.001} for each task individually. We show results for all 6 datasets we used in Figure 18.

## E.2. Using FlexiViT models in LiT for image-text tasks

We use the same 4B image-text pairs dataset as in [70] to train the LiT models, and use identical hyper parameters as LiT models. The only difference is to use the Flexi-

ViT model here, instead of a standard ViT model. FlexiViT models are transferred at a fixed sequence length, i.e. $3 0 ^ { 2 }$ or $1 6 ^ { 2 }$ , with $2 4 0 \times 2 4 0$ image resolution. We report zero-shot classification results on ImageNet [49], zero-shot image-to-text / text-to-image retrieval results on MS-COCO [11] and Flickr30K [45], in Figure 19.

## E.3. Using FlexiViT models in OWL-ViT for zero-shot detection

To evaluate FlexiViT backbones for open-vocabulary detection (Figure 7), we compare OWL-ViTs initialized with either fixed or flexible LiT-B models. Specifically, the backbones start with either a fixed or flexibly pre-trained ViT image model, which is then then frozen and LiT-tuned [70] with a text model at a fixed patch size (the same as final evaluation, i.e. $3 0 ^ { 2 }$ and $1 6 ^ { 2 }$ respectively). OWL-ViTs using these backbones are then trained at a fixed patch size $( 3 0 ^ { \bar { 2 } }$ or $1 6 ^ { 2 } )$ at a resolution of $7 2 0 \times 7 2 0$ on Objects365 [51] and Visual Genome [30] as in the original paper [40]. We report mean average precision (AP) on LVIS [20].

## E.4. Using FlexiViT models in UViM for panoptic segmentation

Apart from using FlexiViT model weights for the initialization, we follow the setup of the original UViM setup [28] as close as possible. In particular, we train the model on the COCO panoptic dataset [11] and report the standard PQ metric [26] on the official validation split. We train the model for 200 epochs, using the custom adafactor optimizer variant [69] with the base learning rate of 0.001. The learning rate for the pretrained part of the model is decreased by a factor of 10. The input image size is $5 1 2 \times 5 1 2$ . More details on the training setting can be found in the UViM paper [28] and official repository of the UViM model <sup>6</sup>.

## E.5. Using FlexiViT models in Segmenter for semantic segmentation

We follow the experimental setup of Segmenter [54] for end-to-end finetuning of Vision Transformer with linear decoder. For data augmentation during training, we apply ran-

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>6</sup>[https://github.com/google-research/big\_vision/ tree/main/big\_vision/configs/proj/uvim](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/uvim)</span></small>

15

<!-- page 16 of 29 -->

![Chart block](images/page016-block01.png)

![Chart block](images/page016-block02.png)

![Chart block](images/page016-block03.png)

![Chart block](images/page016-block04.png)

![Chart block](images/page016-block05.png)

Figure 19. Transfer of FlexiViT at a fixed patch size. The LiT-ViT baselines match LiT-FlexiViT on the sequence length it has been trained for $( 1 6 ^ { 2 } \: \mathrm { o r } \: 3 0 ^ { 2 } )$ , but performance drops quickly when using a different inference sequence length. We observe that the LiT-FlexiViT models work well across different inference sequence lengths, even though only a fixed sequence length is used during LiT transfer. This effect is similar to that described in Section 4.2 and further explored in Figure 20.

![Chart block](images/page016-block07.png)

![Chart block](images/page016-block08.png)

![Chart block](images/page016-block09.png)

![Chart block](images/page016-block10.png)

![Chart block](images/page016-block11.png)

Figure 20. FlexiLiT. We observe consistent but marginal boost when flexifying the LiT training by randomizing patch size during LiTtuning. This again shows that the LiT-FlexiViT baseline performs strongly and it allows fast transfer: transferred cheaply using a large patch size and served at smaller patch sizes for free.

![Chart block](images/page016-block13.png)

![Chart block](images/page016-block14.png)

![Chart block](images/page016-block15.png)

![Chart block](images/page016-block16.png)

![Chart block](images/page016-block17.png)

Figure 21. FlexiCLIP. We observe very similar conclusions as in FlexiViT, that the FlexiCLIP model works well across a large range of evaluation patch sizes during inference. It is interesting that a single fixed sequence length text tower, is able to produce embeddings aligning well with a FlexiViT image tower that allows multiple sequence lengths.

dom resizing of the image with a random ratio between 0.5 and 2.0, photometric augmentation and random horizontal flipping. We randomly crop images to 480 × 480 resolution with padding, therefore preserving aspect ratio. We use the 480 × 480 resolution for both Cityscapes and ADE20k. We train for 127 epochs with minibatch size of 16 (resulting

in 160k iterations on ADE20k). We use the “poly” learning rate decay schedule and sweep the base learning rate in $\{ 1 e - 4 , 3 e - 4 , 8 e - 4 \}$ for all of our runs. Weight-decay is kept fixed at 0.01. At evaluation time, we use the slidingwindow with a resolution 480×480 to handle varying image sizes during inference. Table 3 row 6 in [54] reports 48.06

16

<!-- page 17 of 29 -->

![Chart block](images/page017-block01.png)

Figure 22. A detailed look into the impact of the larger patch size (size of the markers), schedule, and ramp periods on both compute and accuracy. See Appendix H for details.

mIoU. Average of 6 runs in our codebase in the same setting gives 47.6±0.4 mIoU. The results on ADE20k are provided in Table 7.

## F. Full numerical ImageNet-1k-only results

We provide full numerical results of the FlexiViTs trained purely on ImageNet-1k and presented in Figure $^ { 2 , }$ including on additional robustness and OOD test-sets in Tabs. 1 to 4.

## G. FlexiLiT and FlexiCLIP results

FlexiLiT follows exactly the same setup as described in Section E.2, but randomizes patch sizes during LiT training. We show more FlexiLiT results in Figure 20.

For FlexiCLIP, we simply replace the pre-trained and frozen backbone in FlexiLiT with a random initialized and unfrozen backbone, which corresponds to the uu setting in [70] and is equivalent to CLIP [46]. Figure 21 shows the same conclusions in this setting. It is reassuring that the patch size randomization does not hinder learning both image and text representations from scratch simultaneously.

## H. Accelerate pre-training

As discussed in Section 5.4, one can use FlexiViT’s method of varying the patch size and resizing the embedding weights to pretrain ViTs faster. To do that, we specify a curriculum: a sequence $( p _ { k } ) _ { k = 1 } ^ { K }$ of probability distributions over the patch sizes along with a mapping $c   :   \mathbb { N }   \to   [ K ]$ that identifies which distribution $p _ { k }$ to use at training step t. In this section, we use the sequence of patch sizes: (48, 40, 30, 24, 16) to demonstrate the potential savings in compute.

![Chart block](images/page017-block11.png)

Figure 23. This figure illustrates how the distribution of patch sizes changes when accelerating pretraining (see Section 5.4) and $\mathbf { A } \mathbf { p } \mathbf { \cdot }$ pendix H. Here, the first 75% of training steps use the larger patch size $p _ { 1 }$ while the remaining 25% are used for the desired/target patch size. The ramp period in this figure is 40%.

We set $1 6 \times 1 6$ to be the desired/target patch size during evaluation. Hence, we compare a curriculum-based approach of pretraining ViTs (denoted FasterViT) with the standard ViT/B/16 architecture in which the patch size is fixed to 16 × 16 throughout training. Except for the variable patch sizes and the embedding layers, both architecture are otherwise identical.

We use a simple curriculum in this evaluation. Specifically, we have one large patch size (e.g. $4 8 \times 4 8 )$ , which we denote by $p _ { 0 }$ and the desired patch size $1 6 \times 1 6 .$ , which we denote by $p _ { 1 }$ . We initially use the larger patch size before swtiching to the smaller patch size using FlexiViT’s PI-resize. We experiment with three different schedules: (50%, 75%, 90%), where schedule = 75% means that 75% of the training time uses the larger patch size alone. Instead of switching immediately between patch sizes, we also include an optional ramp period as illustrated in Figure 23. A ramp period of, say, 40% means that 40% of the time allocated to the smaller patch size is used to transition gradually between the two distributions. We experiment with three ramp periods: (0%, 50%, 100%). We run each experiment independently and plot the resulting compute and accuracy in Figure 12(x). As shown in the figure, FasterViT achieves the same level of accuracy as standard ViTs but with less compute, although the improvement in not quite significant.

In Figure 22, we provide a detailed view into the impact of the three hyperparameters: patch size, schedule, and ramp period. Not surprisingly, increasing the fraction of time allocated to the larger patch size (e.g. by setting schedule = 90%), improves compute at the expense of accuracy.

## I. Further analysis of cosine similarities between token representations across scales

In Section 6, we measure cosine similarity between the representation of a seed token at one scale and representations of other tokens at other scales, demonstrating that the

17

<!-- page 18 of 29 -->

![Image block](images/page018-block01.png)

Figure 24. Supplemental analysis of similarities of token representations across scales. Each row shows the cosine similarity between a seed token at the center of the feature map at one scale and tokens at different scales, for a single block. In the top group of plots, the seeds are taken from FlexiViT-B/48; in the middle, from FlexiViT-B/16; and at the bottom, from FlexiViT-B/8.

most similar tokens at other scales are those that represent the same spatial location. in Figure 24, we provide additional results for seed tokens at other grid sizes and from additional blocks. Results are consistent with those in the main text.

## J. Ensembling FlexiViT predictions across scales

We ensemble FlexiViT models by averaging models logits.<sup>7</sup> When ensembling all models, we attain 51.7% pre-cision@1 on our ImageNet-21K validation set, which is slightly worse than the accuracy achieved at the largest grid size/smallest patch size (52.0%).

We further explore ensembles of pairs of models in Figure 26. Agreement between models evaluated at large patch sizes is relatively low, with models evaluated at the largest two patch sizes (/48 and /40) agreeing on only 67.4% of examples (Figure 26 middle). Nonetheless, ensembling these models provides no accuracy improvement over simply using FlexiViT-B/40; both strategies achieve 45.8% ImageNet-21K precision@1 (Figure 26 left).

When comparing the computational cost of ensembles of FlexiViT predictions across scales to applying Flexi-ViT at a single scale, a single scale is nearly always better (Figure 26 right). The only configuration where the accuracy of a two-scale ensemble exceeds the accuracy of a single scale with the same computational footprint is the

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>7</sup>We have also explored ensembling based on averaging output probabilities. We find that results are nearly identical, but on average slightly worse.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">80 Shape accuracy 70 Texture accuracy 60 Shape bias 50 ★ 40 ★ ★ ★ 30 20 10 /48 /40 /30 /24 /20 /16 /15 /12 /10 /8 Patch size</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">Figure 25. Shape and texture bias of FlexiViT. Lines reflect values for FlexiViT evaluated at different scales; stars reflect values for baseline ViT models trained at a single scale. Shape and texture accuracy are the top-1 accuracy of the model’s prediction with respect to the shape and texture labels. Shape bias is the percentage of images that a classifier classifies by shape, provided that it correctly classifies them by either shape or texture.</span></small>

18

<!-- page 19 of 29 -->

![Chart block](images/page019-block01.png)

![Chart block](images/page019-block02.png)

![Chart block](images/page019-block03.png)

Figure 26. Ensemble accuracy and agreement of FlexiViT-B evaluated at pairs of scales. Left: Accuracy of ensembles between scales. Middle: Agreement of the top-1 predicted classes across scales. Right: Accuracy of single-scale configurations and two-scale ensembles versus computational cost. Although agreement is relatively low at small patch sizes, there is little benefit to ensembling.

ensemble of FlexiViT-B/10 and /12, which together have a similar computational cost to FlexiViT-B/8 $( / 1 0   +   / 1 2 ;$ 186.8 GFLOPs; /8: 184.5 GFLOPs). The ensemble attains marginally higher accuracy (52.1% vs. 52.0%), but this improvement is unlikely to be statistically significant nor practically meaningful.

## K. Shape and texture bias of FlexiViT

When confronted with images with conflicting shape and texture, ImageNet-trained models tend to produce labels that match their textures, whereas humans instead tend to assign labels that match their shapes [18]. We evaluated FlexiViT using the same dataset as [18], which was generated using the style transfer. In the dataset constructed by [18], images have both shape and texture labels. We define the shape accuracy as the percentage of images for which the top-1 prediction matches the shape label, and texture accuracy as the percentage of images for which the top-1 prediction matches the texture label. As in [18], we define shape bias by taking the ratio of the number of images classified according to their shape label to the number of images classified correctly according to either the shape or texture label and converting this ratio to a percentage. In other words, shape bias the ratio of shape accuracy to the sum of shape and texture accuracy ×100%. To evaluate ImageNet-21K models on the dataset of [18], we use the mapping from WordNet IDs to the dataset classes provided by [18]. We take the top-1 class among the ImageNet-21K classes for which a mapping exists.

In Figure 25, we show that larger patch sizes lead to greater shape bias compared to smaller patch sizes. However, larger patch sizes have greater shape bias primarily because their texture accuracy is lower, rather than because their shape accuracy is higher. The shape biases of FlexiViT-B/16 and FlexiViT-B/30 are similar to the shape biases of ViT-B/16 and ViT-B/30 models trained at a single scale (stars).

## L. Attention relevances

We provide the attention relevance maps [9] for the same image as shown in Figure 13 in the main paper, but for all three classes present in the image, in Figure 28.

We further provide the attention relevance maps for a random selection of 10 royalty-free images obtained from unsplash.com for the class that was predicted by all models in Figure 32 at the end of the Appendix.

## M. Flexifying transfer-learning

When flexifying transfer-learning, we run the exact same setup as when transferring pre-trained FlexiViT models, described in Section E.1, except that we now randomize the patch size during transfer too.

This minor change shows good synergy when combined with using a pre-trained FlexiViT model (green bars), and even enables flexifying plain ViT models during transfer to some degree (orange and olive bars).

## N. Flexifying open-vocabulary detection (OWL-ViT)

For flexifying OWL-ViT (Section 5.3), we use LiT-uu backbones [70], i.e. CLIP-style models in which the image and text encoders are contrastively pre-trained together (as in the OWL-ViT paper [40]). We flexify detection training as described in Algorithm 1 and use resolution $7 2 0 \times 7 2 0$ For flexible detection training, we use patch sizes from $4 8 ^ { 2 }$ to $1 2 ^ { 2 }$ , i.e. omitting $1 0 ^ { 2 }$ and $8 ^ { 2 }$ , which would use excessive memory at the higher resolution. Other training settings are as in the OWL-ViT paper [40]. We find that flexifying both the image-text pre-training and the detection training (Figure 11, pale green line) works slightly better than flexifying just the detection training.

For evaluating on the ELEVATER [34] set of datasets, we use the model for which both image-text pre-training and detection training were flexified (Figure 31).

19

<!-- page 20 of 29 -->

![Chart block](images/page020-block01.png)

![Chart block](images/page020-block02.png)

![Chart block](images/page020-block03.png)

![Chart block](images/page020-block04.png)

Inference patch size

![Chart block](images/page020-block06.png)

![Chart block](images/page020-block07.png)

Figure 27. More transfer results. The legend is the same as in Figure 9. The main take-away is that the results are qualitatively the same across a wide range of image classification tasks.

![Image block](images/page020-block09.png)

Figure 28. Attention relevance maps of same example as in Figure 13, with respect to three different objects present in the image (“caffe au lait”, “knife” and “fork”).

## O. Details on flexible depth, stride alternatives

**Common setup** The setup largely follows that introduced in Section 3.5: distilling the ViT-B/8 model from [53] while simultaneously using it for initialization of the student. Distillation is performed on ImageNet-21k, the labels are ig-

![Chart block](images/page020-block13.png)

Figure 29. Fig 2 using GFLOPs.

nored and the KL-divergence between student and teacher is the only loss, following [5]. Training is performed for 90 epochs with uniform sampling of patch sizes.

**Flexible stride** We train a FlexiViT model variant, where we flexibly change window stride when extracting image patches, but keep the patch size fixed at $3 2 \times 3 2$ . In order to perfectly match default grid sizes, and perfectly cover the whole image and avoid padding, we perform minimally required image resize. For example, to get grid size $8 \times 8$ we resize the image to size $2 4 2 \times 2 4 2$ (from $2 4 0 \times 2 4 0 )$ and apply stride 30, or to get grid size 24 × 24 we resize the image to size 239 × 239 and apply stride 9.

**Flexible depth** For every batch, we sample a depth d uniformly from {3, 5, 9, 12} and perform a forward pass up to layer d. We then apply the classification head, which is shared across all depths, to the class token of layer d and compute the loss. We also explored sampling a depth perexample, but this led to unstable training except when sampling d from all layers $\{ 3 , 4 , \ldots , 1 2 \}$ . Finally, we also tried using a head per depth as well as a class token per depth, which did not lead to any significant improvement.

## P. Figure 2 with GFLOPs

Following the Efficiency Misnomer [15], we also provide a copy of Figure 2 using GFLOPs as the x-axis instead

20

<!-- page 21 of 29 -->

![Chart block](images/page021-block01.png)

Figure 30. Patch-size extrapolation. We evaluate the model trained on the green patch-sizes at patch-sizes outside the range that was seen during training, and plot these in pink. The performance slowly deteriorating means that the model is not able to extrapolate beyond patchsizes or sequence lengths seen during training.

of inference time as Figure 29. This confirms that FLOPs do not always directly translate to wall-clock time in all situations.

## Q. Extrapolating patch-size

We take the model from the main paper’s Section 3.3 and run inference at even smaller patchsizes, see Figure 30. We observe that performance slowly starts to deteriorate. This means that the model does not learn to generalize to patch-sizes or sequence lengths beyond those seen during the training. Note that we do not claim extrapolation capabilities, but rather that training many patchsizes into a single model works without loss of quality.

## R. Full tabular results

We provide Tabs. 1 to 13 which contain the numerical results from all plots from the main paper.

## S. Configuration file for Fig 2

Algorithm S shows the big vision<sup>8</sup>config for training the FlexiViT models from Figure 2.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>8</sup>[https://github.com/google-research/big\_vision](https://github.com/google-research/big_vision)</span></small>

21

<!-- page 22 of 29 -->

![Chart block](images/page022-block01.png)

Figure 31. Inference-time tuning of patch size can improve performance. A single OWL-FlexiViT-B model is evaluated at two different patch sizes $( 3 0 ^ { 2 }$ and $1 6 ^ { 2 } )$ on the $3 5$ different detection tasks of the ELEVATER benchmark. [34]. Tasks are are ordered by the performance difference between patch size $3 0 ^ { 2 }$ and $1 6 ^ { 2 }$ on the validation set; the y-axis shows performance on the test set. The optimal patch size varies by task, such that tuning the patch size on the validation set improves mean test performance from 15.63 AP (at patch size $1 6 ^ { 2 } )$ or 16.19 $\mathbf { A P }$ (at patch size $3 0 ^ { 2 } )$ to 16.62 AP.

22

<!-- page 23 of 29 -->

![Image block](images/page023-block01.png)

Figure 32. Attention relevance (as in [9]) can significantly change at different patch sizes for both ViT and FlexiViT.

23

<!-- page 24 of 29 -->

![Image block](images/page024-block01.png)

Figure 33. First 80 PCA components of the raw underlying 32x32 patch embedding weights of FlexiViT.

![Image block](images/page024-block03.png)

Figure 34. First 80 PCA com- Figure 35. First 80 PCA components of the patch embeddingponents of the patch embedweights from Figure 33 bilin- ding weights from Figure 33 PI-early resized to 8x8. resized to 8x8.

![Image block](images/page024-block05.png)

Figure 36. First 80 PCA components of the patch embedding weights from Figure 33 bilinearly resized to 48x48.

![Image block](images/page024-block07.png)

Figure 37. First 80 PCA components of the patch embedding weights from Figure 33 PI-resized to 48x48.

24

<!-- page 25 of 29 -->

```python
def get_config(arg=None):
    """Config for training FlexiViT on ImageNet1k.
    """
    c = bvcc.parse_arg(arg, variant='B')
    c.total_epochs = 90
    c.num_classes = 1000
    c.loss = 'softmax_xent'

    c.input = {}
    c.input.data = dict(
        name='imagenet2012',
        split='train[:99%]',
    )
    c.input.batch_size = 1024
    c.input.shuffle_buffer_size = 250_000

    c.log_training_steps = 50
    c.ckpt_steps = 1000

    # Model section
    c.student_name = 'proj.flexi.vit'
    c.student_init = f'deit_3_{c.variant}_384_1k'
    c.student = dict(variant=c.variant,
                             pool_type='tok',
                             patch_size=(16, 16))

    c.teachers = ['prof']
    c.prof_name = 'vit'
    c.prof_init = f'deit_3_{c.variant}_384_1k'
    c.prof = dict(variant=c.variant,
                             pool_type='tok',
                             patch_size=(16, 16))

    pp_label = (
        '|onehot(1000, key="{lbl}", key_result="
        labels)'
        '|keep("image", "prof", "labels")')
    c.input.pp = (
        'decode|inception_crop|flip_lr'
        '|copy("image", "prof")'
        '|resize(240)'
        '|vgg_value_range'
        '|resize(384, key="prof")'
        '|vgg_value_range(key="prof")'
        + pp_label.format(lbl='label'))
    pp_eval_both = (
        'decode|copy("image", "prof")'|'
        f'|resize({240//7*8})'
        '|central_crop(240)'
        '|vgg_value_range'
        f'|resize({384//7*8}, key="prof")'
        '|central_crop(384, key="prof")'
        '|vgg_value_range(key="prof")'|')
    pp_eval_student = (
        'decode'
        f'|resize({240//7*8})|central_crop({240})'
        '|value_range(-1, 1)')
    pp_eval_prof = (
        'decode'
        f'|resize({384//7*8})|central_crop(384)'
        '|vgg_value_range(outkey="prof")')

    # Distillation settings
    c.mixup = dict(p=1.0, n=2)
    c.distance = 'kl'
    c.distance_kw = dict(t=1.0)
```

```python
# Optimizer section
c.grad_clip_norm = 1.0
c.optax_name = 'scale_by_adam'
c.optax = dict(mu_dtype='bfloat16')

c.lr = 1e-4
c.wd = 1e-5
c.schedule = dict(
    warmup_steps=5000,
    decay_type='cosine')

# Define the flexible model params:
c.flexi = dict()
c.flexi.seqhw = dict(
    # The settings to sample from.
    # Corresponding patch-sizes at 240px:
    # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
    v=(5, 6, 8, 10, 12, 15, 16, 20, 24, 30),
    # The probs/weights of them (uniform):
    p=(1, 1, 1, 1, 1, 1, 1, 1, 1),
)

###
# All the rest is just evaluations.
minitrain = 'train[:2 %-]
minival = 'train[99%:]'

def get_eval(s, split, dataset='imagenet2012'):
    return dict(
        type='classification',
        pred=f'student_seqhw={s}', 
        data=dict(name=dataset, split=split),
        pp_fn=(pp_eval_student +
            pp_label.format(lbl='label')), 
        loss_name='sigmoid_xent',
        log_percent=0.05,
        cache_final=False,
    )

c.evals = {}
for s in c.flexi.seqhw.v:
    c.evals[f'student_minitrain_{s:02d}'] = \
        get_eval(s, minitrain)
    c.evals[f'student_minival_{s:02d}'] = \
        get_eval(s, minival)
    c.evals[f'student_val_{s:02d}'] = \
        get_eval(s, 'validation')
    c.evals[f'student_v2_{s:02d}'] = \
        get_eval(s, 'test', 'imagenet_v2')
    c.evals[f'student_a_{s:02d}'] = \
        get_eval(s, 'test', 'imagenet_a')
    c.evals[f'student_r_{s:02d}'] = \
        get_eval(s, 'test', 'imagenet_r')
    c.evals[f'student_real_{s:02d}'] = \
        get_eval(s, 'validation',
            'imagenet2012_real')
    c.evals[f'student_real_{s:02d}'.pp_fn = (
        pp_eval_student +
        pp_label.format(lbl='real_label'))

# A bunch more evals here ...

return c
```

25

<!-- page 26 of 29 -->

Table 1. Scores for 1200ep ImageNet-1k-only runs from Figure 2.

| Model | Eps PS | Val ReaL v2 | -A | -R |
| --- | --- | --- | --- | --- |
| FlexiViT-S | 1200 48² | 69.6 76.1 55.5 | 3.3 | 24.1 |
| FlexiViT-S | 1200 40² | 73.7 80.2 60.3 | 4.7 | 26.4 |
| FlexiViT-S | 1200 30² | 78.1 84.3 65.1 | 7.4 | 29.2 |
| FlexiViT-S | 1200 24² | 80.5 86.3 68.4 | 9.4 | 30.5 |
| FlexiViT-S | 1200 20² | 81.6 87.2 70.3 | 12.2 | 31.8 |
| FlexiViT-S | 1200 16² | 82.5 87.9 71.7 | 15.0 | 31.4 |
| FlexiViT-S | 1200 15² | 82.7 88.1 71.8 | 15.7 | 32.9 |
| FlexiViT-S | 1200 12² | 83.2 88.4 72.7 | 17.8 | 32.9 |
| FlexiViT-S | 1200 10² | 83.2 88.4 72.9 | 19.4 | 33.0 |
| FlexiViT-S | 1200 8² | 83.3 88.5 72.9 | 19.3 | 32.6 |
| FlexiViT-B | 1200 48² | 75.0 80.5 61.1 | 5.7 | 28.1 |
| FlexiViT-B | 1200 40² | 78.0 83.3 64.7 | 7.5 | 30.0 |
| FlexiViT-B | 1200 30² | 81.6 86.5 69.7 | 11.4 | 33.0 |
| FlexiViT-B | 1200 24² | 83.2 87.7 71.7 | 15.5 | 34.5 |
| FlexiViT-B | 1200 20² | 84.0 88.4 73.0 | 18.4 | 35.6 |
| FlexiViT-B | 1200 16² | 84.7 88.8 74.0 | 21.7 | 35.8 |
| FlexiViT-B | 1200 15² | 84.7 88.8 74.3 | 22.7 | 36.7 |
| FlexiViT-B | 1200 12² | 84.9 89.1 74.8 | 25.3 | 37.1 |
| FlexiViT-B | 1200 10² | 85.2 89.2 75.0 | 26.7 | 37.2 |
| FlexiViT-B | 1200 8² | 85.1 89.2 74.9 | 27.1 | 37.2 |
| FlexiViT-L | 1200 48² | 77.8 83.3 63.9 | 7.1 | 30.0 |
| FlexiViT-L | 1200 40² | 80.4 85.6 67.2 | 9.9 | 32.7 |
| FlexiViT-L | 1200 30² | 83.2 87.9 70.8 | 14.8 | 35.9 |
| FlexiViT-L | 1200 24² | 84.5 88.8 73.6 | 19.9 | 38.1 |
| FlexiViT-L | 1200 20² | 85.1 89.4 74.9 | 23.5 | 39.4 |
| FlexiViT-L | 1200 16² | 85.7 89.7 76.0 | 28.6 | 39.6 |
| FlexiViT-L | 1200 15² | 85.8 89.9 76.0 | 29.1 | 40.6 |
| FlexiViT-L | 1200 12² | 86.0 90.0 76.5 | 32.0 | 40.8 |
| FlexiViT-L | 1200 10² | 86.0 90.0 76.8 | 33.6 | 40.9 |
| FlexiViT-L | 1200 8² | 86.1 90.0 76.7 | 34.1 | 41.2 |

Table 2. Scores for 600ep ImageNet-1k-only runs from Figure 2.

| Model | Eps | PS | Val ReaL v2 | -A | -R |
| --- | --- | --- | --- | --- | --- |
| FlexiViT-S | 600 | 48² | 68.6 75.1 54.2 | 3.2 | 23.9 |
| FlexiViT-S | 600 | 40² | 72.7 79.4 59.3 | 4.4 | 26.3 |
| FlexiViT-S | 600 | 30² | 77.6 83.8 64.6 | 6.9 | 29.0 |
| FlexiViT-S | 600 | 24² | 80.2 86.0 67.8 | 9.2 | 30.4 |
| FlexiViT-S | 600 | 20² | 81.4 87.0 69.9 | 11.5 | 31.5 |
| FlexiViT-S | 600 | 16² | 82.3 87.7 71.2 | 14.2 | 31.2 |
| FlexiViT-S | 600 | 15² | 82.5 87.9 71.5 | 15.1 | 32.7 |
| FlexiViT-S | 600 | 12² | 83.1 88.3 72.5 | 17.5 | 32.7 |
| FlexiViT-S | 600 | 10² | 83.3 88.5 72.7 | 19.5 | 32.7 |
| FlexiViT-S | 600 | 8² | 83.3 88.5 72.8 | 19.4 | 32.4 |
| FlexiViT-B | 600 | 48² | 74.1 80.0 60.2 | 5.3 | 27.7 |
| FlexiViT-B | 600 | 40² | 77.5 83.0 64.4 | 7.5 | 30.0 |
| FlexiViT-B | 600 | 30² | 81.1 86.1 68.9 | 11.2 | 32.7 |
| FlexiViT-B | 600 | 24² | 82.9 87.5 71.6 | 15.0 | 34.2 |
| FlexiViT-B | 600 | 20² | 83.9 88.2 72.6 | 17.5 | 35.4 |
| FlexiViT-B | 600 | 16² | 84.6 88.7 73.9 | 22.1 | 35.7 |
| FlexiViT-B | 600 | 15² | 84.7 88.8 73.9 | 22.6 | 36.6 |
| FlexiViT-B | 600 | 12² | 84.9 89.0 74.7 | 25.4 | 36.9 |
| FlexiViT-B | 600 | 10² | 85.1 89.2 74.8 | 26.9 | 37.2 |
| FlexiViT-B | 600 | 8² | 85.0 89.2 74.8 | 27.0 | 36.9 |
| FlexiViT-L | 600 | 48² | 77.1 82.6 62.7 | 7.2 | 30.1 |
| FlexiViT-L | 600 | 40² | 80.1 85.2 66.6 | 9.4 | 32.6 |
| FlexiViT-L | 600 | 30² | 83.0 87.7 71.0 | 14.6 | 36.0 |
| FlexiViT-L | 600 | 24² | 84.4 88.8 73.4 | 19.3 | 38.1 |
| FlexiViT-L | 600 | 20² | 85.1 89.4 74.7 | 22.5 | 39.3 |
| FlexiViT-L | 600 | 16² | 85.6 89.7 76.0 | 27.7 | 39.7 |
| FlexiViT-L | 600 | 15² | 85.7 89.8 75.9 | 28.3 | 40.8 |
| FlexiViT-L | 600 | 12² | 85.9 89.9 76.4 | 31.0 | 41.0 |
| FlexiViT-L | 600 | 10² | 86.1 90.0 76.6 | 32.8 | 41.1 |
| FlexiViT-L | 600 | 8² | 86.1 90.0 76.6 | 33.2 | 41.3 |

26

<!-- page 27 of 29 -->

Table 3. Scores for 300ep ImageNet-1k-only runs from Figure 2.

| Model | Eps | PS | Val ReaL v2 | -A | -R |
| --- | --- | --- | --- | --- | --- |
| FlexiViT-S | 300 | 48² | 67.4 74.0 53.2 | 2.8 | 23.5 |
| FlexiViT-S | 300 | 40² | 71.9 78.6 58.2 | 3.9 | 26.0 |
| FlexiViT-S | 300 | 30² | 77.2 83.5 64.2 | 6.5 | 29.0 |
| FlexiViT-S | 300 | 24² | 79.7 85.6 67.6 | 8.8 | 30.2 |
| FlexiViT-S | 300 | 20² | 81.1 86.7 69.6 | 11.1 | 31.4 |
| FlexiViT-S | 300 | 16² | 82.1 87.6 71.1 | 13.9 | 31.1 |
| FlexiViT-S | 300 | 15² | 82.4 87.9 71.3 | 14.9 | 32.7 |
| FlexiViT-S | 300 | 12² | 83.0 88.2 72.3 | 17.6 | 32.5 |
| FlexiViT-S | 300 | 10² | 83.2 88.3 72.9 | 19.3 | 32.4 |
| FlexiViT-S | 300 | 8² | 83.2 88.3 73.0 | 19.4 | 32.2 |
| FlexiViT-B | 300 | 48² | 73.4 79.3 59.7 | 4.9 | 27.4 |
| FlexiViT-B | 300 | 40² | 77.0 82.6 63.7 | 7.0 | 29.6 |
| FlexiViT-B | 300 | 30² | 80.6 85.8 68.4 | 10.4 | 32.7 |
| FlexiViT-B | 300 | 24² | 82.6 87.3 71.2 | 14.6 | 34.1 |
| FlexiViT-B | 300 | 20² | 83.6 88.1 72.5 | 17.3 | 35.1 |
| FlexiViT-B | 300 | 16² | 84.5 88.6 73.8 | 21.8 | 35.4 |
| FlexiViT-B | 300 | 15² | 84.6 88.7 73.9 | 22.5 | 36.5 |
| FlexiViT-B | 300 | 12² | 84.9 89.0 74.6 | 25.5 | 36.9 |
| FlexiViT-B | 300 | 10² | 85.0 89.1 74.8 | 27.3 | 37.0 |
| FlexiViT-B | 300 | 8² | 85.1 89.1 75.0 | 27.1 | 36.9 |
| FlexiViT-L | 300 | 48² | 76.3 81.8 62.4 | 6.5 | 30.2 |
| FlexiViT-L | 300 | 40² | 79.5 84.7 66.4 | 9.0 | 32.6 |
| FlexiViT-L | 300 | 30² | 82.4 87.3 70.5 | 13.7 | 35.9 |
| FlexiViT-L | 300 | 24² | 84.0 88.5 72.8 | 18.0 | 37.7 |
| FlexiViT-L | 300 | 20² | 84.8 89.1 74.3 | 21.7 | 39.2 |
| FlexiViT-L | 300 | 16² | 85.4 89.6 75.7 | 26.7 | 39.4 |
| FlexiViT-L | 300 | 15² | 85.5 89.7 75.8 | 28.1 | 40.6 |
| FlexiViT-L | 300 | 12² | 85.8 89.9 76.4 | 30.9 | 40.8 |
| FlexiViT-L | 300 | 10² | 85.9 89.9 76.8 | 32.7 | 41.0 |
| FlexiViT-L | 300 | 8² | 85.9 90.0 76.7 | 33.4 | 41.2 |

Table 4. Scores for 90ep ImageNet-1k-only runs from Figure 2.

| Model | Eps | PS | Val ReaL v2 | -A | -R |
| --- | --- | --- | --- | --- | --- |
| FlexiViT-S | 90 | 48² | 65.9 72.5 51.9 | 2.9 | 23.2 |
| FlexiViT-S | 90 | 40² | 70.6 77.3 56.9 | 3.5 | 26.0 |
| FlexiViT-S | 90 | 30² | 76.4 82.9 62.9 | 5.9 | 29.3 |
| FlexiViT-S | 90 | 24² | 79.2 85.3 66.8 | 7.8 | 30.6 |
| FlexiViT-S | 90 | 20² | 80.7 86.5 69.0 | 10.6 | 31.5 |
| FlexiViT-S | 90 | 16² | 82.0 87.5 70.9 | 13.9 | 31.7 |
| FlexiViT-S | 90 | 15² | 82.2 87.7 71.1 | 14.4 | 32.7 |
| FlexiViT-S | 90 | 12² | 82.8 88.1 72.0 | 17.3 | 32.5 |
| FlexiViT-S | 90 | 10² | 83.0 88.2 72.7 | 19.0 | 32.4 |
| FlexiViT-S | 90 | 8² | 83.0 88.3 72.7 | 19.4 | 32.2 |
| FlexiViT-B | 90 | 48² | 71.9 77.8 58.0 | 4.6 | 26.9 |
| FlexiViT-B | 90 | 40² | 75.9 81.6 62.2 | 6.1 | 29.3 |
| FlexiViT-B | 90 | 30² | 80.2 85.3 67.3 | 9.5 | 32.3 |
| FlexiViT-B | 90 | 24² | 82.2 87.0 70.2 | 13.1 | 34.1 |
| FlexiViT-B | 90 | 20² | 83.3 87.8 71.9 | 16.5 | 35.1 |
| FlexiViT-B | 90 | 16² | 84.1 88.4 73.2 | 21.3 | 35.2 |
| FlexiViT-B | 90 | 15² | 84.3 88.5 73.9 | 21.8 | 36.4 |
| FlexiViT-B | 90 | 12² | 84.8 88.8 74.4 | 25.2 | 36.8 |
| FlexiViT-B | 90 | 10² | 85.0 89.0 74.5 | 27.3 | 36.9 |
| FlexiViT-B | 90 | 8² | 84.9 89.0 74.7 | 27.7 | 36.6 |
| FlexiViT-L | 90 | 48² | 74.3 80.0 60.2 | 5.6 | 29.4 |
| FlexiViT-L | 90 | 40² | 77.7 83.2 64.5 | 7.7 | 32.2 |
| FlexiViT-L | 90 | 30² | 81.7 86.7 69.4 | 12.1 | 35.3 |
| FlexiViT-L | 90 | 24² | 83.4 88.0 72.4 | 17.1 | 37.3 |
| FlexiViT-L | 90 | 20² | 84.4 88.8 73.9 | 20.9 | 38.9 |
| FlexiViT-L | 90 | 16² | 85.1 89.3 75.4 | 26.5 | 39.4 |
| FlexiViT-L | 90 | 15² | 85.3 89.5 75.6 | 27.2 | 40.3 |
| FlexiViT-L | 90 | 12² | 85.6 89.7 76.3 | 31.2 | 40.5 |
| FlexiViT-L | 90 | 10² | 85.7 89.8 76.7 | 33.1 | 40.7 |
| FlexiViT-L | 90 | 8² | 85.8 89.9 76.6 | 33.7 | 40.6 |

27

<!-- page 28 of 29 -->

Table 5. Numerical data for Figure 3.

| Model | /48 /40 /30 /24 /20 /16 /15 /12 /10 /8 |
| --- | --- |
| Flexi | 39.5 43.2 46.6 48.4 49.0 49.7 49.8 50.2 50.3 50.2 |
| B/16 | 0.0 0.1 2.4 21.6 41.7 50.5 50.4 47.9 43.3 30.5 |
| B/30 | 14.0 30.2 47.1 45.9 42.5 35.9 33.3 21.0 11.9 2.9 |

Table 6. Numerical data for Figure 5.

Table 8. Numerical data for Figure 8.

<table><tr><td>/40</td><td>/30</td><td>/24</td><td>/20</td><td>/15</td><td>/12</td><td>/10</td><td>/8</td></tr><tr><td colspan="8">ViT-B/30 transferred at /30</td></tr><tr><td>78.3</td><td>82.4</td><td>83.0</td><td>82.7</td><td>80.4</td><td>75.1</td><td>66.7</td><td>58.6</td></tr><tr><td colspan="8">FlexiViT-B transferred at /30</td></tr><tr><td>78.0</td><td>81.8</td><td>83.4</td><td>84.3</td><td>85.0</td><td>85.2</td><td>85.3</td><td>84.9</td></tr><tr><td colspan="8">FlexiViT-B transferred at /16</td></tr><tr><td>77.0</td><td>81.2</td><td>83.2</td><td>84.5</td><td>85.5</td><td>85.9</td><td>86.0</td><td>85.7</td></tr><tr><td colspan="8">FlexiViT-B transferred at /8</td></tr><tr><td>76.2</td><td>80.6</td><td>82.8</td><td>84.1</td><td>85.3</td><td>85.8</td><td>86.1</td><td>86.4</td></tr></table>

<table><tr><td colspan="2"></td><td>/5</td><td>/6</td><td>/8</td><td>/10</td><td>/12</td><td>/15</td><td>/16</td><td>/20</td><td>/24</td><td>/30</td></tr><tr><td colspan="12">Top-1 accuracy</td></tr><tr><td rowspan="3">T-init</td><td>90</td><td>40.8</td><td>43.8</td><td>47.9</td><td>49.4</td><td>50.5</td><td>51.3</td><td>51.4</td><td>51.9</td><td>51.9</td><td>52.0</td></tr><tr><td>300</td><td>43.1</td><td>45.9</td><td>48.7</td><td>50.3</td><td>51.1</td><td>51.6</td><td>51.6</td><td>51.9</td><td>52.0</td><td>52.0</td></tr><tr><td>1000</td><td>44.1</td><td>46.6</td><td>49.2</td><td>50.6</td><td>51.2</td><td>51.9</td><td>51.8</td><td>52.1</td><td>52.3</td><td>52.2</td></tr><tr><td colspan="2">R-init</td><td>41.5</td><td>44.2</td><td>47.0</td><td>48.5</td><td>48.9</td><td>49.6</td><td>49.8</td><td>50.0</td><td>50.1</td><td>50.0</td></tr><tr><td colspan="2">None</td><td>40.6</td><td>43.4</td><td>46.6</td><td>48.1</td><td>48.9</td><td>49.7</td><td>49.7</td><td>50.0</td><td>50.2</td><td>50.1</td></tr><tr><td colspan="2">Teacher</td><td colspan="10">52.2</td></tr><tr><td colspan="12">Top-1 agreement</td></tr><tr><td rowspan="3">T-init</td><td>90</td><td>56.0</td><td>61.9</td><td>69.5</td><td>74.7</td><td>78.1</td><td>81.1</td><td>81.8</td><td>83.8</td><td>84.5</td><td>84.4</td></tr><tr><td>300</td><td>59.6</td><td>65.7</td><td>73.0</td><td>77.3</td><td>80.1</td><td>82.8</td><td>82.9</td><td>84.2</td><td>84.5</td><td>84.5</td></tr><tr><td>1000</td><td>62.0</td><td>67.5</td><td>74.4</td><td>78.6</td><td>81.3</td><td>83.4</td><td>83.7</td><td>84.6</td><td>85.2</td><td>85.0</td></tr><tr><td colspan="2">R-init</td><td>56.4</td><td>60.7</td><td>66.2</td><td>69.0</td><td>70.7</td><td>72.1</td><td>72.4</td><td>72.8</td><td>73.1</td><td>73.1</td></tr><tr><td colspan="12">CKA similarity</td></tr><tr><td rowspan="3">T-init</td><td>90</td><td>.65</td><td>.69</td><td>.76</td><td>.80</td><td>.84</td><td>.86</td><td>.87</td><td>.88</td><td>.89</td><td>.89</td></tr><tr><td>300</td><td>.68</td><td>.72</td><td>.78</td><td>.82</td><td>.85</td><td>.86</td><td>.87</td><td>.87</td><td>.88</td><td>.87</td></tr><tr><td>1000</td><td>.68</td><td>.72</td><td>.78</td><td>.81</td><td>.83</td><td>.85</td><td>.85</td><td>.86</td><td>.86</td><td>.86</td></tr><tr><td colspan="2">R-init</td><td>.41</td><td>.43</td><td>.45</td><td>.47</td><td>.48</td><td>.49</td><td>.50</td><td>.50</td><td>.50</td><td>.50</td></tr></table>

Table 9. Numerical data for Figure 9.

<table><tr><td>Model</td><td>Transfer</td><td>SUN</td><td>Food</td><td>Pet</td><td>Flow</td><td>C10</td><td>C100</td></tr><tr><td colspan="8">Evaluated at /30</td></tr><tr><td>ViT-B/30</td><td>/30</td><td>79.7</td><td>90.8</td><td>93.6</td><td>99.4</td><td>99.0</td><td>91.9</td></tr><tr><td>ViT-B/30</td><td>Flexi</td><td>79.3</td><td>90.8</td><td>93.3</td><td>99.4</td><td>99.0</td><td>91.9</td></tr><tr><td>ViT-B/16</td><td>Flexi</td><td>77.7</td><td>89.0</td><td>92.3</td><td>99.2</td><td>98.8</td><td>91.8</td></tr><tr><td>FlexiViT-B</td><td>Flexi</td><td>79.3</td><td>90.3</td><td>93.9</td><td>99.4</td><td>98.9</td><td>92.3</td></tr><tr><td colspan="8">Evaluated at /16</td></tr><tr><td>ViT-B/16</td><td>/16</td><td>82.3</td><td>93.7</td><td>94.9</td><td>99.6</td><td>99.1</td><td>93.2</td></tr><tr><td>ViT-B/30</td><td>Flexi</td><td>81.6</td><td>93.0</td><td>93.8</td><td>99.5</td><td>98.7</td><td>91.3</td></tr><tr><td>ViT-B/16</td><td>Flexi</td><td>82.1</td><td>93.6</td><td>94.9</td><td>99.5</td><td>99.1</td><td>92.8</td></tr><tr><td>FlexiViT-B</td><td>Flexi</td><td>82.8</td><td>93.8</td><td>94.8</td><td>99.6</td><td>99.1</td><td>93.0</td></tr><tr><td colspan="8">Evaluated at /8</td></tr><tr><td>ViT-B/30</td><td>Flexi</td><td>80.9</td><td>92.7</td><td>92.2</td><td>98.8</td><td>97.8</td><td>89.0</td></tr><tr><td>ViT-B/16</td><td>Flexi</td><td>82.4</td><td>94.0</td><td>95.0</td><td>99.6</td><td>98.7</td><td>91.6</td></tr><tr><td>FlexiViT-B</td><td>Flexi</td><td>83.2</td><td>94.7</td><td>94.9</td><td>99.6</td><td>98.9</td><td>92.8</td></tr></table>

Table 7. Numerical data for Figure 7.

| Method / | 30→/30 | F→/30 | /16→/16 | F→/16 |
| --- | --- | --- | --- | --- |
| Clf SUN397 | 79.1 | 79.7 | 82.3 | 82.5 |
| Clf Food101 | 90.4 | 90.8 | 93.7 | 94.1 |
| Clf Pets | 93.6 | 93.6 | 94.9 | 94.9 |
| Clf Flowers102 | 99.4 | 99.4 | 99.6 | 99.6 |
| Clf CIFAR-10 | 98.8 | 99.0 | 99.1 | 99.1 |
| Clf CIFAR-100 | 92.3 | 91.9 | 93.2 | 93.3 |
| UViM coco PQ | 24.8 | 24.1 | 30.5 | 34.3 |
| OWL-ViT lvis AP | 18.8 | 18.4 | 22.7 | 23.4 |
| LiT i2t coco | 42.2 | 41.6 | 44.4 | 45.8 |
| Seg (lin) City mIoU | 61.0 | 61.1 | 69.3 | 70.0 |
| Seg (lin) ADE mIoU | 43.1 | 43.5 | 46.1 | 47.5 |

Table 10. Numerical data for Figure 10.

| Base | LiT | /48 /30 /24 /16 /12 /10 /8 |
| --- | --- | --- |
| ViT-B/30 | /30 | 46.7 65.2 65.2 64.5 30.9 10.3 3.2 |
| ViT-B/16 | /16 | 3.2 42.7 60.2 71.9 61.6 45.7 57.8 |
| FlexiViT-B | /30 | 49.0 59.8 67.0 72.5 72.6 74.7 74.1 |
| FlexiViT-B | /16 | 48.2 62.6 66.6 73.3 73.1 74.5 75.0 |
| FlexiViT-B | Flexi | 51.0 62.5 69.2 73.4 74.5 75.5 75.1 |

Table 11. Numerical data for Figure 11.

| Base | OWL | /48 /40 /30 /24 /20 /16 /12 |
| --- | --- | --- |
| LiT-B/30 | /30 | 6.2 10.8 20.6 15.6 7.8 1.7 0.3 |
| LiT-B/30 | /16 | 0.1 0.5 4.1 9.5 16.7 26.8 15.0 |
| LiT-B/30 | Flexi | 15.7 17.8 21.2 23.0 24.2 25.6 26.0 |
| FlexiLiT-B | Flexi | 16.0 18.5 21.5 23.5 24.9 26.7 27.1 |

28

<!-- page 29 of 29 -->

Table 12. Numerical data for Figure 4.

| Resize | /2 | /4 | /6 | /8 | /10 | /12 | /14 | /16 | /18 | /20 | /22 | /24 | /26 | /28 | /30 | /32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PI | 10.3 | 44.5 | 48.9 | 52.4 | 52.4 | 52.4 | 52.4 | 52.3 | 52.3 | 52.4 | 52.3 | 52.4 | 52.3 | 52.3 | 52.4 | 52.4 |
| Area | 24.5 | 42.6 | 46.1 | 52.4 | 47.7 | 48.3 | 48.6 | 48.1 | 48.7 | 48.4 | 48.5 | 48.7 | 48.5 | 48.5 | 48.6 | 48.4 |
| Norm | 10.8 | 40.7 | 46.4 | 52.4 | 45.6 | 44.1 | 42.0 | 39.4 | 37.1 | 33.4 | 29.8 | 26.1 | 22.0 | 18.1 | 14.8 | 11.9 |
| Vanilla | 25.5 | 41.6 | 45.6 | 52.4 | 42.5 | 37.2 | 25.5 | 12.8 | 5.4 | 1.9 | 0.6 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 |

Table 13. Numerical data for Figure 14.

<table><tr><td>Setting</td><td>Params</td><td>GFLOPs</td><td>Speed</td><td>Prec</td></tr><tr><td colspan="5">Patch FlexiViT-B</td></tr><tr><td>Eval at patch /30</td><td>87.5M</td><td>15.7</td><td>2745</td><td>47.9</td></tr><tr><td>Eval at patch /24</td><td>87.5M</td><td>20.6</td><td>2022</td><td>49.4</td></tr><tr><td>Eval at patch /20</td><td>87.5M</td><td>27.7</td><td>1135</td><td>50.5</td></tr><tr><td>Eval at patch /16</td><td>87.5M</td><td>41.9</td><td>806</td><td>51.3</td></tr><tr><td>Eval at patch /15</td><td>87.5M</td><td>47.6</td><td>595</td><td>51.4</td></tr><tr><td>Eval at patch /12</td><td>87.5M</td><td>75.3</td><td>362</td><td>51.9</td></tr><tr><td>Eval at patch /10</td><td>87.5M</td><td>111.5</td><td>246</td><td>51.9</td></tr><tr><td>Eval at patch /8</td><td>87.5M</td><td>184.5</td><td>128</td><td>52.0</td></tr><tr><td colspan="5">Stride FlexiViT-B/32</td></tr><tr><td>Eval at stride 30</td><td>87.8M</td><td>11.7</td><td>2317</td><td>47.5</td></tr><tr><td>Eval at stride 23</td><td>87.8M</td><td>18.2</td><td>1768</td><td>49.3</td></tr><tr><td>Eval at stride 19</td><td>87.8M</td><td>26.3</td><td>1041</td><td>50.3</td></tr><tr><td>Eval at stride 15</td><td>87.8M</td><td>41.7</td><td>750</td><td>51.0</td></tr><tr><td>Eval at stride 14</td><td>87.8M</td><td>47.6</td><td>563</td><td>51.2</td></tr><tr><td>Eval at stride 11</td><td>87.8M</td><td>76.4</td><td>347</td><td>51.5</td></tr><tr><td>Eval at stride 9</td><td>87.8M</td><td>113.7</td><td>235</td><td>51.6</td></tr><tr><td>Eval at stride 7</td><td>87.8M</td><td>188.4</td><td>124</td><td>51.7</td></tr><tr><td colspan="5">Depth FlexiViT-B/8</td></tr><tr><td>Eval at depth 3</td><td>22.1M</td><td>46.3</td><td>403</td><td>35.2</td></tr><tr><td>Eval at depth 6</td><td>43.4M</td><td>92.2</td><td>216</td><td>44.8</td></tr><tr><td>Eval at depth 9</td><td>64.6M</td><td>138.2</td><td>147</td><td>48.7</td></tr><tr><td>Eval at depth 12</td><td>85.9M</td><td>184.2</td><td>112</td><td>51.0</td></tr></table>

29