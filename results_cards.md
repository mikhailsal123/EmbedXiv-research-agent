# EmbedXiv suggestions

Source: `papers/CBAM_paper.pdf`

40 kept paper(s).

## Problem

Convolutional networks do not explicitly emphasize the most informative channels and spatial locations in intermediate feature maps.

### [CBAM: Convolutional Block Attention Module](https://www.semanticscholar.org/paper/de95601d9e3b20ec51aa33e1f27b1880d2c44ef2)

- **Date:** 2018
- **arXiv:** `1807.06521`
- **Relation:** Same problem
- **Why:** The candidate paper introduces CBAM, which directly addresses the source's problem of improving feature refinement through sequential channel and spatial attention. The implementation details (shared MLP with average/max pooling for channel attention, 7x7 convolution for spatial attention) and claims (sequential attention improves performance) are identical to the source extraction.
- **Via S2 graph from:** `1807.06514`
- **Abstract:** We propose Convolutional Block Attention Module (CBAM), a simple yet effective attention module for feed-forward convolutional neural networks. Given an intermediate feature map, our module sequentially infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to…

## Claim

Sequential attention at channel and spatial granularities improves feature refinement.

### [Revisiting the Ordering of Channel and Spatial Attention: A Comprehensive Study on Sequential and Parallel Designs](https://arxiv.org/abs/2601.07310)

- **Date:** 2026-01-26
- **arXiv:** `2601.07310`
- **Relation:** Supports a claim
- **Why:** The candidate paper systematically evaluates sequential vs. parallel channel-spatial attention designs, directly supporting the source's claim that sequential attention improves feature refinement. It validates the source's implementation details (e.g., 7x7 spatial convolution) and extends analysis to data-scale-dependent performance patterns.
- **Abstract:** Attention mechanisms have become a core component of deep learning models, with Channel Attention and Spatial Attention being the two most representative architectures. Current research on their fusion strategies primarily bifurcates into sequential and parallel paradigms, yet the selection process remains largely emp…

### [Attention mechanisms in computer vision: A survey](https://www.semanticscholar.org/paper/45f686be3b96302ede327645227134e1c304dbab)

- **Date:** 2021
- **arXiv:** `2111.07624`
- **Relation:** Supports a claim
- **Why:** The survey paper categorizes and discusses channel and spatial attention mechanisms, explicitly mentioning CBAM (the source's method) as a representative example of combined channel-spatial attention. This supports the source's claim that sequential attention improves feature refinement by situating it within a broader taxonomy of attention mechanisms.
- **Via S2 graph from:** `2210.07828`
- **Abstract:** Humans can naturally and effectively find salient regions in complex scenes. Motivated by this observation, attention mechanisms were introduced into computer vision with the aim of imitating this aspect of the human visual system. Such an attention mechanism can be regarded as a dynamic weight adjustment process base…

### [PKCAM: Previous Knowledge Channel Attention Module](https://arxiv.org/abs/2211.07521)

- **Date:** 2022-11-28
- **arXiv:** `2211.07521`
- **Relation:** Extends a claim
- **Why:** The candidate paper introduces PKCAM, which extends the concept of channel attention by aggregating previous layers' features to model global context, building upon the sequential channel-spatial attention framework (as in CBAM) but adding cross-layer knowledge reuse. This aligns with the source's problem of emphasizing informative channels/spatial locations but introduces a novel extension.
- **Abstract:** Recently, attention mechanisms have been explored with ConvNets, both across the spatial and channel dimensions. However, from our knowledge, all the existing methods devote the attention modules to capture local interactions from a uni-scale. In this paper, we propose a Previous Knowledge Channel Attention Module(PKC…

### [FcaNet: Frequency Channel Attention Networks](https://arxiv.org/abs/2012.11879)

- **Date:** 2021-07-26
- **arXiv:** `2012.11879`
- **Relation:** Extends a claim
- **Why:** FcaNet improves channel attention by introducing frequency-based compression (DCT) as an alternative to scalar-based methods like GAP, which aligns with the source's focus on refining channel attention. While it does not address spatial attention explicitly, it extends the source's claim about enhancing channel attention mechanisms.
- **Abstract:** Attention mechanism, especially channel attention, has gained great success in the computer vision field. Many works focus on how to design efficient channel attention mechanisms while ignoring a fundamental problem, i.e., channel attention mechanism uses scalar to represent channel, which is difficult due to massive…

### [Information Bottleneck Approach to Spatial Attention Learning](https://arxiv.org/abs/2108.03418)

- **Date:** 2025-02-04
- **arXiv:** `2108.03418`
- **Relation:** Extends a claim
- **Why:** The candidate proposes an IB-inspired spatial attention mechanism that addresses part of the source's problem (spatial feature refinement) with a different theoretical foundation (information bottleneck theory) and implementation (mutual information optimization + quantization). While the source focuses on sequential channel-spatial attention (CBAM), this work extends the problem domain by formal-
- **Abstract:** The selective visual attention mechanism in the human visual system (HVS) restricts the amount of information to reach visual awareness for perceiving natural scenes, allowing near real-time information processing with limited computational capacity [Koch and Ullman, 1987]. This kind of selectivity acts as an 'Informa…

### [A Discriminative Channel Diversification Network for Image Classification](https://arxiv.org/abs/2112.05861)

- **Date:** 2021-12-14
- **arXiv:** `2112.05861`
- **Relation:** Extends a claim
- **Why:** The candidate proposes a novel channel diversification module that addresses the same problem of emphasizing informative channels in CNNs but uses a distinct approach (combining global average pooling and channel relationship matrices) compared to the source's sequential channel-spatial attention. It extends the claim by introducing an alternative method for feature refinement.
- **Abstract:** Channel attention mechanisms in convolutional neural networks have been proven to be effective in various computer vision tasks. However, the performance improvement comes with additional model complexity and computation cost. In this paper, we propose a light-weight and effective attention module, called channel dive…

### [CAT: Learning to Collaborate Channel and Spatial Attention from Multi-Information Fusion](https://arxiv.org/abs/2212.06335)

- **Date:** 2022-12-14
- **arXiv:** `2212.06335`
- **Relation:** Extends a claim
- **Why:** The candidate paper extends the source's claim about sequential channel-spatial attention by introducing trainable coefficients (colla-factors) to dynamically combine attention contributions and adding global entropy pooling (GEP) for noise suppression, while retaining the 7×7 convolution for spatial attention and sequential application of channel/space modules.
- **Abstract:** Channel and spatial attention mechanism has proven to provide an evident performance boost of deep convolution neural networks (CNNs). Most existing methods focus on one or run them parallel (series), neglecting the collaboration between the two attentions. In order to better establish the feature interaction between…

### [Sharpen Focus: Learning with Attention Separability and Consistency](https://arxiv.org/abs/1811.07484)

- **Date:** 2019-08-09
- **arXiv:** `1811.07484`
- **Relation:** Extends a claim
- **Why:** The candidate paper introduces attention separability and cross-layer consistency losses to improve class-discriminative attention, extending the concept of sequential channel-spatial attention (as in CBAM) to address visual confusion between classes. While it shares the domain and some attention mechanisms (e.g., channel attention via MLP), its primary focus is on discriminative attention rather
- **Abstract:** Recent developments in gradient-based attention modeling have seen attention maps emerge as a powerful tool for interpreting convolutional neural networks. Despite good localization for an individual class of interest, these techniques produce attention maps with substantially overlapping responses among different cla…

### [Convolutional Rectangular Attention Module](https://arxiv.org/abs/2503.10875)

- **Date:** 2025-09-01
- **arXiv:** `2503.10875`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces a novel spatial attention mechanism with rectangular constraints, which directly addresses the source problem of improving spatial feature emphasis in CNNs. While the source uses a 7x7 convolution for spatial attention, this work proposes a parametrized rectangular region as an alternative implementation, offering a different approach to the same problem domain.
- **Abstract:** In this paper, we introduce a novel spatial attention module that can be easily integrated to any convolutional network. This module guides the model to pay attention to the most discriminative part of an image. This enables the model to attain a better performance by an end-to-end training. In conventional approaches…

### [AttZoom: Attention Zoom for Better Visual Features](https://arxiv.org/abs/2508.03625)

- **Date:** 2025-08-06
- **arXiv:** `2508.03625`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes a standalone spatial attention mechanism (AttZoom) that addresses the same problem of emphasizing informative spatial regions in CNNs, but uses a different implementation approach (learnable zoom filters and upscaling) compared to the source's sequential channel-spatial attention (CBAM).
- **Abstract:** We present Attention Zoom, a modular and model-agnostic spatial attention mechanism designed to improve feature extraction in convolutional neural networks (CNNs). Unlike traditional attention approaches that require architecture-specific integration, our method introduces a standalone layer that spatially emphasizes…

### [Parameter-Free Channel Attention for Image Classification and Super-Resolution](https://arxiv.org/abs/2303.11055)

- **Date:** 2023-03-21
- **arXiv:** `2303.11055`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes a parameter-free channel attention module (PFCA) as an alternative implementation to the source's parameterized channel attention mechanism (using MLPs). Both address the problem of emphasizing informative channels in CNNs, but PFCA avoids learnable parameters by using fixed statistical operations (mean/variance) instead.
- **Abstract:** The channel attention mechanism is a useful technique widely employed in deep convolutional neural networks to boost the performance for image processing tasks, eg, image classification and image super-resolution. It is usually designed as a parameterized sub-network and embedded into the convolutional layers of the n…

### [Parameter-Free Average Attention Improves Convolutional Neural Network Performance (Almost) Free of Charge](https://arxiv.org/abs/2210.07828)

- **Date:** 2022-10-17
- **arXiv:** `2210.07828`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces PfAAM, a parameter-free attention module that addresses the same problem of emphasizing informative channels and spatial locations in CNN feature maps. While the source uses CBAM with MLP-based channel attention and 7x7 convolutions for spatial attention, PfAAM replaces these with simple averaging operations and a sigmoid gate, offering an alternative implementation.
- **Abstract:** Visual perception is driven by the focus on relevant aspects in the surrounding world. To transfer this observation to the digital information processing of computers, attention mechanisms have been introduced to highlight salient image regions. Here, we introduce a parameter-free attention mechanism called PfAAM, tha…

### [DAS: A Deformable Attention to Capture Salient Information in CNNs](https://arxiv.org/abs/2311.12091)

- **Date:** 2023-11-22
- **arXiv:** `2311.12091`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces DAS, a deformable attention mechanism that holistically computes pixel-wise attention without explicitly separating channel and spatial attention stages, contrasting with the source's CBAM approach which sequentially applies channel then spatial attention. Both address CNN feature refinement but differ in implementation.
- **Abstract:** Convolutional Neural Networks (CNNs) excel in local spatial pattern recognition. For many vision tasks, such as object recognition and segmentation, salient information is also present outside CNN's kernel boundaries. However, CNNs struggle in capturing such relevant information due to their confined receptive fields.…

### [An Attention Module for Convolutional Neural Networks](https://www.semanticscholar.org/paper/7c59475b1645696e81260f714b7dddbc73cb3321)

- **Date:** 2021
- **arXiv:** `2108.08205`
- **Relation:** Implementation alternative
- **Why:** The candidate paper proposes an attention module (AW-convolution) that addresses different limitations (approximation and capacity problems in attention maps) compared to the source paper's CBAM. While both focus on attention in CNNs, the candidate introduces a distinct implementation approach that can be combined with CBAM, making it an alternative method in the same domain.
- **Via S2 graph from:** `2111.13470`
- **Abstract:** Attention mechanism has been regarded as an advanced technique to capture long-range feature interactions and to boost the representation capability for convolutional neural networks. However, we found two ignored problems in current attentional activations-based models: the approximation problem and the insufficient…

### [Partial Channel Network: Compute Fewer, Perform Better](https://www.semanticscholar.org/paper/9a51be658c34ba445c049b41d4182e441f8a720b)

- **Date:** 2025
- **arXiv:** `2502.01303`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces a partial channel mechanism (PCM) and PATConv, which parallelizes attention and convolution operations on split channels to reduce computational cost. While it shares domain relevance (CNNs, attention mechanisms) with the source's CBAM, its focus on efficiency through parallelized, split-channel operations differs from the source's sequential channel-spatialattention
- **Via S2 graph from:** `1910.03151`
- **Abstract:** Designing a module or mechanism that enables a network to maintain low parameters and FLOPs without sacrificing accuracy and throughput remains a challenge. To address this challenge and exploit the redundancy within feature map channels, we propose a new solution: partial channel mechanism (PCM). Specifically, throug…

### [Vision Eagle Attention: A New Lens for Advancing Image Classification](https://www.semanticscholar.org/paper/c5d4ef3e119882166a7a8774d81c37feee7ab63e)

- **Date:** 2024
- **arXiv:** `2411.10564`
- **Relation:** Implementation alternative
- **Why:** The candidate introduces Vision Eagle Attention, a convolutional spatial attention mechanism applied to ResNet-18, which shares the domain of CNN feature refinement but uses a different implementation (convolutional blocks) compared to the source's CBAM (channel + spatial attention via MLP and 7x7 conv).
- **Via S2 graph from:** `2210.07828`
- **Abstract:** In computer vision tasks, the ability to focus on relevant regions within an image is crucial for improving model performance, particularly when key features are small, subtle, or spatially dispersed. Convolutional neural networks (CNNs) typically treat all regions of an image equally, which can lead to inefficient fe…

### [Lightweight Channel Attention for Efficient CNNs](https://www.semanticscholar.org/paper/e4b9aa55c32508d861fbda6a84e25085ecfd2c2a)

- **Date:** 2026
- **arXiv:** `2601.01002`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes Lite Channel Attention (LCA) as a parameter-efficient alternative to existing channel attention mechanisms (e.g., SE, ECA), which aligns with the source's focus on channel attention but does not address spatial attention. It offers an alternative implementation for the channel attention component of the source's CBAM framework.
- **Via S2 graph from:** `2107.02145`
- **Abstract:** Attention mechanisms have become integral to modern convolutional neural networks (CNNs), delivering notable performance improvements with minimal computational overhead. However, the efficiency accuracy trade off of different channel attention designs remains underexplored. This work presents an empirical study compa…

## Implementation

Apply channel attention first, followed by spatial attention.

### [CSA-Net: Channel-wise Spatially Autocorrelated Attention Networks](https://arxiv.org/abs/2405.05755)

- **Date:** 2024-05-14
- **arXiv:** `2405.05755`
- **Relation:** Implementation alternative
- **Why:** The CSA-Net paper proposes a different implementation for channel-wise attention (using spatial autocorrelation inspired by geography) compared to the source's CBAM (using shared MLP and 7x7 convolution). Both address the same problem of emphasizing informative channels/spatial locations but differ in technical execution.
- **Abstract:** In recent years, convolutional neural networks (CNNs) with channel-wise feature refining mechanisms have brought noticeable benefits to modelling channel dependencies. However, current attention paradigms fail to infer an optimal channel descriptor capable of simultaneously exploiting statistical and spatial relations…

### [ELA: Efficient Local Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/2403.01123)

- **Date:** 2024-03-05
- **arXiv:** `2403.01123`
- **Relation:** Implementation alternative
- **Why:** The candidate paper proposes ELA, an alternative attention mechanism that shares the same problem (emphasizing informative channels/spatial locations) and claim (sequential attention improves feature refinement) as CBAM. While ELA uses 1D convolutions and Group Normalization instead of CBAM's 7x7 convolutions and Batch Normalization, the matched implementation detail (7x7 convolution for spatial)
- **Abstract:** The attention mechanism has gained significant recognition in the field of computer vision due to its ability to effectively enhance the performance of deep neural networks. However, existing methods often struggle to effectively utilize spatial information or, if they do, they come at the cost of reducing channel dim…

### [TDAM: Top-Down Attention Module for Contextually Guided Feature Selection in CNNs](https://arxiv.org/abs/2111.13470)

- **Date:** 2022-10-24
- **arXiv:** `2111.13470`
- **Relation:** Implementation alternative
- **Why:** TDAM proposes a top-down attention mechanism with iterative feedback and joint modeling of top/bottom features, differing from CBAM's sequential channel-spatial approach but addressing the same problem of improving feature refinement in CNNs.
- **Abstract:** Attention modules for Convolutional Neural Networks (CNNs) are an effective method to enhance performance on multiple computer-vision tasks. While existing methods appropriately model channel-, spatial- and self-attention, they primarily operate in a feedforward bottom-up manner. Consequently, the attention mechanism…

### [Channel Locality Block: A Variant of Squeeze-and-Excitation](https://arxiv.org/abs/1901.01493)

- **Date:** 2019-01-08
- **arXiv:** `1901.01493`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes a different implementation for channel attention (using convolutional layers to learn local channel correlations) compared to the source's method (shared MLP for global statistics). Both address channel emphasis but differ in technical execution.
- **Abstract:** Attention mechanism is a hot spot in deep learning field. Using channel attention model is an effective method for improving the performance of the convolutional neural network. Squeeze-and-Excitation block takes advantage of the channel dependence, selectively emphasizing the important channels and compressing the re…

### [Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions](https://arxiv.org/abs/2112.05561)

- **Date:** 2021-12-13
- **arXiv:** `2112.05561`
- **Relation:** Implementation alternative
- **Why:** The candidate paper proposes a Global Attention Mechanism (GAM) that sequentially applies channel and spatial attention (like CBAM) but uses a 3D-permutation MLP for channel attention and a 7×7 convolution for spatial attention, differing from the source's shared MLP with average/max pooling and 7×7 convolution. This constitutes an alternative implementation addressing the same problem of refining
- **Abstract:** A variety of attention mechanisms have been studied to improve the performance of various computer vision tasks. However, the prior methods overlooked the significance of retaining the information on both channel and spatial aspects to enhance the cross-dimension interactions. Therefore, we propose a global attention…

### [Convolutional Neural Network optimization via Channel Reassessment Attention module](https://arxiv.org/abs/2010.05605)

- **Date:** 2020-10-13
- **arXiv:** `2010.05605`
- **Relation:** Implementation alternative
- **Why:** The CRA module proposes a different implementation for channel attention (using global depthwise convolution on compressed spatial features) compared to CBAM's shared MLP with average/max pooling, while addressing the same problem of refining features through channel and spatial emphasis.
- **Abstract:** The performance of convolutional neural networks (CNNs) can be improved by adjusting the interrelationship between channels with attention mechanism. However, attention mechanism in recent advance has not fully utilized spatial information of feature maps, which makes a great difference to the results of generated cha…

### [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)

- **Date:** 2020-04-08
- **arXiv:** `1910.03151`
- **Relation:** Implementation alternative
- **Why:** ECA-Net proposes an efficient channel attention module as an alternative to the source's CBAM approach, addressing the same problem of emphasizing informative channels in CNNs.
- **Abstract:** Recently, channel attention mechanism has demonstrated to offer great potential in improving the performance of deep convolutional neural networks (CNNs). However, most existing methods dedicate to developing more sophisticated attention modules for achieving better performance, which inevitably increase model complex…

### [Attention as Activation](https://arxiv.org/abs/2007.07729)

- **Date:** 2020-08-04
- **arXiv:** `2007.07729`
- **Relation:** Implementation alternative
- **Why:** The candidate paper proposes ATAC units that address the same problem of emphasizing informative channels and spatial locations in CNNs but uses a different implementation (local channel attention as an activation function) compared to the source's CBAM (sequential channel and spatial attention modules with shared MLP and 7x7 convolution).
- **Abstract:** Activation functions and attention mechanisms are typically treated as having different purposes and have evolved differently. However, both concepts can be formulated as a non-linear gating function. Inspired by their similarity, we propose a novel type of activation units called attentional activation (ATAC) units a…

### [SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention](https://arxiv.org/abs/2407.05128)

- **Date:** 2024-11-13
- **arXiv:** `2407.05128`
- **Relation:** Implementation alternative
- **Why:** The candidate paper proposes SCSA, which combines spatial and channel attention in a reversed order (spatial first, then channel) compared to the source's CBAM (channel first, then spatial). While both address feature refinement via sequential attention, the candidate uses distinct mechanisms (e.g., multi-scale convolutions in SMSA and self-attention in PCSA) and reverses the implementation order,
- **Abstract:** Channel and spatial attentions have respectively brought significant improvements in extracting feature dependencies and spatial structure relations for various downstream vision tasks. While their combination is more beneficial for leveraging their individual strengths, the synergy between channel and spatial attenti…

### [BAM: Bottleneck Attention Module](https://arxiv.org/abs/1807.06514)

- **Date:** 2018-07-19
- **arXiv:** `1807.06514`
- **Relation:** Implementation alternative
- **Why:** The BAM module in the candidate paper also applies channel attention followed by spatial attention (matching the source's implementation sequence), but uses dilated convolutions for spatial attention instead of the 7x7 convolution specified in the source. This represents a distinct implementation approach to the same sequential attention principle.
- **Abstract:** Recent advances in deep neural networks have been developed via architecture search for stronger representational power. In this work, we focus on the effect of attention in general deep neural networks. We propose a simple and effective attention module, named Bottleneck Attention Module (BAM), that can be integrated…

### [WaveNets: Wavelet Channel Attention Networks](https://arxiv.org/abs/2211.02695)

- **Date:** 2024-03-13
- **arXiv:** `2211.02695`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes WaveNet, a wavelet-based channel attention mechanism that replaces the source's MLP and 7x7 convolution with wavelet transforms for feature compression. While both works address channel-spatial sequential attention, the candidate's implementation diverges technically (wavelet vs. pooling/convolution) but remains functionally aligned with the source's problem and claim.
- **Abstract:** Channel Attention reigns supreme as an effective technique in the field of computer vision. However, the proposed channel attention by SENet suffers from information loss in feature learning caused by the use of Global Average Pooling (GAP) to represent channels as scalars. Thus, designing effective channel attention…

### [SA-Net: Shuffle Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/2102.00240)

- **Date:** 2021-02-02
- **arXiv:** `2102.00240`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes Shuffle Attention (SA), which combines channel and spatial attention through grouped processing and channel shuffling, differing from CBAM's sequential application without grouping. Both use sequential attention (channel → spatial) and 7x7 convolutions for spatial refinement, but SA introduces a novel grouped architecture with shuffling to reduce computational overhead while
- **Abstract:** Attention mechanisms, which enable a neural network to accurately focus on all the relevant elements of the input, have become an essential component to improve the performance of deep neural networks. There are mainly two attention mechanisms widely used in computer vision studies, \textit{spatial attention} and \tex…

### [GPCA: A Probabilistic Framework for Gaussian Process Embedded Channel Attention](https://arxiv.org/abs/2003.04575)

- **Date:** 2021-08-11
- **arXiv:** `2003.04575`
- **Relation:** Implementation alternative
- **Why:** The GPCA module applies channel attention followed by spatial attention (matching the source's implementation) but uses a probabilistic Gaussian process framework with Sigmoid-Gaussian approximations, differing fundamentally from the source's functional approach (shared MLP + 7x7 convolution). This constitutes an alternative implementation methodology for the same sequential attention paradigm.
- **Abstract:** Channel attention mechanisms have been commonly applied in many visual tasks for effective performance improvement. It is able to reinforce the informative channels as well as to suppress the useless channels. Recently, different channel attention modules have been proposed and implemented in various ways. Generally s…

### [Efficient Multi-Scale Attention Module with Cross-Spatial Learning](https://arxiv.org/abs/2305.13563)

- **Date:** 2023-06-07
- **arXiv:** `2305.13563`
- **Relation:** Implementation alternative
- **Why:** The candidate proposes a parallel multi-scale attention module (EMA) with cross-spatial learning, differing from the source's sequential CBAM-like approach (channel then spatial attention). While both address channel/spatial refinement, EMA uses parallel 1x1/3x3 convolutions and avoids the source's 7x7 spatial kernel, offering an alternative implementation strategy.
- **Abstract:** Remarkable effectiveness of the channel or spatial attention mechanisms for producing more discernible feature representation are illustrated in various computer vision tasks. However, modeling the cross-channel relationships with channel dimensionality reduction may bring side effect in extracting deep visual represe…

### [BA-Net: Bridge Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/2112.04150)

- **Date:** 2022-06-02
- **arXiv:** `2112.04150`
- **Relation:** Implementation alternative
- **Why:** The candidate paper BA-Net shares the same sequential application of channel attention followed by spatial attention (as in CBAM) and uses a 7x7 convolution for spatial attention (matching Detail 0.0.2). However, it introduces a novel 'bridge' mechanism to integrate features from previous layers into attention weight computation, differing from CBAM's design while reusing the spatial attention CNN
- **Abstract:** In recent years, channel attention mechanism has been widely investigated due to its great potential in improving the performance of deep convolutional neural networks (CNNs) in many vision tasks. However, in most of the existing methods, only the output of the adjacent convolution layer is fed into the attention laye…

### [BA-Net: Bridge Attention in Deep Neural Networks](https://arxiv.org/abs/2410.07860)

- **Date:** 2024-10-14
- **arXiv:** `2410.07860`
- **Relation:** Implementation alternative
- **Why:** The candidate paper applies channel attention followed by spatial attention (as in the source's Detail 0.0.0), but its primary focus is on cross-layer feature integration via 'bridge attention' rather than the source's two-stage feature refinement. While the sequential attention order matches, the context and purpose differ: the source emphasizes intra-layer refinement, while the candidate bridges
- **Abstract:** Attention mechanisms, particularly channel attention, have become highly influential in numerous computer vision tasks. Despite their effectiveness, many existing methods primarily focus on optimizing performance through complex attention modules applied at individual convolutional layers, often overlooking the synerg…

### [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928)

- **Date:** 2017-02-14
- **arXiv:** `1612.03928`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces attention transfer via teacher-student networks, using activation-based attention maps (e.g., sum/max of activations across channels). While it shares the concept of sequential channel-spatial attention (matched detail), its implementation differs fundamentally from CBAM: it focuses on knowledge distillation rather than explicit feature refinement via shared MLPs and
- **Abstract:** Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defin…

## Implementation

Compute spatial attention with a 7x7 convolution over pooled channel features.

### [Augmenting Convolutional networks with attention-based aggregation](https://arxiv.org/abs/2112.13692)

- **Date:** 2021-12-28
- **arXiv:** `2112.13692`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces an attention-based aggregation layer to replace final pooling in CNNs, using a 7x7 convolution for spatial attention (matching Detail 0.0.2). However, it focuses on global patch aggregation for non-local reasoning rather than the source's sequential channel-spatial refinement. Both use attention in CNNs but address different architectural goals.
- **Abstract:** We show how to augment any convolutional network with an attention-based global map to achieve non-local reasoning. We replace the final average pooling by an attention-based aggregation layer akin to a single transformer block, that weights how the patches are involved in the classification decision. We plug this lea…

### [Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks](https://arxiv.org/abs/1810.12348)

- **Date:** 2019-01-15
- **arXiv:** `1810.12348`
- **Relation:** Implementation alternative
- **Why:** The candidate paper's 'Gather-Excite' framework introduces a spatial attention mechanism using a 7x7 convolution over pooled features (matching the source's implementation detail), but differs in its overall design (e.g., global aggregation vs. sequential channel-spatial refinement). Both address feature context exploitation in CNNs but through distinct architectural choices.
- **Abstract:** While the use of bottom-up local operators in convolutional neural networks (CNNs) matches well some of the statistics of natural images, it may also prevent such models from capturing contextual long-range feature interactions. In this work, we propose a simple, lightweight approach for better context exploitation in…

### [Tiled Squeeze-and-Excite: Channel Attention With Local Spatial Context](https://arxiv.org/abs/2107.02145)

- **Date:** 2021-07-06
- **arXiv:** `2107.02145`
- **Relation:** Implementation alternative
- **Why:** The candidate paper proposes Tiled Squeeze-and-Excite (TSE), which modifies the spatial context aggregation in channel attention mechanisms. While the source paper uses a 7×7 convolution for spatial attention, TSE replaces global pooling with local-tiled pooling, offering an alternative implementation for spatial context computation. Both address channel attention but differ in spatial aggregation
- **Abstract:** In this paper we investigate the amount of spatial context required for channel attention. To this end we study the popular squeeze-and-excite (SE) block which is a simple and lightweight channel attention mechanism. SE blocks and its numerous variants commonly use global average pooling (GAP) to create a single descr…

### [MCA: Moment Channel Attention Networks](https://arxiv.org/abs/2403.01713)

- **Date:** 2024-03-05
- **arXiv:** `2403.01713`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces a Cross Moment Convolution (CMC) module using channel-wise convolution to fuse multi-order moment information, which serves a similar spatial feature processing role as the source's 7x7 spatial attention convolution. However, the candidate's focus on statistical moment aggregation (e.g., first/second/third-order moments) differs fundamentally from the source's CBAM's
- **Abstract:** Channel attention mechanisms endeavor to recalibrate channel weights to enhance representation abilities of networks. However, mainstream methods often rely solely on global average pooling as the feature squeezer, which significantly limits the overall potential of models. In this paper, we investigate the statistica…

### [Partial Convolution Meets Visual Attention](https://arxiv.org/abs/2503.03148)

- **Date:** 2025-03-06
- **arXiv:** `2503.03148`
- **Relation:** Implementation alternative
- **Why:** The candidate paper introduces Partial Attention (PAT) blocks that use a 7x7 convolution for spatial attention over pooled features, matching the source's implementation detail. While the source uses this for sequential channel-spatial refinement (CBAM), the candidate applies it within partial convolution to improve efficiency, making it an alternative implementation approach.
- **Abstract:** Designing an efficient and effective neural network has remained a prominent topic in computer vision research. Depthwise onvolution (DWConv) is widely used in efficient CNNs or ViTs, but it needs frequent memory access during inference, which leads to low throughput. FasterNet attempts to introduce partial convolutio…
