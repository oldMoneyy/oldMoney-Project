Here is the Markdown formatted version of the paper.

***

# INFLLM-V2: DENSE-SPARSE SWITCHABLE ATTENTION FOR SEAMLESS SHORT-TO-LONG ADAPTATION

**Weilin Zhao$^1$, Zihan Zhou$^2$, Zhou Su$^2$, Chaojun Xiao$^{1*}$, Yuxuan Li$^2$, Yanghao Li$^1$, Yudi Zhang$^3$, Weilun Zhao$^2$, Zhen Li$^2$, Yuxiang Huang$^1$, Ao Sun$^2$, Xu Han$^{1*}$, Zhiyuan Liu$^{1*}$**
$^1$Tsinghua University  $^2$OpenBMB  $^3$Harbin Institute of Technology
`zwl23@mails.tsinghua.edu.cn` `{xcj,han-xu,liuzy}@tsinghua.edu.cn`

## ABSTRACT

Long-sequence processing is a critical capability for modern large language models. However, the self-attention mechanism in the standard Transformer architecture faces severe computational and memory bottlenecks when processing long sequences. While trainable sparse attention methods offer a promising solution, existing approaches such as NSA introduce excessive extra parameters and disrupt the conventional *pretrain-on-short, finetune-on-long* workflow, resulting in slow convergence and difficulty in acceleration. To overcome these limitations, we introduce dense-sparse switchable attention framework, termed as InfLLM-V2. InfLLM-V2 is a trainable sparse attention that seamlessly adapts models from short to long sequences. Specifically, InfLLM-V2 reuses dense attention parameters through parameter-free architecture modification, maintaining consistency between short and long sequence processing. Additionally, InfLLM-V2 ensures computational efficiency across all sequence lengths, by using dense attention for short inputs and smoothly transitioning to sparse attention for long sequences. To achieve practical acceleration, we further introduce an efficient implementation of InfLLM-V2 that significantly reduces the computational overhead. Our experiments on long-context understanding and chain-of-thought reasoning demonstrate that InfLLM-V2 is $4\times$ faster than dense attention while retaining 98.1% and 99.7% of the performance, respectively. Based on the InfLLM-V2 framework, we have trained and open-sourced MiniCPM4.1[^1], a hybrid reasoning model, providing a reproducible implementation for the research community.

---
*Corresponding Authors.
[^1]: https://huggingface.co/openbmb/MiniCPM4.1-8B

## 1 INTRODUCTION

With the rapid development of large language models (LLMs) (Brown et al., 2020; Bommasani et al., 2021; Han et al., 2021; OpenAI, 2023), the demand for long-sequence processing capabilities has become increasingly critical. From long-input scenarios such as deep research (Zheng et al., 2025; Xu & Peng, 2025), chatbots with long-term memory, and software issue resolution (Jimenez et al., 2023; Yang et al., 2025), to long-output tasks including complex reasoning (OpenAI et al., 2024; DeepSeek et al., 2025) and LLM-driven agents (Wang et al., 2024), a model's capability to understand and generate long sequences directly determines its performance in real-world applications. However, the self-attention mechanism in the existing Transformer (Vaswani et al., 2017) architecture faces severe computational and memory bottlenecks when processing long sequences.

To address the challenge of processing long sequences, efforts have been devoted to exploring sparse attention mechanisms (Beltagy et al., 2020; Zaheer et al., 2020; Tay et al., 2022), which restrict each token within the context to attend to only a subset of tokens related to that token. Early research in this area focuses on the training-free setting, leveraging the sparsity naturally occurring in self-attention mechanisms to accelerate inference (Xiao et al., 2024a;b; Jiang et al., 2024). However, the training-free setting introduces a fundamental trade-off between sparsity and model performance. To avoid significant performance degradation, the degree of sparsity that can be applied is often limited, which in turn restricts the potential efficiency gains.

**Figure 1:** The comparison of Vanilla Full Attention, NSA (Yuan et al., 2025), and our InfLLM-V2.

Given the limitations of training-free attention mechanisms, trainable sparse attention mechanisms have garnered increasing attention from researchers (Lu et al., 2025; Gao et al., 2024). Among them, the natively trainable sparse attention (NSA) (Yuan et al., 2025) method adopts the widely-used block-sparse attention (Child et al., 2019) structure, designing three different sparse attention modules and developing corresponding CUDA kernels to accelerate model computation. Despite its effectiveness, we find **misalignment between the sparse architecture of NSA and the standard pretrain-on-short, finetune-on-long workflow**. A widely used way to build long LLMs is to pretrain on short sequences and finetune on long sequences. The NSA creates an architectural mismatch with vanilla full attention, as it introduces three sets of key-value parameters and three attention modules, forcing the model to abruptly switch from a single-output attention to a multi-output attention architecture. As shown in Section 4, this mismatch destabilizes training, erases what the model has already learned, and introduces a significant efficiency bottleneck for short sequences.

To address all the above issues, we propose dense-sparse switchable attention framework (**InfLLM-V2**). InfLLM-V2 is built on InfLLM (Xiao et al., 2024a), a training-free block-sparse attention mechanism, and introduces three core innovations:

1. **Seamless Short-to-Long Adaptation:** As depicted in Figure 1, different from NSA, which requires additional parameters and multiple attention modules, InfLLM-V2 seamlessly transitions from dense to sparse attention by directly reusing existing dense attention parameters. This design naturally aligns with the standard pretrain-on-short, finetune-on-long workflow, eliminating architectural mismatches and training instability.
2. **Efficiency for Both Short and Long Sequences:** Because the transition from dense to sparse attention in InfLLM-V2 requires no additional parameters and introduces minimal distributional shifts, the model preserves its strong performance on short texts and can easily switch back to dense attention for short sequence efficiency.
3. **Accelerated Block Selection Mechanism:** The block selection step before sparse attention inherently undermines the efficiency gains of the sparse attention itself. We propose a hardware-awared efficient implementation, effectively removing the bottleneck and unlocking the full potential of sparse attention.

We evaluate our method on long-context understanding and long chain-of-thought (CoT) generation benchmarks. Our InfLLM-V2 is $4\times$ faster than dense attention while maintaining 98.1% and 99.7% of the original performance on these tasks, respectively. We will release all associated implementations to facilitate future research on efficient attention.

## 2 RELATED WORK

As the demand for LLMs to understand and generate long sequences continues to grow, research on improving attention efficiency has garnered increasing attention (Tay et al., 2022; Sun et al., 2025; Zhang et al., 2025a). In this section, we discuss the sparse attention paradigm from two perspectives: training-free and trainable sparse attention approaches.

### 2.1 TRAINING-FREE SPARSE ATTENTION

Training-free sparse attention approaches aim to utilize the intrinsic sparsity of attention layers. These methods enable LLMs trained with dense attention to perform sparse attention between each token and a small subset of relevant contexts. Based on the selection strategy for relevant contexts, these algorithms can be categorized into predefined sparse patterns and dynamic sparse patterns.

**Predefined Sparse Patterns.** Sparse attention with a predefined pattern employs manually defined heuristic rules to determine which contextual tokens should be selected for attention computation (Xiao et al., 2024b; Han et al., 2024; Child et al., 2019; Zaheer et al., 2020; Beltagy et al., 2020; Xiao et al., 2025). For instance, sliding window attention restricts each token to interact only with neighboring tokens (Beltagy et al., 2020). Building upon sliding windows, some works select special tokens such as initial tokens or segment separators, requiring all tokens to attend to these special tokens (Xiao et al., 2024b; Chen et al., 2024; Child et al., 2019). These approaches typically rely on human observations to formulate heuristic rules for selecting relevant contexts.

**Dynamic Sparse Patterns.** Dynamic sparse patterns incorporate the semantic information of query tokens into the context selection process by computing the relevance between query tokens and candidate contexts. Early works primarily perform similarity computation at the token level (Kitaev et al., 2020; Roy et al., 2021; Wang et al., 2020). As sequence lengths increase, block sparse methods have gained widespread adoption, which partition contexts into contiguous block units and perform relevance computation and context selection at the block granularity (Xiao et al., 2024a; Jiang et al., 2024; Xu et al., 2025; Tang et al., 2024; Zhang et al., 2025b; Lai et al., 2025). Furthermore, research on attention sparsity has inspired the development of key-value (KV) eviction and compression methods, which reduce memory consumption by discarding or compressing KV caches with low attention probabilities (Zhang et al., 2023; Li et al., 2024; Huang et al., 2024; 2025).

Training-free methods, while focusing on improving the inference efficiency of dense attention models, are often constrained by insufficient sparsity levels in order to avoid severe performance degradation and finally suffer from limited acceleration benefits.

### 2.2 TRAINABLE SPARSE ATTENTION

To further enhance efficiency for long sequence processing, researchers incorporate sparse attention into the model training phase. SeerAttention (Gao et al., 2024) employs a self-distillation post-training algorithm to train a router that selects relevant contexts for query blocks. MoBA (Lu et al., 2025) employs a block sparse attention structure during the short-to-long adaptation phase, training routers between query blocks and KV blocks for context selection. These methods partition query tokens into blocks and can only accelerate the prefilling phase. NSA (Yuan et al., 2025) designs three attention components for token-level sparsity, effectively accelerating both prefilling and decoding processes. However, NSA introduces substantial additional parameters, making it unsuitable for efficient short-to-long adaptation and imposing significant computational overhead on short-sequence processing. In this paper, we focus on proposing a sparse attention mechanism that effectively and efficiently processes both short and long sequences, supporting both prefilling and decoding.

## 3 METHOD

### 3.1 BACKGROUND

**Grouped-Query Attention.** Attention mechanisms enable models to selectively focus on relevant parts of the input sequence. Among various attention variants, grouped-query attention (GQA) (Ainslie et al., 2023) has emerged as a popular method that strikes a balance between model performance and computational efficiency. Given an input sequence of hidden states $\mathbf{X} \in \mathbb{R}^{n \times d}$, where $n$ is the sequence length and $d$ is the model dimension, GQA computes the queries ($\mathbf{Q}$), keys ($\mathbf{K}$), and values ($\mathbf{V}$) via linear projections as $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$. The projection matrices have the shapes $\mathbf{W}_Q \in \mathbb{R}^{d \times (h_q d_h)}$ and $\mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times (h_{kv} d_h)}$, with the head dimension $d_h$. These tensors are then reshaped to form $h_q$ query heads $\{\mathbf{Q}_i\}_{i=1}^{h_q}$, $h_{kv}$ KV heads $\{\mathbf{K}_j, \mathbf{V}_j\}_{j=1}^{h_{kv}}$, with each head having the shape $n \times d_h$. The query heads are partitioned by a group size $G = h_q / h_{kv}$. The attention scores $\mathbf{S}_i$ and the attention output $\mathbf{O}_i$ for the $i$-th query head are computed by attending to its corresponding KV heads with the index $j = \lfloor (i - 1) / G \rfloor + 1$:

$$
\mathbf{S}_i = \text{Softmax} \left(\frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d_h}}\right), \quad \mathbf{O}_i = \mathbf{S}_i \mathbf{V}_j. \quad \quad (1)
$$

The final output is obtained by concatenating the attention outputs and projecting them through a final linear layer $\mathbf{W}_O \in \mathbb{R}^{(h_q d_h) \times d}$: $\text{Attention}(\mathbf{X}) = \text{Concat}(\mathbf{O}_1, \dots, \mathbf{O}_{h_q})\mathbf{W}_O$.

**Figure 2:** The overview of NSA and InfLLM-V2. InfLLM-V2 uses a shared KV for both Sparse Attention and Dense Attention. InfLLM-V2 fuses Selected Attention and Sliding Attention and eliminates the output of Compressed Attention. InfLLM-V2 introduces no extra parameters.

**NSA.** NSA (Yuan et al., 2025) is an enhancement of GQA designed for efficiency on long sequences. The key insight is that for long sequences, e.g., when $n > 32k$, the attention score matrix $\mathbf{S}$ exhibits strong sparsity. This allows for approximating the attention matrix by ignoring negligible values, leading to faster computation. As illustrated in Figure 2, NSA utilizes three distinct modules and combines them using a gating module. Based on the observation that adjacent attention scores are similar (Jiang et al., 2024), NSA splits the sequences into blocks of size $B$. First, *Compressed Attention* employs a compressed representation of the KV tensors to reduce the computational complexity. Second, *Selected Attention* leverages the attention scores from compressed attention to compute only the blocks with high attention scores. Finally, *Sliding Attention* is used to focus on local contextual information within the sequence. For these three attention modes, they introduce three sets of KV projection matrices: $\mathbf{W}_K^{\text{cmp}}, \mathbf{W}_V^{\text{cmp}}, \mathbf{W}_K^{\text{slc}}, \mathbf{W}_V^{\text{slc}}, \mathbf{W}_K^{\text{win}}, \mathbf{W}_V^{\text{win}}$. This final output can be mathematically represented as $\text{Output} = g^{\text{cmp}}\mathbf{O}^{\text{cmp}} + g^{\text{slc}}\mathbf{O}^{\text{slc}} + g^{\text{win}}\mathbf{O}^{\text{win}}$, where $\mathbf{O}^{\text{cmp}}$, $\mathbf{O}^{\text{slc}}$, and $\mathbf{O}^{\text{win}}$ are the outputs of the three respective modules, and the gate scores $g^{\text{cmp}}$, $g^{\text{slc}}$, and $g^{\text{win}}$ are derived from the input features $\mathbf{X}$ via an MLP and a sigmoid activation. They also train an MLP module for compressing the KV tensors. The three distinct KV projections, combined with an additional MLP and gating module, result in a highly complex architecture. This complexity, in turn, makes the model poorly suited for training from scratch on short-sequence data and also complicates the process of converting pretrained dense models to sparse ones.

### 3.2 OVERALL FRAMEWORK

We propose InfLLM-V2, a more concise framework with zero extra parameters that more closely aligns dense and sparse attention patterns.

**Shared Key-Value Projection.** We find that using three separate sets of KV projection parameters in NSA (Yuan et al., 2025) is unnecessary, which not only complicates the adaptation from short to long sequences but also significantly slows down computation for short sequences. Therefore, we propose using a single shared set of projection parameters, $\mathbf{W}_K$ and $\mathbf{W}_V$, initialized with the pretrained dense attention parameters and used for finetuning on long sequences.

**Aligned Computation.** In addition to ensuring that sparse and dense attention share the same parameters, their computational processes must also be closely aligned. In NSA, the three attention modules all generate outputs that are aggregated by an extra gating module. This forces the computation of all three modules even for short sequences, leading to substantial overhead. To mitigate this, we take a union of the two sparse patterns in *Selected Attention* and *Sliding Attention* and eliminate the output of *Compressed Attention*, forming a unified *Sparse Attention* module. Specifically, the original *Selected Attention* module identifies important token blocks based on the attention scores from the *Compressed Attention* module, $\mathbf{S}^{\text{cmp}}$. For a query token with index $i$, located in the block $b_i = \lfloor \frac{i-1}{B} \rfloor + 1$, attention is always granted to a fixed set of initial blocks and a set of local blocks:

$$
\mathcal{I}_{\text{init}} = \{1, 2, \dots, N_{\text{init}}\}, \quad \mathcal{I}_{\text{local}}(i) = \{b_i - N_{\text{local}} + 1, \dots, b_i - 1, b_i\}. \quad \quad (2)
$$

The top-k selection is then applied to $\mathbf{S}^{\text{cmp}}$ over the set of remaining blocks, denoted as $\mathcal{I}_{\text{topk}}(i)$. The complete set of attended block indices for this query token is the union of these three sets:

$$
\mathcal{I}(i) = \mathcal{I}_{\text{init}} \cup \mathcal{I}_{\text{local}}(i) \cup \mathcal{I}_{\text{topk}}(i). \quad \quad (3)
$$

If we denote the set of token indices in the $j$-th block as $\mathcal{T}_j = \{jB + 1, \dots, (j + 1)B\}$, the selected attention allows a token in the block $b_i$ to attend to the union of blocks $\bigcup_{j \in \mathcal{I}(i)} \mathcal{T}_j$. The *Sliding Attention*, on the other hand, allows the $i$-th token to attend to a range $\{i - w + 1, \dots, i\}$ of window size $w$. Since the local blocks in *Selected Attention* and the window in *Sliding Attention* create overlapping, we merge them by expanding the number of local blocks within our unified *Sparse Attention* to strictly cover the region of the *Sliding Attention*, that is, $N_{\text{local}} \ge \lceil \frac{w}{B} \rceil + 1$, as illustrated in Figure 3.

**Figure 3:** The illustration of the union of *Selected Attention* and *Sliding Attention*.

Furthermore, we eliminate the output of the *Compressed Attention* module, only retaining its attention scores $\mathbf{S}^{\text{cmp}}$ for block selection in *Sparse Attention*. This single-output design more closely mirrors dense attention and aids the training of the sparse attention model. InfLLM-V2 can thus dynamically switch between dense and sparse attention patterns based on the input sequence length.

**Simplified and Efficient Compression Module.** Since we eliminate the output of the *Compression Attention*, using MLP for token compression would not receive gradients. We replace it with a more intuitive parameter-free pooling function, which will be detailed in Section 3.3. Additionally, computing the attention scores $\mathbf{S}^{\text{cmp}}$ introduces non-negligible overhead, and we will reduce this overhead in Section 3.4.

### 3.3 BLOCK REPRESENTATION

**Figure 4:** The illustration of the 3-stage group-level compression, compared with the 1-stage token-level compression.

Simply compressing a long sequence with a large block size $B$ in 1-stage can lead to a significant loss of granular information (Yuan et al., 2025). To address this, we implement a 3-stage, coarse-grained to fine-grained compression process, as shown in Figure 4. In the first stage, we process the input key sequence $\mathbf{K}$ to produce an intermediate and coarse-grained representation $\mathbf{K}^{C_1}$. By denoting the initial compression block size as $l_{C_1}$ and the stride as $s_{C_1}$, we achieve this by applying a **mean-pooling** operation over sequential blocks:

$$
\mathbf{K}_i^{C_1} = \text{Mean}(\mathbf{K}_{i \cdot s_{C_1} : i \cdot s_{C_1} + l_{C_1}}). \quad \quad (4)
$$

Then, we compute the attention scores $\mathbf{S}^{C_1}$ between the query $\mathbf{Q}$ and $\mathbf{K}^{C_1}$:

$$
\mathbf{S}^{C_1} = \text{Softmax}(\mathbf{Q}(\mathbf{K}^{C_1})^\top). \quad \quad (5)
$$

In the second stage, we employ block-wise sparse attention rather than token-level approaches for the efficiency of *Sparse Attention*. In a model utilizing GQA, we can achieve this by forcing the block selection pattern across all heads within a group to be the same. We conduct **summation** within the head group to get the shared importance score $\mathbf{S}^{\text{shared}}$:

$$
\mathbf{S}^{\text{shared}} = \sum_{h=1}^G \mathbf{S}^{C_1}(h). \quad \quad (6)
$$

In the third stage, we apply a **max-pooling** operation, which can preserve the most salient features. The aggregated score $\mathbf{S}^{\text{cmp}}$ are defined as follows and used for the *Sparse Attention*:

$$
\mathbf{S}_i^{\text{cmp}} = \text{Max}(\mathbf{S}_{i \cdot s : i \cdot s + l}^{\text{shared}}). \quad \quad (7)
$$

---
**Algorithm 1** Computation of $\mathbf{S}^{\text{shared}}$ (Suppose $h_{kv} = 1$ without loss of generality.)
**Require:** $\mathbf{Q} \in \mathbb{R}^{n \times G \times d_h}, \mathbf{K}^{C_1} \in \mathbb{R}^{(n/s_{C_1}) \times d_h}, \mathbf{K}^{C_2} \in \mathbb{R}^{(n/s_{C_2}) \times d_h}$ in HBM. Block sizes $B_q, B_k$.
Divide $\mathbf{Q}$ into $T_q = \lceil n/B_q \rceil$ blocks $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_q}$ of size $B_q \times G \times d_h$ each.
Divide $\mathbf{K}^{C_1}$ into $T_1 = \lceil n/s_{C_1}/B_k \rceil$ blocks $\mathbf{K}^{C_1}_1, \dots, \mathbf{K}^{C_1}_{T_1}$ of size $B_k \times d_h$ each.
Divide $\mathbf{K}^{C_2}$ into $T_2 = \lceil n/s_{C_2}/B_k \rceil$ blocks $\mathbf{K}^{C_2}_1, \dots, \mathbf{K}^{C_2}_{T_2}$ of size $B_k \times d_h$ each.
Divide $\mathbf{S}^{\text{shared}}$ into $T_q \times T_1$ blocks of size $B_q \times B_k$ each.
**for** $i = 1, \dots, T_q$ (parallel) **do**
&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{Q}_i$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, initialize online-softmax related statistic log-sum-exp $lse$.
&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1, \dots, T_2$ (sequential) **do** $\quad\quad\quad \triangleright$ First pass (Coarse-grained)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{K}^{C_2}_j$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute attention scores $\mathbf{S}^{C_2}_{ij} \in \mathbb{R}^{G \times B_q \times B_k}$ as in Eq. (8) and update $lse$.
&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1, \dots, T_1$ (sequential) **do** $\quad\quad\quad \triangleright$ Second pass (Fine-grained)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{K}^{C_1}_j$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute attention scores $\mathbf{S}^{C_1}_{ij} \in \mathbb{R}^{G \times B_q \times B_k}$ as in Eq. (5) and normalize it using $lse$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute the final block $\mathbf{S}^{\text{shared}}_{ij} \in \mathbb{R}^{B_q \times B_k}$ by summing $\mathbf{S}^{C_1}_{ij}$ over the head group.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Write the block $\mathbf{S}^{\text{shared}}_{ij}$ to its corresponding position in HBM.
**return** the output $\mathbf{S}^{\text{shared}}$.
---

In our method, we set $l_{C_1} = \frac{B}{2}$, $s_{C_1} = \frac{B}{4}$, $l = 5$, and $s = 4$ so that it can achieve the same compression ratio as 1-stage compression of block size $B$. Intuitively, we compute the sparse scores of the entire block based on several sliding sub-blocks within the block.

### 3.4 EFFICIENT IMPLEMENTATION

For efficient *Sparse Attention*, we follow the techniques in NSA (Yuan et al., 2025) to set the group size $G$ of GQA to 16, a configuration well-suited for block sparse attention. More details can be found in Appendix A. **However, our profiling reveals that the computation of the compression score, $\mathbf{S}^{\text{cmp}}$, introduces a significant performance bottleneck.** A primary source of this slowdown is the substantial I/O required to store the first-stage attention scores $\mathbf{S}^{C_1}$ into the slow GPU HBM. The amount of data that needs to be written is $h_q n^2 / s_{C_1}$, where $n$ is the full sequence length. Given that $s_{C_1} \ll n$, materializing the full attention score matrix to GPU HBM incurs a prohibitive cost.

Drawing inspiration from FlashAttention (Dao, 2024), we aim to minimize this I/O by ensuring the attention scores remain within the fast GPU SRAM as much as possible. Our approach, ***Fused Head Group Summation***, is to fuse the summation over the head group, required for the second-stage compression, directly into the SRAM-based computation loop of FlashAttention. After that, we can only store the reduced attention scores $\mathbf{S}^{\text{shared}}$ into GPU HBM, whose size is $h_q n^2 / (s_{C_1} G)$.

Another challenge arises from the fact that summing over the head group dimension and performing the online-softmax (Dao, 2024) along the sequence dimension are not commutative operations. This conflict prevents a straightforward fusion. To overcome this, we implement a two-pass approach. In the first pass, we compute the log-sum-exp ($lse$) term required for the softmax normalization within the SRAM. In the second pass, we leverage the $lse$ to calculate the final attention scores, perform the summation across the head group within the SRAM, and write the reduced scores to the HBM. The trade-off of this two-pass method is that it doubles the computational workload. Therefore, we propose ***LSE Approximation*** to approximate the $lse$ computation by using a coarser-grained attention score $\mathbf{S}^{C_2}$. Following Eq. (4) and Eq. (5), we change them to

$$
\mathbf{K}_i^{C_2} = \text{Mean}(\mathbf{K}_{i \cdot s_{C_2} : i \cdot s_{C_2} + l_{C_2}}), \quad \mathbf{S}^{C_2} = \text{Softmax}(\mathbf{Q}(\mathbf{K}^{C_2})^\top). \quad \quad (8)
$$

By setting $s_{C_2} = 4 s_{C_1}$ and $l_{C_2} = 4 l_{C_1}$, the computational overhead was reduced from $2\times$ to $1.25\times$. We summarize the procedure for computing $\mathbf{S}^{\text{shared}}$ in Algorithm 1. To further reduce memory I/O, the max-pooling and top-k operations related to $\mathbf{S}^{\text{cmp}}$ could also be fused into the kernel; however, we leave this implementation for future work.

**Figure 5:** The training loss of models. We only show the last few iterations of the short pretraining.

## 4 EXPERIMENT

We evaluate InfLLM-V2 on tasks ranging from short to long contexts, and demonstrate its efficiency.

### 4.1 EXPERIMENT SETUP

**Pretraining Setup.** We first use full attention to pretrain a model on short-sequence data, marked as SHORT. We employ a standard GQA (Ainslie et al., 2023) model backbone with 8B parameters, with the hidden size $d = 4096$, the number of heads $h_q = 32$, $h_{kv} = 2$, and the head dimension $d_h = 128$. The pretraining dataset consists of 8T tokens of 4k-length sequences, primarily comprising FineWeb-Edu (Penedo et al., 2024) and Stack-v2 (Lozhkov et al., 2024). We set 8M tokens per batch, and use a WSD learning rate scheduler (Hu et al., 2024) with 2000 warmup steps to an initial learning rate of 7.5e-3, and 27000 decay steps to the final learning rate of 3e-4.

**Long-Context Adaptation.** When transitioning to long-context finetuning, we switch to INFLLM-V2 (SPARSE). Following NSA (Yuan et al., 2025), we set the compression block size $l_{C_1} = 32$, stride $s_{C_1} = 16$, and attention block size $B = 64$. For our efficient block selection implementation in Section 3.4, we additionally set the LSE Approximation block size $l_{C_2} = 128$ and stride $s_{C_2} = 64$. We set the selected block count $|\mathcal{I}| = 96$ (including $|\mathcal{I}_{\text{init}}| = 1$, $|\mathcal{I}_{\text{topk}}| = 63$, and $|\mathcal{I}_{\text{local}}| = 32$) for both training and inference. Therefore, the total number of visible tokens is $|\mathcal{I}| \cdot B = 6k$. We conduct long-sequence finetuning on the pretrained model using 5B tokens, with an initial learning rate of 3e-4 and linear decay to 2.75e-4. The training batches contain sequences from four length intervals: 0-4k, 4-12k, 12-24k, and 24-32k, with token counts in a 1:1:1:1 ratio.

**Baselines.** We finetune a baseline model with full attention, marked as FULLATTN, using the same training configuration as INFLLM-V2 (SPARSE). We then apply several typical training-free sparse attention methods on FULLATTN as baselines, including InfLLM (Xiao et al., 2024a) and MInference (Jiang et al., 2024). In addition, we present the results of SHORT with YaRN (Peng et al., 2023) to extend the context window size. In terms of trainable sparse attention, we compare with NSA (Yuan et al., 2025). By using the same training settings as in INFLLM-V2 (SPARSE), we finetune our pretrained model into an NSA version. We initialize NSA's three sets of KV parameters by replicating the original KV parameters in dense attention. As NSA does not publish their code, we adopt an open-source Triton implementation of NSA for experiments[^2].

**Table 1:** Task Performance on RULER. Best results in sparse attention are bolded.

| Method | SG1 | SG2 | SG3 | MK1 | MK2 | MK3 | MV | MQ | VT | CWE | FWE | QA1 | QA2 | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FULLATTN | 100.00 | 100.00 | 100.00 | 96.00 | 94.00 | 92.00 | 82.00 | 98.50 | 93.20 | 44.40 | 91.33 | 48.00 | 56.00 | 84.26 |
| SHORT+YARN | 98.00 | 68.00 | 50.00 | 46.00 | 6.00 | 0.00 | 32.00 | 31.50 | 36.00 | 21.40 | 87.33 | 26.00 | 26.00 | 40.63 |
| INFLLM | 98.00 | 6.00 | 4.00 | 10.00 | 10.00 | 10.00 | 9.00 | 7.50 | 70.00 | 16.00 | 80.67 | 18.00 | 24.00 | 27.94 |
| MINFERENCE | **100.00** | **100.00** | **100.00** | 76.00 | 36.00 | 46.00 | 79.50 | 93.50 | 88.00 | **64.20** | **92.67** | 32.00 | **44.00** | 73.22 |
| NSA | **100.00** | 88.00 | 82.00 | 54.00 | 38.00 | 30.00 | 59.00 | 61.50 | 56.00 | 34.40 | 86.00 | 56.00 | 34.00 | 59.92 |
| INFLLM-V2 (SPARSE) | | | | | | | | | | | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;w/ LSE Approx | **100.00** | **100.00** | **100.00** | **94.00** | **82.00** | 62.00 | **98.50** | 94.50 | **98.00** | 50.40 | 82.67 | **72.00** | 40.00 | **82.62** |
| &nbsp;&nbsp;&nbsp;&nbsp;w/o LSE Approx | **100.00** | **100.00** | **100.00** | 92.00 | 80.00 | **64.00** | **98.50** | **95.50** | **98.00** | 47.80 | 81.33 | 70.00 | 40.00 | 82.09 |
| INFLLM-V2 (DENSE) | 100.00 | 100.00 | 100.00 | 94.00 | 98.00 | 98.00 | 99.00 | 98.00 | 98.40 | 52.80 | 90.00 | 76.00 | 44.00 | 88.32 |

[^2]: https://github.com/XunhaoLai/native-sparse-attention-triton

**Table 2:** Task Performance on LongBench and LongPPL. Best results in sparse attention are bolded.

| Benchmark | FULLATTN | SHORT+YARN | INFLLM | MINFERENCE | NSA | INFLLM-V2 (SPARSE) | INFLLM-V2 (DENSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LongBench $\uparrow$ | 42.30 | 37.86 | 32.30 | 41.55 | 37.10 | **42.54** | 42.49 |
| LongPPL $\downarrow$ | 2.06 | 5.28 | 12.01 | 2.62 | 4.24 | **2.12** | 2.00 |

For all the above sparse attention methods, we maintain the same sparsity level to ensure a fair comparison. We provide the training curve for trainable methods in Figure 5. NSA causes a disruption in the loss, while INFLLM-V2 is closer to FULLATTN.

### 4.2 TASK PERFORMANCE

In this section, we evaluate InfLLM-V2 and other baselines across various tasks. Notably, while the original NSA paper demonstrates performance comparable to full attention when training on long sequences from scratch, NSA fails to achieve satisfactory results in short-to-long adaptation settings. *This indicates that the substantial parameter overhead introduced by NSA renders it unsuitable for the conventional "pretraining-on-short, finetuning-on-long" paradigm.*

**Long-Context Understanding.** To evaluate InfLLM-V2's performance on long-input tasks, we compare InfLLM-V2 and different baselines on RULER (Hsieh et al., 2024), LongBench (Bai et al., 2024) and LongPPL (Fang et al., 2025). RULER is a synthetic dataset with a configurable average length. LongBench is a bilingual benchmark for long-context understanding. Compared to RULER, LongBench is primarily built from existing, real-world datasets. LongPPL is a perplexity evaluation benchmark for long sequences. The experimental results of RULER when the length is 32k are shown in Table 1. The results on LongBench and LongPPL are shown in Table 2. Please refer to Appendix B for detailed performance of the sub-tasks in LongBench. From the results, we can observe that: 1) INFLLM-V2 achieves the best performance compared to other sparse methods, with its results being highly competitive and closely matching the strong, FULLATTN baseline. Alternative approaches, whether applying training-free sparsity or training-based sparsity, result in a substantial drop in performance. 2) Compared to NSA, INFLLM-V2 can achieve significant performance improvements through minimal finetuning on long-sequences. Although NSA has low training loss, its high perplexity on the LongPPL evaluations indicates that NSA has not adequately learned long-range dependencies. 3) A unique advantage of INFLLM-V2 is the flexibility to seamlessly switch between dense mode and sparse mode. This flexibility not only provides an option for dense computation but can also lead to a further improvement in performance, surpassing even the full attention baseline. 4) Furthermore, the INFLLM-V2 (SPARSE) variant with LSE Approximation does not lose any performance, confirming the effectiveness of our acceleration technique.

**Table 3:** Task Performance on Long Reasoning Tasks.

| Method | MATH-500 | AIME 24 | AIME 25 | LCB v5 | LCB v6 | Avg. $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FULLATTN | 86.00 | 37.50 | 30.63 | 30.67 | 29.14 | 42.79 |
| NSA | 83.80 | 28.75 | 23.54 | 25.15 | 25.14 | 37.28 |
| InfLLM-V2 (Sparse) | 87.80 | 38.33 | 29.38 | 29.94 | 27.83 | 42.66 |
| InfLLM-V2 (Dense) | 86.40 | 36.67 | 23.33 | 29.94 | 26.29 | 40.53 |

**Long Reasoning.** To evaluate the performance of InfLLM-V2 in long-output scenarios, we compared several major Long Reasoning tasks, including MATH-500 (Hendrycks et al., 2021b), AIME (MAA), and LiveCodeBench (LCB) (Jain et al., 2025). We finetune InfLLM-V2 and baselines on OpenMathReasoning (Moshkov et al., 2025) and OpenCodeReasoning (Ahmad et al., 2025). As InfLLM and MInference primarily accelerate long-input processing, we exclude them from this long-output evaluation. The experimental results are shown in Table 3. The results show that InfLLM-V2 attains performance on par with full attention, confirming its effectiveness for long-output scenarios.

**General Tasks.** We verify that the InfLLM-V2 architecture can freely switch back to Dense mode without performance degradation on short-sequence tasks after long-sequence fine-tuning. Zero-shot evaluations on MMLU (Hendrycks et al., 2021a), MMLU-Redux (Gema et al., 2025), CEval (Huang et al., 2023), MATH-500 (Hendrycks et al., 2021b), HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021) and BBH (Suzgun et al., 2023) are shown in Table 4. Experimental results show that InfLLM-V2 achieves performance comparable to full attention.

### 4.3 EFFICIENCY

We first evaluate the efficiency of our kernel implementation on NVIDIA A100 and NVIDIA 4090. We evaluate InfLLM-V2's inference efficiency on the batch=1 setting. We select FlashAttention-2 (Dao, 2024) implementation for full attention. For a fair efficiency comparison with NSA, we ignore its sliding attention component, and compare solely on the compression and sparse attention parts by selecting an equal number of blocks $|\mathcal{I}|$. Experiment results are shown in Figure 6. When the number of selected blocks is 16, InfLLM-V2 achieves up to $7.4\times$ over FlashAttention on A100 and $9.3\times$ on 4090. In contrast, NSA's speedup is limited to $3.5\times$ in the same setting. The breakdown of the execution time shows that the overhead from the *Block Selection* stage is greatly optimized by our efficient implementation in Section 3.4. We further conduct an ablation study on the *Block Selection*, as shown in Table 5, which shows the effectiveness of our proposed *LSE Approximation*.

**Table 4:** Task Performance on General Tasks.

| Method | MMLU | MMLU-Redux | CEval | MATH-500 | HumanEval | MBPP | BBH | Avg. $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SHORT | 72.73 | 72.71 | 76.17 | 54.40 | 70.73 | 75.49 | 51.90 | 67.73 |
| FULLATTN | 73.38 | 70.24 | 78.11 | 54.60 | 71.34 | 75.10 | 49.13 | 67.41 |
| NSA | 68.27 | 66.39 | 74.33 | 44.40 | 62.20 | 65.00 | 43.81 | 60.63 |
| InfLLM-V2 (Dense) | 71.29 | 69.73 | 77.70 | 54.80 | 73.17 | 73.54 | 47.09 | 66.76 |

**Figure 6:** Speed of the kernels on NVIDIA A100 and NVIDIA 4090.

**Table 5:** Ablation study of *Block Selection* efficiency, with and without *LSE Approximation*. All measurements are in time (ms), and the number of selected blocks is set to 16.

| Device | A100 (32k) | A100 (64k) | A100 (96k) | A100 (128k) | 4090 (32k) | 4090 (64k) | 4090 (96k) | 4090 (128k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| w/o LSE Approximation | 4.67 | 18.20 | 42.46 | 75.36 | 4.89 | 19.95 | 46.51 | 83.26 |
| w/ LSE Approximation | **3.93** | **14.07** | **32.44** | **56.59** | **3.70** | **14.39** | **33.16** | **59.04** |

The end-to-end inference speed (with a $|\mathcal{I}| = 96$ and W4A16 quantization (Frantar et al., 2025)) is shown in Figure 7. InfLLM-V2 can achieve $2.13\times$ prefilling speedup and $2.32\times$ decoding speedup. Since InfLLM-V2 does not accelerate the Feed-Forward Network (FFN) layers, a higher speedup ratio can be achieved by incorporating FFN-specific acceleration techniques in future work.

**Figure 7:** End-to-end inference speed of our 8B model when the number of visible tokens is 6k. TTFT means time-to-first-token, and TPOT means time-per-output-token.

## 5 CONCLUSION

In this paper, we introduced InfLLM-V2, a dense-sparse switchable attention framework designed to overcome the limitations of existing trainable sparse attention mechanisms. By ensuring architectural alignment with the standard pretrain-on-short and finetune-on-long workflow, InfLLM-V2 facilitates a seamless and efficient sparse adaptation to long contexts without requiring extra parameters or causing disruptive distributional shifts. We believe InfLLM-V2 offers a practical and powerful solution for advancing the capabilities of large language models in the long-context era.

## REFERENCES

Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding. *arXiv preprint arXiv:2504.01943*, 2025.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 4895–4901, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3119–3137, 2024.

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. *Preprint*, 2021.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In *Advances in Neural Information Processing Systems*, volume 33, pp. 1877–1901, 2020.

Guoxuan Chen, Han Shi, Jiawei Li, Yihang Gao, Xiaozhe Ren, Yimeng Chen, Xin Jiang, Zhenguo Li, Weiyang Liu, and Chao Huang. Sepllm: Accelerate large language models by compressing one segment into one separator. In *Forty-second International Conference on Machine Learning*, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*, 2019.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In *The Twelfth International Conference on Learning Representations*, 2024.

Daya DeepSeek, Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Lizhe Fang, Yifei Wang, Zhaoyang Liu, Chenheng Zhang, Stefanie Jegelka, Jinyang Gao, Bolin Ding, and Yisen Wang. What is wrong with perplexity for long-context language modeling? In *The Thirteenth International Conference on Learning Representations*, 2025.

Elias Frantar, Roberto L Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. Marlin: Mixed-precision auto-regressive parallel inference on large language models. In *Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming*, pp. 239–251, 2025.

Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden Kwok-Hay So, Ting Cao, Fan Yang, et al. Seerattention: Learning intrinsic sparse attention in your llms. *arXiv preprint arXiv:2410.13276*, 2024.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 5069–5096, 2025.

Chi Han, Qifan Wang, Hao Peng, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Zero-shot extreme length generalization for large language models. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 3991–4008, 2024.

Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Yuan Yao, Ao Zhang, Liang Zhang, et al. Pre-trained models: Past, present and future. *AI Open*, 2:225–250, 2021.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations*, 2021a.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021b.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. Ruler: What's the real context size of your long-context language models? In *First Conference on Language Modeling*, 2024.

Shengding Hu, Yuge Tu, Xu Han, Ganqu Cui, Chaoqun He, Weilin Zhao, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. In *First Conference on Language Modeling*, 2024.

Yuxiang Huang, Binhang Yuan, Xu Han, Chaojun Xiao, and Zhiyuan Liu. Locret: Enhancing eviction in long-context llm inference with trained retaining heads on consumer-grade devices. *arXiv preprint arXiv:2410.01805*, 2024.

Yuxiang Huang, Mingye Li, Xu Han, Chaojun Xiao, Weilin Zhao, Ao Sun, Hao Zhou, Jie Zhou, Zhiyuan Liu, and Maosong Sun. APB: Accelerating distributed long-context inference by passing compressed context blocks across GPUs. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 10708–10727, 2025.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. *Advances in Neural Information Processing Systems*, 36:62991–63010, 2023.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In *The Thirteenth International Conference on Learning Representations*, 2025.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. *Advances in Neural Information Processing Systems*, 37:52481–52515, 2024.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. Swe-bench: Can language models resolve real-world github issues? In *The Twelfth International Conference on Learning Representations*, 2023.

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In *International Conference on Learning Representations*, 2020.

Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. In *The Thirteenth International Conference on Learning Representations*, 2025.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. *Advances in Neural Information Processing Systems*, 37:22947–22970, 2024.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. *arXiv preprint arXiv:2402.19173*, 2024.

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms. *arXiv preprint arXiv:2502.13189*, 2025.

MAA. American invitational mathematics examination-aime. URL `https://maa.org/maa-invitational-competitions/`.

Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. *arXiv preprint arXiv:2504.16891*, 2025.

OpenAI. GPT-4 technical report. *Preprint*, 2023.

Aaron OpenAI, Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. *arXiv preprint arXiv:2412.16720*, 2024.

Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. *Advances in Neural Information Processing Systems*, 37:30811–30849, 2024.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. In *The Twelfth International Conference on Learning Representations*, 2023.

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. *Transactions of the Association for Computational Linguistics*, 9:53–68, 2021.

Yutao Sun, Zhenyu Li, Yike Zhang, Tengyu Pan, Bowen Dong, Yuyi Guo, and Jianyong Wang. Efficient attention mechanisms for large language models: A survey. *arXiv preprint arXiv:2507.19595*, 2025.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. In *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 13003–13051, 2023.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity for efficient long-context llm inference. In *Forty-first International Conference on Machine Learning*, 2024.

Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey. *ACM Comput. Surv.*, 55(6), 2022. ISSN 0360-0300. doi: 10.1145/3530811.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6):186345, 2024.

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. *arXiv preprint arXiv:2006.04768*, 2020.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. *Advances in Neural Information Processing Systems*, 37:119638–119661, 2024a.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In *The Twelfth International Conference on Learning Representations*, 2024b.

Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Shang Yang, Haotian Tang, Yao Fu, Song Han, et al. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. In *The Thirteenth International Conference on Learning Representations*, 2025.

Renjun Xu and Jingwen Peng. A comprehensive survey of deep research: Systems, methodologies, and applications. *arXiv preprint arXiv:2506.12594*, 2025.

Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. Xattention: Block sparse attention with antidiagonal scoring. In *Forty-second International Conference on Machine Learning*, 2025.

John Yang, Kilian Lieret, Carlos E Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. Swe-smith: Scaling data for software engineering agents. *arXiv preprint arXiv:2504.21798*, 2025.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. *arXiv preprint arXiv:2502.11089*, 2025.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. *Advances in neural information processing systems*, 33:17283–17297, 2020.

Jintao Zhang, Rundong Su, Chunyu Liu, Jia Wei, Ziteng Wang, Pengle Zhang, Haoxu Wang, Huiqiang Jiang, Haofeng Huang, Chendong Xiang, et al. A survey of efficient attention methods: Hardware-efficient, sparse, compact, and linear attention. 2025a.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Haocheng Xi, Jun Zhu, Jianfei Chen, et al. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. In *Forty-second International Conference on Machine Learning*, 2025b.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36:34661–34710, 2023.

Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. *arXiv preprint arXiv:2504.03160*, 2025.

## A IMPLEMENTATION DETAIL

We have shown the implementation of *Block Selection* in Section 3.4. We show the implementation detail of *Sparse Attention* here in Algorithm 2.

---
**Algorithm 2** Computation of *Sparse Attention*. (Suppose $h_{kv} = 1$ without loss of generality.)
**Require:** $\mathbf{Q} \in \mathbb{R}^{n \times G \times d_h}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_h}$. Block sizes $B_k$.
Divide $\mathbf{Q}$ into $n$ blocks $\mathbf{Q}_1, \dots, \mathbf{Q}_n$ of size $G \times d_h$ each.
Divide $\mathbf{K}, \mathbf{V}$ into $T_k = \lceil n/B_k \rceil$ blocks $\mathbf{K}_1, \dots, \mathbf{K}_{T_k}$ and $\mathbf{V}_1, \dots, \mathbf{V}_{T_k}$ of size $B_k \times d_h$ each.
Divide $\mathbf{O} \in \mathbb{R}^{n \times G \times d_h}$ into $n$ blocks of size $G \times d_h$ each.
Divide the log-sum-exp $lse$ into $n$ blocks of size $G$ each.
**for** $i = 1, \dots, n$ (parallel) **do**
&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{Q}_i$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, initialize $\mathbf{O}_i^{(0)} = (\mathbf{0})_{G \times d_h}, \ell_i^{(0)} = (\mathbf{0})_G, m_i^{(0)} = (-\infty)_G$.
&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1, \dots, T_k$ (sequential) **do**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\mathbf{K}_j$ in visible tokens (determined by the $|\mathcal{I}(i)|$ in Eq. 3) **then**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{K}_j, \mathbf{V}_j$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute attention scores $\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{G \times B_k}$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $m_i^{(j)} = \max(m_i^{(j-1)}, \text{rowmax}(\mathbf{S}_{ij})) \in \mathbb{R}^G$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - m_i^{(j)}) \in \mathbb{R}^{G \times B_k}$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\ell_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)})\ell_i^{(j-1)} + \text{rowsum}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^G$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\mathbf{O}_i^{(j)} = \text{diag}(\exp(m_i^{(j-1)} - m_i^{(j)}))^{-1} \mathbf{O}_i^{(j-1)} + \tilde{\mathbf{P}}_{ij} \mathbf{V}_j$.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\mathbf{O}_i = \text{diag}(\ell_i^{(T_k)})^{-1} \mathbf{O}_i^{(T_k)}$.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $lse_i = m_i^{(T_k)} + \log(\ell_i^{(T_k)})$.
&nbsp;&nbsp;&nbsp;&nbsp;Write $\mathbf{O}_i$ to HBM as the $i$-th block of $\mathbf{O}$.
&nbsp;&nbsp;&nbsp;&nbsp;Write $lse_i$ to HBM as the $i$-th block of $lse$.
**return** the output $\mathbf{O}$ and the log-sum-exp $lse$.
---

---
**Algorithm 3** Computation of *Dense Attention*. (Suppose $h_{kv} = 1$ without loss of generality.)
**Require:** $\mathbf{Q} \in \mathbb{R}^{n \times G \times d_h}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_h}$. Block sizes $B_q, B_k$.
Divide $\mathbf{Q}$ into $T_q = G \times \lceil n/B_q \rceil$ blocks $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_q}$ of size $B_q \times d_h$ each.
Divide $\mathbf{K}, \mathbf{V}$ into $T_k = \lceil n/B_k \rceil$ blocks $\mathbf{K}_1, \dots, \mathbf{K}_{T_k}$ and $\mathbf{V}_1, \dots, \mathbf{V}_{T_k}$ of size $B_k \times d_h$ each.
Divide $\mathbf{O} \in \mathbb{R}^{n \times G \times d_h}$ into $T_q$ blocks of size $B_q \times d_h$ each.
Divide the log-sum-exp $lse$ into $T_q$ blocks of size $B_q$ each.
**for** $i = 1, \dots, T_q$ (parallel) **do**
&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{Q}_i$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, initialize $\mathbf{O}_i^{(0)} = (\mathbf{0})_{B_q \times d_h}, \ell_i^{(0)} = (\mathbf{0})_{B_q}, m_i^{(0)} = (-\infty)_{B_q}$.
&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1, \dots, T_k$ (sequential) **do**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Load $\mathbf{K}_j, \mathbf{V}_j$ from HBM to on-chip SRAM.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute attention scores $\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_q \times B_k}$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $m_i^{(j)} = \max(m_i^{(j-1)}, \text{rowmax}(\mathbf{S}_{ij})) \in \mathbb{R}^{B_q}$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - m_i^{(j)}) \in \mathbb{R}^{B_q \times B_k}$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\ell_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)})\ell_i^{(j-1)} + \text{rowsum}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^{B_q}$.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\mathbf{O}_i^{(j)} = \text{diag}(\exp(m_i^{(j-1)} - m_i^{(j)}))^{-1} \mathbf{O}_i^{(j-1)} + \tilde{\mathbf{P}}_{ij} \mathbf{V}_j$.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $\mathbf{O}_i = \text{diag}(\ell_i^{(T_k)})^{-1} \mathbf{O}_i^{(T_k)}$.
&nbsp;&nbsp;&nbsp;&nbsp;On chip, compute $lse_i = m_i^{(T_k)} + \log(\ell_i^{(T_k)})$.
&nbsp;&nbsp;&nbsp;&nbsp;Write $\mathbf{O}_i$ to HBM as the $i$-th block of $\mathbf{O}$.
&nbsp;&nbsp;&nbsp;&nbsp;Write $lse_i$ to HBM as the $i$-th block of $lse$.
**return** the output $\mathbf{O}$ and the log-sum-exp $lse$.
---

Similar to FlashAttention (Dao, 2024), the algorithm divides the input into blocks. The differences are: 1) The FlashAttention block size $B_k$ of $\mathbf{K}$, should divide the sparse attention block size $B$. That is, $B$ should be a multiple of $B_k$. 2) The FlashAttention block of $\mathbf{Q}$ typically contains a single attention head and multiple tokens. We follow NSA (Yuan et al., 2025) to make it contain a group of attention heads of a single token, so that they can share the same sparse pattern. 3) The inner loop of the FlashAttention iterates over all blocks of $\mathbf{K}$, whereas our method's loop only covers the visible blocks of the sparse attention. We also show the FlashAttention implementation of Dense Attention to Algorithm 3 for reference.

## B BENCHMARK DETAILS

We provide the detailed results of the LongBench benchmark, mentioned in Table 2, in Table 6. Following LongBench (Bai et al., 2024), the "Overall" score is computed by the macro-average over the six task categories.

**Table 6:** Task Performance on LongBench. Best results in sparse attention are bolded.

| Category | Task | FULLATTN | SHORT + YARN | INFLLM | MINFERENCE | NSA | INFLLM-V2 (SPARSE) | INFLLM-V2 (DENSE) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-Doc QA | NarQA <br> Qasper <br> MFQA-en <br> MFQA-zh | 21.38 <br> 43.80 <br> 55.07 <br> 57.26 | 18.17 <br> 30.98 <br> 43.81 <br> 54.51 | **21.02** <br> 34.92 <br> 49.39 <br> 51.75 | 20.16 <br> 44.51 <br> **54.83** <br> 57.00 | 18.34 <br> 39.96 <br> 51.35 <br> 59.06 | 20.75 <br> **45.29** <br> 53.53 <br> **59.33** | 21.03 <br> 45.29 <br> 53.54 <br> 59.64 |
| Multi-Doc QA | HotpotQA <br> 2WikiQA <br> MuSiQue <br> Dureader | 50.13 <br> 39.54 <br> 24.68 <br> 33.54 | 48.49 <br> 32.71 <br> 23.22 <br> 33.00 | 44.03 <br> 30.58 <br> 17.85 <br> 33.01 | 48.00 <br> 36.22 <br> **22.87** <br> **33.94** | 46.78 <br> 35.33 <br> 16.97 <br> 33.62 | **54.11** <br> **37.86** <br> 21.74 <br> 33.39 | 54.07 <br> 37.86 <br> 21.24 <br> 33.29 |
| Summary | GovReport <br> QMSum <br> MultiNews <br> VCSUM | 32.17 <br> 24.35 <br> 26.70 <br> 16.37 | 31.93 <br> 22.45 <br> 26.46 <br> 16.55 | 21.40 <br> 20.96 <br> 22.90 <br> 17.81 | **32.21** <br> **25.05** <br> **26.50** <br> 16.17 | 28.72 <br> 23.81 <br> 25.02 <br> **19.12** | 30.33 <br> 24.58 <br> 25.71 <br> 16.17 | 30.38 <br> 24.35 <br> 25.75 <br> 16.20 |
| Few-shot Learning | TREC <br> TriviaQA <br> SAMSum <br> LSHT | 45.00 <br> 84.35 <br> 40.26 <br> 37.75 | 65.50 <br> 85.67 <br> 42.92 <br> 38.00 | **61.00** <br> 75.78 <br> 37.46 <br> 24.57 | 43.50 <br> 81.93 <br> 39.81 <br> **35.75** | 23.50 <br> 83.95 <br> 38.47 <br> 25.50 | 22.50 <br> **84.22** <br> **40.69** <br> 22.01 | 24.00 <br> 84.22 <br> 40.51 <br> 21.47 |
| Synthetic Task | PsgCount <br> PsgRe-en <br> PsgRe-zh | 4.00 <br> 86.50 <br> 90.50 | 4.06 <br> 20.75 <br> 42.00 | 3.00 <br> 19.00 <br> 43.00 | 3.50 <br> 85.00 <br> 90.50 | 3.50 <br> 66.00 <br> 68.00 | **5.00** <br> **92.00** <br> **90.50** | 4.50 <br> 91.00 <br> 90.50 |
| Code | LCC <br> RepoBen-P | 35.72 <br> 35.00 | 58.65 <br> 43.93 | 31.35 <br> 30.72 | 35.91 <br> 34.17 | 33.83 <br> 34.95 | **44.73** <br> **44.62** | 44.73 <br> 44.76 |
| Overall $\uparrow$ | | 42.30 | 37.86 | 32.30 | 41.55 | 37.10 | **42.54** | 42.49 |




Here is the Markdown version of the provided paper:

***

# MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling

**MiniCPM Team**

🤗 [https://huggingface.co/openbmb/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA)
🐙 [https://github.com/OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)

> **Abstract**
>
> The evolution of large language models (LLMs) towards applications with ultra-long contexts faces challenges posed by the high computational and memory costs of the Transformer architecture. While existing sparse and linear attention mechanisms attempt to mitigate these issues, they typically involve a trade-off between memory efficiency and model performance. This paper introduces MiniCPM-SALA[^a], a 9B-parameter hybrid architecture that integrates the high-fidelity long-context modeling of sparse attention (InfLLM-V2) with the global efficiency of linear attention (Lightning Attention). By employing a layer selection algorithm to integrate these mechanisms in a 1:3 ratio and utilizing a hybrid positional encoding (HyPE), the model maintains efficiency and performance for long-context tasks. Furthermore, we introduce a cost-effective continual training framework that transforms pre-trained Transformer-based models into hybrid models, which reduces training costs by approximately 75% compared to training from scratch. Extensive experiments show that MiniCPM-SALA maintains general capabilities comparable to full-attention models while offering improved efficiency. On a single NVIDIA A6000D GPU, the model achieves up to $3.5\times$ the inference speed of the full-attention model at the sequence length of 256K tokens and supports context lengths of up to 1M tokens, a scale where traditional full-attention 8B models fail because of memory constraints.

[^a]: SALA stands for Sparse Attention and Linear Attention.

## 1 Introduction

As large language models (LLMs) (OpenAI et al., 2024; Comanici et al., 2025; Grattafiori et al., 2024; Yang et al., 2025a; DeepSeek-AI et al., 2025) become increasingly effective, the application scenarios of LLMs are undergoing a profound paradigm shift, transitioning from simple question-answering (Brown et al., 2020) to more advanced applications, such as deep understanding and generation of ultra-long contexts (Bai et al., 2024, 2025; Zhou et al., 2025; Shao et al., 2024), repository-scale code engineering (Guo et al., 2024; Jimenez et al., 2024; Liu et al., 2024), and long-horizon agents for complex tasks (Qian et al., 2024; Mialon et al., 2023; Li et al., 2026). For these advanced applications, models are no longer confined to processing fragmented information. Instead, they must demonstrate the capacity to handle ultra-long contexts, such as grasping entire technical manuals at once, analyzing comprehensive project dependency trees containing tens of thousands of lines of code, and maintaining coherent task states and memory over multi-day human-AI collaborations. This pursuit of holistic contextual information makes the ability to process millions of tokens a critical aspect for advanced LLMs (Kimi Team et al., 2025; NVIDIA et al., 2025b).

However, the Transformer architecture (Vaswani et al., 2017), which is the foundation of modern LLMs, encounters severe computational bottlenecks when handling ultra-long contexts due to its core full-attention mechanism. This bottleneck manifests primarily in two dimensions: (1) the *compute bottleneck* of computational complexity: for the standard attention mechanism, the computational cost grows quadratically with the sequence length $N$, i.e., its complexity is $\mathcal{O}(N^2)$. When the context scales to the level of millions of tokens, the huge overhead causes the inference latency to increase dramatically; (2) the *memory bottleneck* of KV-Cache: during the auto-regressive generation process, the model must store the key and value states (KVs) of all historical contextual tokens to avoid redundant computation. For a typical 8B-parameter model, even when utilizing Grouped Query Attention (GQA) (Ainslie et al., 2023), the KV-Cache required for millions of tokens can reach dozens or even hundreds of gigabytes.

To address the aforementioned challenges, existing solutions have developed two primary paradigms: Sparse Attention (Yuan et al., 2025; DeepSeek-AI et al., 2025; Xiao et al., 2024; Zhao et al., 2025) and Linear Attention (Yang et al., 2024a; Gu & Dao, 2024; Peng et al., 2023; Yang et al., 2024b, 2025b). Both paradigms present distinct advantages and inherent limitations. Sparse attention methods attempt to break the compute bottleneck by computing only the most salient portions of the attention matrix, such as adopting sliding windows or global anchors. However, these methods are hindered by a “sparse computation, dense storage” limitation. While local computation reduces immediate processing overhead, the model must still retain the full KV-Cache to support contextual information retrieval. Linear attention utilizes recurrent formulations to successfully reduce computational complexity to $\mathcal{O}(N)$. Nevertheless, this extreme efficiency is achieved by the lossy compression of contextual information and inevitably results in performance degradation.

MiniCPM-SALA employs a hybrid architecture of sparse and linear attention (Chen et al., 2026), specifically designed to achieve efficient ultra-long sequence modeling. This architecture combines the high-fidelity long-context modeling capabilities of InfLLM-V2 (Zhao et al., 2025) and the global computational efficiency of Lightning Attention (Qin et al., 2024). Through this integrated approach, the model significantly mitigates inference overhead and memory consumption, while simultaneously addressing the precision bottleneck typical of pure linear architectures in long-range information processing. Consequently, MiniCPM-SALA provides a balanced solution that maintains both efficiency and high performance for long-context tasks. Furthermore, we employ the continual training paradigm to transform a pre-trained Transformer model into our hybrid model. By eschewing training from scratch, this approach significantly reduces the computational costs of model development. While several works have begun exploring the integration of sparse and linear attention (Hu et al., 2025; Hou et al., 2025; He & Garner, 2025), to the best of our knowledge, MiniCPM-SALA is the first to demonstrate through large-scale experimentation that these hybrids can match the performance of full-attention baselines. Furthermore, the model exhibits high efficiency and strong performance in long-context processing.

In summary, the main contributions of this study can be outlined as follows:
*   We introduce a Sparse-Linear hybrid attention mechanism integrating 25% InfLLM-V2 and 75% Lightning Attention to strike a balance between throughput and precision. By leveraging the granular focus of sparse attention for local details and the $\mathcal{O}(N)$ efficiency of linear attention for broad context, the architecture maintains high semantic accuracy as the sequence length scales up.
*   We demonstrate that the Transformer-to-hybrid paradigm is a highly effective strategy for building strong hybrid models. This approach circumvents the inefficiencies of cold-start training by performing an architectural transformation on the pre-trained weights, thereby reducing the total training budget to approximately 25% relative to training a comparable model from scratch.
*   We adopt HyPE (Hybrid Positional Encoding) (Chen et al., 2026) to effectively harmonize the performance across both short and long contexts. While maintaining general capabilities (e.g., knowledge, mathematics, and coding) comparable to modern full-attention models like Qwen3-8B, MiniCPM-SALA has substantial advantages across multiple long-context benchmarks.
*   MiniCPM-SALA demonstrates substantial resource savings and speed advantages in long-context scenarios. On the NVIDIA A6000D GPU, MiniCPM-SALA achieves up to $3.5\times$ the inference speed of Qwen3-8B at a sequence length of 256K tokens. Furthermore, MiniCPM-SALA supports inference at context lengths of up to 1M tokens on both NVIDIA A6000D and 5090 GPUs, whereas Qwen3-8B fails at this length due to out-of-memory (OOM) errors. These results demonstrate the broad prospects of MiniCPM-SALA in edge-side information-intensive applications.

## 2 Model Development

In this section, we introduce the model architecture and training strategies for MiniCPM-SALA. Specifically, we combine the efficient sparse attention for long-context modeling and linear attention for global efficiency in MiniCPM-SALA. Moreover, we also introduce an efficient training method, which can transform a standard Transformer model into sparse-linear hybrid attention.

**Figure 1:** Architecture of MiniCPM-SALA. The model adopts an efficient hybrid design that combines InfLLM-V2 (Zhao et al., 2025) and Lightning Attention (Qin et al., 2024) modules in a 1:3 ratio. Building on an intermediate MiniCPM-4.0 (MiniCPM-Team et al., 2025) checkpoint, MiniCPM-SALA undergoes a continual training phase to convert a standard Transformer model into a sparse-linear hybrid model.

### 2.1 Model Architecture

The overall architecture of MiniCPM-SALA is illustrated in Figure 1. MiniCPM-SALA adopts a hybrid architecture that interleaves sparse attention layers and linear attention layers. We retain the Feed-Forward Network (FFN) block after each attention block in the Transformer architecture to ensure high-capacity knowledge representation. Inspired by the architectural designs of recent representative studies, such as Qwen3-Next (Qwen Team, 2025) and Kimi-Linear (Kimi Team et al., 2025), as well as our internal small-scale preliminary experiments, we employ a 1:3 mixing ratio: 25% of the layers adopt sparse attention while the remaining 75% employ linear attention.

This hybrid configuration leverages the complementary strengths of both attention mechanisms. Linear attention layers have constant computational and memory complexities with respect to sequence length, facilitating efficient processing of long contexts. On the other hand, sparse attention layers facilitate effective modeling of long-range dependencies. Rather than naively uniformly interleaving the two attention variants, we determine the placement of sparse attention modules using the layer selection mechanism proposed by Chen et al. (2026), which results in superior downstream performance.

**Training Strategy** Existing paradigms for training hybrid models generally fall into two categories: (1) training from scratch (Zuo et al., 2025; Qwen Team, 2025; Kimi Team et al., 2025; NVIDIA et al., 2025b) and (2) converting a pre-trained Transformer model into a hybrid model via cross-architecture distillation (Wang et al., 2024a; Hoshino et al., 2025; Li et al., 2025; Gu et al., 2025). Although training from scratch offers simplicity and maximum architectural flexibility, continual-training conversion is a more resource-efficient alternative that leverages parameter inheritance from established pre-trained models. By recycling pre-trained weights and representations, the continual-training method significantly reduces the immense computational cost typically associated with *de novo* training, achieving competitive performance with a fraction of the budget. Accordingly, MiniCPM-SALA leverages a conversion-based framework that uses continual training to adapt a Transformer into an efficient hybrid version while preserving its core capabilities.

**Table 1:** Overview of the whole training process to build MiniCPM-SALA.

| Stage | Trainable Parameters | Sparse Attention | Sequence Length | # Tokens |
| :--- | :--- | :---: | :---: | :--- |
| Architecture Conversion (HALO) | Linear Attention | Disabled | 0.5K | 1.3B |
| Continual Stable-Training | All Parameters | Disabled | 4K | 314.6B |
| Short-Decay Training | All Parameters | Disabled | 4K | 1006.6B |
| Long-Decay Training | All Parameters | Enabled | 32K<br>160K<br>520K | 102.2B<br>62.9B<br>50.6B |
| Supervised Fine-Tuning | All Parameters | Enabled | 64K<br>140K | 204.5B<br>213.3B |

**Sparse Attention and Linear Attention** For the sparse attention layers, we incorporate InfLLM-V2 (Zhao et al., 2025), which offers the distinct advantage of introducing no additional parameters to the architecture. Its inherent flexibility and ability to switch seamlessly between dense and sparse modes are highly compatible with our conversion process. This compatibility facilitates a stable training initialization by allowing sparse modules to inherit dense weights without architectural discrepancies, ensuring that the conversion to a hybrid structure does not compromise the model capacity. For the linear attention layers, we utilize Lightning Attention (Qin et al., 2024). Given our Transformer-to-hybrid conversion paradigm, Lightning Attention is selected for its functional proximity to the standard softmax attention. This structural alignment is intended to mitigate the complexities of parameter adaptation, thereby preserving pre-trained knowledge and ensuring robust downstream performance. Lightning Attention also provides better length generalization capabilities according to Chen et al. (2026), which may improve data efficiency during long-context continual-training.

**Other Architectural Improvements** Following HypeNet (Chen et al., 2026), we also introduce several architectural modifications to enhance the expressivity and training stability of MiniCPM-SALA. These include QK-Normalization (Henry et al., 2020), HyPE (Chen et al., 2026), and the integration of output gates.

*   **QK-Normalization:** This is applied to all attention layers (both sparse and linear layers) to prevent the activation spikes that often occur in long-context training and further improve and boost the expressivity of linear attention modules.
*   **HyPE (Hybrid Positional Encoding):** To balance rich positional awareness and long-range information retention, we employ a hybrid approach to positional encoding. We apply Rotary Positional Embedding (RoPE) (Su et al., 2023) to the linear attention layers to facilitate position-sensitive memory, allowing the model to preserve the relative order of tokens within the global context. On the other hand, we remove RoPE in the sparse attention layers. This strategic omission prevents the decay of long-distance information often associated with RoPE, thereby enabling more precise recall over extended contexts.
*   **Output gates:** Furthermore, we incorporate an output gate after each attention block (both sparse and linear). This architectural choice aligns with recent advances in the gated attention mechanism (Qiu et al., 2025), in which the output gate has been shown to effectively mitigate issues such as attention sink. By regulating the information flow, the output gate prevents excessive focus on specific tokens and ensures a more flexible distribution of attention weights. Empirically, we observe that integrating output gates into both linear and sparse attention significantly improves model stability and performance.

### 2.2 Model Training

The training of MiniCPM-SALA is conducted through a multi-stage process that starts from an intermediate checkpoint of MiniCPM-4.0 (MiniCPM-Team et al., 2025), which has already been trained on 7T tokens. This methodology represents an extended implementation of Hybrid Attention via Layer Optimization (HALO) (Chen et al., 2026). In the initial phase, we use the HALO framework to convert softmax attention to linear attention. This conversion serves as the starting point for subsequent pipeline stages, including continual pre-training and post-training. By leveraging this approach, the model can transition from a dense architecture to a hybrid structure while preserving the general capabilities acquired during the backbone’s earlier training phases. The entire conversion process, consisting of five stages, is shown in Table 1. It is worth noting that the Transformer-to-hybrid training of MiniCPM-SALA consumes approximately 2T tokens. This corresponds to roughly 25% of the data volume required to train MiniCPM-4.0 from scratch (8T tokens).

**Architecture Conversion (HALO)** The first stage uses HALO to convert the Transformer model from a full attention architecture to a hybrid architecture. During this phase, the training configuration of MiniCPM-SALA differs from the standard HALO approach in two aspects. First, regarding layer selection, we keep the first and last layers unconverted to improve training stability. For the remaining layers, we utilize the HALO selection algorithm to determine which layers are preserved as softmax attention layers. These preserved softmax attention layers are subsequently trained as sparse attention in later stages. The second difference from standard HALO is that we do not perform the final fine-tuning step of the original HALO process. Instead, we conduct more extensive continual pre-training and post-training, which comprise the subsequent stages of our methodology. The training process at this stage is highly efficient, using only 1.3B tokens with a sequence length of 512 tokens. Furthermore, only the converted linear-attention layers are trainable during this stage, while all other parameters remain frozen.

**Continual Stable-Training** The second stage is continual stable-training. We use the checkpoint from the previous stage as the starting point for further training on the MiniCPM-4.0 pre-training dataset. The primary objective of this phase is to facilitate better coordination between the converted linear attention layers and other model components, including full attention layers, FFN layers, and embeddings. The sequence length for this process is set to 4K tokens, with a total training volume of 314.6B tokens. Since the sequence length remains relatively short, the sparse attention is disabled at this stage to maintain computational efficiency. For the hyperparameter configuration, the learning rate (LR) is set to $7.5 \times 10^{-3}$ and held constant after a 2,000-step LR warmup period. Accounting for the sequence length and the number of GPUs, the global batch size is set to 7.8M tokens.

**Short-Decay Training** The third stage is short-decay training, during which the LR undergoes exponential decay from $7.5 \times 10^{-3}$ to $3.75 \times 10^{-4}$. This process utilizes a sequence length of 4K tokens and a global batch size of 7.8M tokens. This stage involves training on 1T tokens, representing the most extensive data volume in the entire development pipeline. Building on the MiniCPM-4.0 decay strategy, we significantly increase the weight of L2 high-quality selection data (Wang et al., 2026) and introduce a large volume of PDF corpora and L3 synthetic data. This approach aims to enhance general capabilities and logical reasoning using high-information-density training data, achieving the efficient compression and internalization of massive amounts of knowledge.

**Long-Decay Training** The fourth stage, long-decay, progressively extends the context length from 4K to 32K, 160K, and finally 520K tokens. These processes use data volumes of 102.2B tokens, 62.9B tokens, and 50.6B tokens, respectively. To accommodate the increased sequence lengths, the global batch size is adjusted to 7.8M, 9.8M, and 10.1M tokens, while the LR is systematically decays from $3 \times 10^{-4}$ to $2 \times 10^{-4}$ at 32K, then to $1 \times 10^{-4}$ at 160K, and finally to $3.75 \times 10^{-5}$ at 520K to conclude the process. At this stage, we up-sample the proportion of long-context data to better align the model with long-sequence distributions. Given the growing computational advantages of sparse attention at longer sequences, we enable the sparse attention mechanism at this stage and maintain full-parameter training, thereby allowing the model to effectively learn the synergy between sparse attention and linear attention.

**Supervised Fine-Tuning** The SFT corpus for this stage is composed of high-quality reasoning-intensive data, encompassing code, mathematics, knowledge, function calls, and general dialogue. This selection is designed to fully catalyze the reasoning and task-execution capabilities under complex logic. Furthermore, we specifically synthesize long-context data to enhance the precision of information retrieval and cross-document comprehension within extended sequences. During the SFT stage, the context length is set to 64K and increased to 140K afterwards, utilizing 204.5B and 213.3B tokens, respectively. Sparse attention remains enabled throughout this entire process. By bridging shorter and longer contexts, this strategy allows the model to better balance general capabilities with long-context proficiency. For both phases, the LR follows a schedule with a 1,000-step warmup to a peak of $1 \times 10^{-3}$ before decaying to $1 \times 10^{-4}$, while the global batch sizes are set to 15.7M for the 64K phase and 17.8M for the 140K phase.

## 3 Experiments

### 3.1 Model Performance

**Benchmarks** To thoroughly assess the general capabilities of the model, we conducted evaluations across a diverse array of benchmarks. These include knowledge-intensive tasks (CMMLU (Li et al., 2023), MMLU-Pro (Wang et al., 2024b)), coding benchmarks (HumanEval (Chen et al., 2021), LCB-v5/v6 (Jain et al., 2025), MBPP (Austin et al., 2021)), and mathematical reasoning sets (AIME24/25 (AIME, 2025)), alongside other representative benchmarks such as BBH (Suzgun et al., 2022) and IFEval (Zhou et al., 2023). We further evaluated long-context capabilities using RULER (Hsieh et al., 2024), MRCR[^1], and NoLiMa (Modarressi et al., 2025). We utilized the OpenCompass framework (Contributors, 2023) to conduct the evaluations.

**Table 2:** Standard evaluation results of MiniCPM-SALA and other open-source LLMs.

| Models | Qwen3 | Nemotron-Nano-v2 | MiniCPM-4.1 | Ministral-3-R | Falcon-H1R | MiniCPM-SALA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **# Param.** | 8B | 9B | 8B | 8B | 7B | 9B |
| **Knowledge** | | | | | | |
| CMMLU | 81.68 | 61.59 | 84.72 | 71.74 | 63.55 | 81.55 |
| MMLU-Pro | 73.26 | 71.79 | 72.70 | 68.75 | 70.98 | 67.04 |
| **Code** | | | | | | |
| HumanEval | 93.90 | 93.90 | 91.46 | 96.95 | 96.34 | 95.12 |
| LCB-v5 | 56.89 | 68.26 | 56.89 | 65.87 | 67.66 | 60.48 |
| LCB-v6 | 48.57 | 60.00 | 51.43 | 53.71 | 57.71 | 52.00 |
| MBPP | 81.32 | 93.39 | 91.05 | 94.16 | 91.05 | 89.11 |
| **Math** | | | | | | |
| AIME24 | 73.33 | 71.67 | 80.83 | 81.46 | 86.67 | 83.75 |
| AIME25 | 66.67 | 56.67 | 72.08 | 75.00 | 81.04 | 78.33 |
| **Other** | | | | | | |
| BBH | 74.17 | 74.28 | 82.68 | 64.39 | 63.17 | 81.55 |
| IFEval | 84.66 | 86.69 | 77.45 | 70.06 | 86.32 | 76.34 |
| Average | 73.45 | 73.82 | 76.13 | 74.21 | 76.45 | **76.53** |

[^1]: [https://huggingface.co/datasets/openai/mrcr](https://huggingface.co/datasets/openai/mrcr)

**Table 3:** Long-context evaluation results of MiniCPM-SALA and other open-source LLMs.

| Models | | Qwen3 | Nemotron-Nano-v2 | Ministral-3-R | Falcon-H1R | MiniCPM-SALA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **# Param.** | | 8B | 9B | 8B | 7B | 9B |
| **RULER** | 64K | 80.53 | 88.77 | 70.66 | 56.50 | 92.65 |
| | 128K | 71.74 | 68.01 | 45.09 | 36.33 | 89.37 |
| **MRCR** | 64K-2N | 29.20 | 20.91 | 44.02 | 13.18 | 29.77 |
| | 64K-4N | 21.56 | 13.69 | 35.80 | 9.06 | 20.57 |
| | 64K-8N | 17.82 | 13.24 | 17.23 | 6.93 | 16.56 |
| | 128K-2N | 26.50 | 14.61 | 50.30 | 9.17 | 28.62 |
| | 128K-4N | 14.75 | 12.20 | 22.66 | 8.22 | 19.62 |
| | 128K-8N | 12.15 | 7.55 | 14.47 | 7.54 | 10.12 |
| **NoLiMa** | 32K | 43.40 | 19.69 | 3.78 | 14.89 | 54.54 |
| | 64K | 23.35 | 11.82 | 2.48 | 9.87 | 42.95 |
| | 128K | 11.25 | 5.80 | 3.48 | 4.73 | 23.86 |
| Average | | 32.02 | 25.12 | 28.18 | 16.04 | **38.97** |

**Baseline Models** Given that MiniCPM-SALA is a 9B-parameter model, we selected a series of modern baselines of comparable size, encompassing both hybrid and full-attention architectures. Specifically, the baselines include Qwen3-8B (Yang et al., 2025a), Nemotron-Nano-v2-9B (NVIDIA et al., 2025a), MiniCPM-4.1-8B (MiniCPM-Team et al., 2025), Ministral-3-Reasoning-8B (Liu et al., 2026), and Falcon-H1R-7B (Team et al., 2026). We exclude MiniCPM-4.1-8B from the evaluation of long contexts because of its limitation to a context length of 64K.

**Results of Standard Evaluation** Table 2 presents the performance of MiniCPM-SALA across a variety of standard benchmarks. The model achieves an average score of 76.53, which represents a competitive level among open-source models of a similar scale. In coding tasks, the model demonstrates high proficiency with scores of 95.12 on HumanEval and 89.11 on MBPP. Mathematical reasoning capabilities also remain robust, as evidenced by the scores of 83.75 on AIME24 and 78.33 on AIME25. These results indicate that the integration of long-context mechanisms does not result in a significant degradation of general capabilities or short-context performance. The model maintains a performance profile that is comparable to, and in some cases exceeds, the performance of models such as Qwen3-8B and Falcon-H1R-7B in standard evaluation settings.

**Results of Long-Context Evaluation** The evaluation of long-context capabilities is summarized in Table 3, covering benchmarks such as RULER, MRCR, and NoLiMa. MiniCPM-SALA shows a notable proficiency in managing extended input sequences. On the RULER benchmark at a 128K context length, the model maintains a score of 89.37, while many other baselines exhibit a more pronounced decrease in accuracy at the same scale. The advantage of the model is particularly visible in the NoLiMa benchmark, where it achieves a score of 23.86 at the 128K level. This performance is substantially higher than the scores recorded for other models in the comparison. With an overall average long-context score of 38.97, the model demonstrates improved stability and effective information retrieval across large context windows.

**Table 4:** Ultra-long context evaluation results of MiniCPM-SALA and other open-source LLMs. $\ast$ denotes results cited from the official Qwen3-Next documentation.

| RULER | 128K | 512K | 1000K | 2048K |
| :--- | :---: | :---: | :---: | :---: |
| Qwen3-30B-A3B-Instruct-2507$^{\ast}$ | 89.1 | 78.4 | 72.8 | - |
| Qwen3-235B-A22B-Instruct-2507$^{\ast}$ | 93.9 | 90.9 | 84.5 | - |
| Qwen3-Next-80B-A3B-Instruct$^{\ast}$ | 96.0 | 86.9 | 80.3 | - |
| MiniCPM-SALA (9B) | 89.4 | 87.1 | 86.3 | 81.6 |

**Figure 2:** Inference speed comparison between Qwen3-8B and MiniCPM-SALA. For each tested sequence length, the models process a specified input (prefilling) and generate 1K tokens (decoding). “TTFT” denotes Time To First Token, representing the prefilling latency, while “End-to-end” measures the total latency including both prefilling and decoding phases.
*(Note: Subfigures (a) TTFT (s) on A6000D non-quantized, (b) End-to-end (s) latency on A6000D non-quantized, (c) TTFT (s) on A6000D quantized, (d) End-to-end (s) latency on A6000D quantized are referenced here).*

**Figure 3:** Inference speed comparison between Qwen3-8B and MiniCPM-SALA. For each tested sequence length, the models process a specified input (prefilling) and generate 1K tokens (decoding).
*(Note: Subfigures (a) TTFT (s) on 5090 non-quantized, (b) End-to-end (s) latency on 5090 non-quantized, (c) TTFT (s) on 5090 quantized, (d) End-to-end (s) latency on 5090 quantized are referenced here).*

**Results of Ultra-Long Context** As demonstrated in Table 4, MiniCPM-SALA exhibits surprising length extrapolation capabilities. The results for the Qwen3 models are sourced from the official Qwen3-Next documentation[^2]. Despite being restricted to a 520K training length, the model successfully extrapolates to 2048K tokens without a significant degradation in performance, maintaining a score of 81.6. It is worth noting that this extrapolation requires no auxiliary techniques (e.g., YaRN (Peng et al., 2024)). This result highlights the efficacy of our approach in handling context windows far beyond the training stage. Additionally, MiniCPM-SALA shows remarkable parameter efficiency, surpassing the performance of the Qwen3-Next-80B-A3B-Instruct model at the 1000K context length (86.3 vs. 80.3), proving that effective long-context processing does not necessarily require massive parameter counts. The length extrapolation capabilities of MiniCPM-SALA can be attributed to the NoPE configuration within the sparse attention layers. In this design, the stored KV-Cache does not require combination with positional information, which can otherwise hinder the capture of long-range dependencies.

[^2]: [https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

### 3.2 Inference Speed

We assessed the inference speed of MiniCPM-SALA and Qwen3-8B across different hardware and sequence lengths. To verify the long-text processing capabilities of the model in edge computing scenarios, we conducted experiments not only on cloud-grade inference chips, such as the NVIDIA A6000D, but also on consumer-grade edge GPUs, such as the NVIDIA 5090. For each sequence length, we measured both the Time To First Token (TTFT) and the end-to-end latency. The former serves as an indicator of the prefilling speed, while the latter reflects the combined performance of the prefilling and decoding phases. To align the evaluation with practical deployment scenarios, we assessed the inference latency for both non-quantized models and models compressed via GPTQ (Frantar et al., 2023) INT4 quantization.

Figure 2 presents a comprehensive comparison of inference latency between Qwen3-8B and MiniCPM-SALA on an NVIDIA A6000D GPU (96GB VRAM). We evaluated performance across sequence lengths ranging from 64K to 1024K tokens. As illustrated, MiniCPM-SALA demonstrates a significant performance advantage over the baseline across all tested configurations. In non-quantized settings, MiniCPM-SALA consistently achieves lower latency. Notably, at a sequence length of 256K, MiniCPM-SALA reduces the TTFT from 180.8s (Qwen3) to just 51.6s.

Crucially, the results highlight a distinct advantage in memory efficiency. While Qwen3-8B encounters OOM failures at sequence lengths of 512K and 1024K, MiniCPM-SALA successfully processes these extended contexts. For example, at 1024K tokens, MiniCPM-SALA maintains a TTFT of 250.3s (non-quantized) and 256.9s (quantized), whereas the baseline fails to complete the inference. This trend persists in the end-to-end latency metrics, proving that MiniCPM-SALA is robust enough for ultra-long context generation tasks where full-attention models fail.

Figure 3 demonstrates the critical advantage of MiniCPM-SALA on memory-constrained hardware. On the RTX 5090 (32GB VRAM), the baseline Qwen3-8B hits a “memory wall” significantly earlier than on the A6000D, triggering OOM errors at just 128K tokens in non-quantized settings and 256K in quantized settings. In stark contrast, MiniCPM-SALA successfully scales to 1024K context lengths without memory failure. This suggests that MiniCPM-SALA effectively democratizes long-context inference, enabling 1M-token processing on consumer-level GPUs where full-attention architectures are unusable.

## 4 Conclusion

In this paper, we presented MiniCPM-SALA, a hybrid architecture that combines sparse and linear attention to overcome the computational and memory bottlenecks of ultra-long context modeling. By utilizing a cost-effective Transformer-to-hybrid training paradigm, we successfully retained the general capabilities of full-attention models while reducing training costs by approximately 75%. Experimental results confirm that MiniCPM-SALA achieves a substantial inference speedup and enables 1M-token context processing on single GPUs (e.g., NVIDIA A6000D), surpassing the limitations of standard 8B models. These results establish MiniCPM-SALA as a scalable and accessible solution for next-generation, information-intensive applications.

## 5 Contributions and Acknowledgments

MiniCPM-SALA is the result of the collective efforts of all members of our team. Please refer to Chen et al. (2026) and Zhao et al. (2025) for model architecture details.

**Contributors (Ordered by the last name)** Wenhao An, Yingfa Chen, Yewei Fang, Jiayi Li, Xin Li, Yaohui Li, Yishan Li, Yuxuan Li, Biyuan Lin, Chuan Liu$^\star$, Hezi Liu, Siyuan Liu, Hongya Lyu, Yinxu Pan, Shixin Ren, Xingyu Shen, Zhou Su, Haojun Sun, Yangang Sun, Zhen Leng Thai, Xin Tian, Rui Wang$^\star$, Xiaorong Wang, Yudong Wang, Bo Wu, Xiaoyue Xu, Dong Xu, Shuaikang Xue, Jiawei Yang, Bowen Zhang, Jinqian Zhang, Letian Zhang, Shengnan Zhang, Xinyu Zhang, Xinyuan Zhang$^\star$, Zhu Zhang, Hengyu Zhao, Jiacheng Zhao$^\star$, Jie Zhou, Zihan Zhou

**Project Design and Coordination** Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu, Maosong Sun

**Affiliations** Contributors marked with $^\star$ are affiliated with XCORE SIGMA, while the remaining contributors are affiliated with OpenBMB.

## References

AIME. AIME problems and solutions, 2025. URL `https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions`.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. URL `https://aclanthology.org/2023.emnlp-main.298/`.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, 2024. URL `https://aclanthology.org/2024.findings-emnlp.74/`.

Yushi Bai, Jiajie Zhang, Xin Lv, Linzhi Zheng, Siqi Zhu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longwriter: Unleashing 10,000+ word generation from long context LLMs. In *The Thirteenth International Conference on Learning Representations*, 2025. URL `https://openreview.net/forum?id=kQ5s9Yh0WI`.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL `https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf`.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Yingfa Chen, Zhen Leng Thai, Zihan Zhou, Zhu Zhang, Xingyu Shen, Shuo Wang, Chaojun Xiao, Xu Han, and Zhiyuan Liu. Hybrid linear attention done right: Efficient distillation and effective architectures for extremely long contexts, 2026. URL `https://arxiv.org/abs/2601.22156`.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL `https://arxiv.org/abs/2507.06261`.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. `https://github.com/open-compass/opencompass`, 2023.

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3.2: Pushing the frontier of open large language models, 2025. URL `https://arxiv.org/abs/2512.02556`.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers, 2023. URL `https://arxiv.org/abs/2210.17323`.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models, 2024. URL `https://arxiv.org/abs/2407.21783`.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In *First Conference on Language Modeling*, 2024. URL `https://openreview.net/forum?id=tEYskw1VY2`.

Yuxian Gu, Qinghao Hu, Shang Yang, Haocheng Xi, Junyu Chen, Song Han, and Han Cai. Jet-nemotron: Efficient language model with post neural architecture search, 2025. URL `https://arxiv.org/abs/2508.15884`.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming – the rise of code intelligence, 2024. URL `https://arxiv.org/abs/2401.14196`.

Mutian He and Philip N. Garner. Alleviating forgetfulness of linear attention by hybrid sparse attention and contextualized learnable token eviction, 2025. URL `https://arxiv.org/abs/2510.20787`.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 4246–4253, 2020.

Yuichiro Hoshino, Hideyuki Tachibana, Muneyoshi Inahara, and Hiroto Takegawa. Rad: Redundancy-aware distillation for hybrid models via self-speculative decoding, 2025. URL `https://arxiv.org/abs/2505.22135`.

Haowen Hou, Zhiyi Huang, Kaifeng Tan, Rongchang Lu, and Fei Richard Yu. Rwkv-x: A linear complexity hybrid language model, 2025. URL `https://arxiv.org/abs/2504.21463`.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In *First Conference on Language Modeling*, 2024. URL `https://openreview.net/forum?id=kIoBbc76Sy`.

Xiang Hu, Jiaqi Leng, Jun Zhao, Kewei Tu, and Wei Wu. Hardware-aligned hierarchical sparse attention for efficient long-term memory access, 2025. URL `https://arxiv.org/abs/2504.16795`.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In *The Thirteenth International Conference on Learning Representations*, 2025. URL `https://openreview.net/forum?id=chfJJYC3iL`.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In *The Twelfth International Conference on Learning Representations*, 2024. URL `https://openreview.net/forum?id=VTF8yNQM66`.

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, et al. Kimi linear: An expressive, efficient attention architecture, 2025. URL `https://arxiv.org/abs/2510.26692`.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese, 2023.

Keyu Li, Junhao Shi, Yang Xiao, Mohan Jiang, Jie Sun, Yunze Wu, Shijie Xia, Xiaojie Cai, Tianze Xu, Weiye Si, Wenjie Li, Dequan Wang, and Pengfei Liu. Agencybench: Benchmarking the frontiers of autonomous agents in 1m-token real-world contexts, 2026. URL `https://arxiv.org/abs/2601.11044`.

Yanhong Li, Songlin Yang, Shawn Tan, Mayank Mishra, Rameswar Panda, Jiawei Zhou, and Yoon Kim. Distilling to hybrid attention models via kl-guided layer selection, 2025. URL `https://arxiv.org/abs/2512.20569`.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, et al. Ministral 3, 2026. URL `https://arxiv.org/abs/2601.08584`.

Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code autocompletion systems. In *The Twelfth International Conference on Learning Representations*, 2024. URL `https://openreview.net/forum?id=pPjZIOuQuF`.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants, 2023. URL `https://arxiv.org/abs/2311.12983`.

MiniCPM-Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, et al. Minicpm4: Ultra-efficient llms on end devices, 2025. URL `https://arxiv.org/abs/2506.07900`.

Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, and Hinrich Schuetze. Nolima: Long-context evaluation beyond literal matching. In *Forty-second International Conference on Machine Learning*, 2025. URL `https://openreview.net/forum?id=0OshX1hiSa`.

NVIDIA, Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, et al. Nvidia nemotron nano 2: An accurate and efficient hybrid mamba-transformer reasoning model, 2025a. URL `https://arxiv.org/abs/2508.14444`.

NVIDIA, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, et al. Nemotron 3 nano: Open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning, 2025b. URL `https://arxiv.org/abs/2512.20848`.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report, 2024. URL `https://arxiv.org/abs/2303.08774`.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Gv, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan Wind, Stanisław Woźniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. RWKV: Reinventing RNNs for the transformer era. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, 2023. URL `https://aclanthology.org/2023.findings-emnlp.936/`.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. In *The Twelfth International Conference on Learning Representations*, 2024. URL `https://openreview.net/forum?id=wHBfxhZu1u`.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. ChatDev: Communicative agents for software development. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2024. URL `https://aclanthology.org/2024.acl-long.810/`.

Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Various lengths, constant speed: Efficient language modeling with lightning attention. In *Forty-first International Conference on Machine Learning*, 2024. URL `https://openreview.net/forum?id=Lwm6TiUP4X`.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025. URL `https://openreview.net/forum?id=1b7whO4SfY`.

Qwen Team. Qwen3-Next: Towards Ultimate Training & Inference Efficiency, 2025. URL `https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd`.

Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Monica Lam. Assisting in writing Wikipedia-like articles from scratch with large language models. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, 2024. URL `https://aclanthology.org/2024.naacl-long.347/`.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. URL `https://arxiv.org/abs/2104.09864`.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. *arXiv preprint arXiv:2210.09261*, 2022.

Falcon LLM Team, Iheb Chaabane, Puneesh Khanna, Suhail Mohmad, Slim Frikha, Shi Hu, Abdalgader Abubaker, Reda Alami, Mikhail Lubinets, Mohamed El Amine Seddik, and Hakim Hacid. Falcon-h1r: Pushing the reasoning frontiers with a hybrid model for efficient test-time scaling, 2026. URL `https://arxiv.org/abs/2601.02346`.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL `https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf`.

Junxiong Wang, Daniele Paliotta, Avner May, Alexander M. Rush, and Tri Dao. The mamba in the llama: Distilling and accelerating hybrid models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), *Advances in Neural Information Processing Systems*, volume 37, pp. 62432–62457. Curran Associates, Inc., 2024a. doi: 10.52202/079017-1996. URL `https://proceedings.neurips.cc/paper_files/paper/2024/file/723933067ad315269b620bc0d2c05cba-Paper-Conference.pdf`.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), *Advances in Neural Information Processing Systems*, volume 37, pp. 95266–95290. Curran Associates, Inc., 2024b. doi: 10.52202/079017-3018. URL `https://proceedings.neurips.cc/paper_files/paper/2024/file/ad236edc564f3e3156e1b2feafb99a24-Paper-Datasets_and_Benchmarks_Track.pdf`.

Yudong Wang, Zixuan Fu, Hengyu Zhao, Chen Zhao, Chuyue Zhou, Xinle Lin, Hongya Lyu, Shuaikang Xue, Yi Yi, Yingjiao Wang, Zhi Zheng, Yuzhou Zhang, Jie Zhou, Chaojun Xiao, Xu Han, Zhiyuan Liu, and Maosong Sun. Data science and technology towards agi part i: Tiered data management, 2026. URL `https://arxiv.org/abs/2602.09003`.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), *Advances in Neural Information Processing Systems*, volume 37, pp. 119638–119661. Curran Associates, Inc., 2024. doi: 10.52202/079017-3801. URL `https://proceedings.neurips.cc/paper_files/paper/2024/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf`.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report, 2025a. URL `https://arxiv.org/abs/2505.09388`.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In *Forty-first International Conference on Machine Learning*, 2024a. URL `https://openreview.net/forum?id=ia5XvxFUJT`.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024b. URL `https://openreview.net/forum?id=y8Rm4VNRPH`.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In *The Thirteenth International Conference on Learning Representations*, 2025b. URL `https://openreview.net/forum?id=r8H7xhYPwz`.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2025. URL `https://aclanthology.org/2025.acl-long.1126/`.

Weilin Zhao, Zihan Zhou, Zhou Su, Chaojun Xiao, Yuxuan Li, Yanghao Li, Yudi Zhang, Weilun Zhao, Zhen Li, Yuxiang Huang, Ao Sun, Xu Han, and Zhiyuan Liu. Infllm-v2: Dense-sparse switchable attention for seamless short-to-long adaptation, 2025. URL `https://arxiv.org/abs/2509.24663`.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL `https://arxiv.org/abs/2311.07911`.

Zihan Zhou, Chong Li, Xinyi Chen, Shuo Wang, Yu Chao, Zhili Li, Haoyu Wang, Qi Shi, Zhixing Tan, Xu Han, Xiaodong Shi, Zhiyuan Liu, and Maosong Sun. LLM$\times$MapReduce: Simplified long-sequence processing using large language models. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2025. URL `https://aclanthology.org/2025.acl-long.1341/`.

Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, and Slim Frikha. Falcon-h1: A family of hybrid-head language models redefining efficiency and performance, 2025. URL `https://arxiv.org/abs/2507.22448`.