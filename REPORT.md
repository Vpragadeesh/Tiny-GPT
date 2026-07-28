> Generated 2026-07-20 · depth: standard · workspace: /home/pragadeesh/Tiny-GPT

# Tiny-GPT: Can a 30M-Parameter GPT Produce Coherent English on a 4GB GPU?

## Executive Summary

1. **No sharp coherence threshold exists in scaling laws.** Kaplan et al. (2020) show loss scales as a smooth power law L(N) = (Nc/N)^0.076 across 7+ orders of magnitude with no observed phase transition [1]. Doubling parameters from 10M to 300M yields only ~23% loss reduction [9], confirming improvement is gradual, not stepwise.

2. **~100M+ parameters is the practical floor for readable general-domain English.** TinyStories (2023) explicitly states that 125M-parameter models (GPT-2 small, GPT-Neo small) trained on general corpora "can rarely generate coherent and consistent English text beyond a few words" [6] — though this is a single source from 2023 and may not reflect current best practices with better tokenization and higher-quality data.

3. **A 30M-param model almost certainly cannot produce coherent general English.** On simplified TinyStories data, coherence requires hidden dim ≥128 and ≥2 layers [7]; on general web data, the evidence strongly suggests 30M is far below the viability threshold.

4. **CPU offloading is unnecessary for a 30M-param model on 4GB VRAM.** A 30M-param transformer with AdamW + gradient checkpointing uses ~480MB VRAM total (model: 120MB, grads: 120MB, optimizer: 240MB, activations: ~50-100MB at seq-len 512), leaving ~3.5GB headroom [F7:7]. The brief's Question 2 may rest on a false assumption.

5. **Cross-entropy ~3 nats/token is the "sweet spot" for human-rated text quality.** Mirostat calibration (2020) shows texts at τ=3 received best fluency/coherence ratings, with >50% of human raters mistaking AI text for human-written [F5:1]. Below 2.5 causes repetition ("boredom trap") [F5:2]; above 5 causes incoherence ("confusion trap") [F5:3].

6. **PyTorch SDPA with FlashAttention delivers O(N) memory but NOT O(N) compute.** FLOPs remain O(N²) [F3:1]; the benefit is reduced HBM accesses. At seq-len 1024, Flash is ~38× faster than the Math backend [F3:7]. The most common pitfall: FP32 inputs silently force fallback to the O(N²) Math backend [F3:4].

7. **FineWeb-Edu sample-10BT is the best dataset choice for a tiny GPT.** Its educational-quality filtering (BERT classifier, F1=82%, trained on Llama3-70B annotations) means each token carries more learning signal than unfiltered web crawls like C4 [F4:2][F4:10]. The 10B token sample is a reasonable starting point; scaling laws suggest diminishing returns beyond ~10-50B tokens for sub-300M models [F4:6].

8. **No empirical training-run data exists for 30M-param GPT on web-scale data.** Community benchmarks (nanoGPT, llm.c, Cerebras-GPT) all start at ≥111M parameters [F6:6]. All claims about minimum viable size at 30M are extrapolations from scaling laws or constrained-domain experiments, not direct measurements.

## Background & Scope

This report addresses whether a ~30M parameter dense GPT-style autoregressive transformer, trained from scratch on consumer GPUs (4GB VRAM) using CPU-offloaded AdamW and FlashAttention, can produce coherent English text. The investigation spans five themes: (1) scaling laws and minimum viable model size, (2) VRAM budget and CPU offloading necessity, (3) SDPA/FlashAttention behavior at sequence lengths 256–512, (4) dataset suitability and token requirements, and (5) coherence evaluation via loss-to-readability calibration.

**Scope boundaries**: Decoder-only autoregressive transformers only (no encoder-decoder); 10M–300M parameters with emphasis on ~30M; consumer GPUs with ≤4GB VRAM; English text only; scratch training (no fine-tuning); PyTorch 2.x with CUDA 11.8+.

## 1. Scaling Laws & Minimum Viable Model Size

### 1.1 The Power Law Has No Phase Transition

Kaplan et al. (2020) established that loss scales as a power law with model size, dataset size, and compute, spanning 7+ orders of magnitude with no observed lower bound or discontinuity [1][2]. The fitted relationship L(N) = (Nc/N)^0.076 means the trend is monotonic — there is no "cliff" where models suddenly become coherent [2]. The smallest models trained in the study had ~768 non-embedding parameters [3], far below the 30M target, but even these followed the power law.

Chinchilla (Hoffmann et al. 2022) extended this to compute-optimal training, finding that model size and training tokens should scale equally (~20 tokens per parameter) [4]. For a 30M-param model, this implies ~600M tokens for compute-optimal training — well within the FineWeb-Edu sample-10BT.

The exponent αN ≈ 0.076 implies going from 10M to 300M parameters (30×) reduces loss by only ~23% [9]. This confirms no sharp phase transition exists in the scaling curve.

### 1.2 Empirical Evidence from Constrained Domains

TinyStories (Eldan & Li, 2023) demonstrated that models below 10M parameters can produce fluent stories on a simplified vocabulary of ~1,500 words [5]. However, this required: (a) hidden dimension ≥128 and ≥2 layers for coherence (not just grammar) [7], and (b) a heavily constrained domain that eliminates the long-tail vocabulary problem.

Crucially, TinyStories explicitly states that 125M-parameter models (GPT-2 small, GPT-Neo small) trained on general corpora (Pile, Common Crawl) "can rarely generate coherent and consistent English text beyond a few words even after extensive training" [6]. **This is a single-source claim from 2023** and may not reflect improvements in tokenization, data quality, or training recipes since then.

### 1.3 Community Training Runs at ≥100M Scale

The nanoGPT speedrun (2026) achieves val loss 3.28 on FineWeb with a 162M-parameter model trained on 1.8B tokens [F6:2][F6:3]. Karpathy's llm.c reproduces GPT-2 124M at val loss 3.28 on 1.8B FineWeb tokens [F6:1]. Cerebras-GPT starts at 111M parameters, following Chinchilla scaling rules on The Pile [F6:6][F6:7].

**Gap**: No community benchmark exists for models below 100M parameters on web-scale data. The nanoGPT scaling_laws.ipynb includes a ~12M parameter configuration [F6:8], but actual loss values at that scale could not be extracted from the notebook source.

## 2. VRAM Budget & CPU Offloading

### 2.1 A 30M-Param Model Fits Easily in 4GB

The VRAM breakdown for a 30M-param GPT with AdamW at seq-len 512:

| Component | Size |
|-----------|------|
| Model params (bf16) | ~60 MB |
| Gradients (bf16) | ~60 MB |
| AdamW states (fp32 m + v) | ~240 MB |
| Activations (seq-len 512, grad checkpointing) | ~50–100 MB |
| **Total** | **~410–460 MB** |

This leaves **~3.5GB headroom** on a 4GB GPU [F7:7]. Even without gradient checkpointing, total usage stays under 1GB at seq-len 512 [F7:2]. CPU offloading would only become necessary at seq-len >4,096 or batch_size >32, where attention activations dominate [F7:8].

### 2.2 ZeRO-Offload Is Proven but Unnecessary at This Scale

ZeRO-Offload (Rajbhandari et al. 2021) is the foundational technique for CPU-offloaded AdamW, proven to train 10B+ parameter models on a single GPU [F2:1]. DeepSpeed's CPU Adam implementation is 5–7× faster than standard PyTorch [F2:3]. However, for a 30M-param model, offloading introduces PCIe transfer overhead with no memory benefit [F7:6].

FSDP2 with CPU offload shows documented convergence slowdowns compared to DDP in practice [F2:5], further suggesting it is counterproductive for models that fit in GPU memory.

**Resolution**: CPU offloading is a proven technique for large models (>1B params) but is unnecessary and potentially harmful for a 30M-param model on 4GB VRAM. The brief's Question 2 may rest on a false assumption.

## 3. PyTorch SDPA & FlashAttention

### 3.1 Memory vs. Compute: The O(N) Distinction

FlashAttention achieves O(N) HBM memory by tiling computation and never materializing the full attention matrix [F3:1][F3:2]. However, FLOPs remain O(N²) — the improvement is in memory I/O, not computation [F3:1]. At seq-len 1024, Flash runs ~38× faster than the Math backend on A100 (2,272 μs vs 87,472 μs) [F3:7].

For seq-len 256–512 on consumer GPUs, the O(N) memory savings are meaningful: a naive attention matrix at seq-len 512 with 12 heads would require 512×512×12×2 bytes ≈ 6MB per layer, which FlashAttention avoids materializing entirely.

### 3.2 Edge Cases & Silent Fallbacks

The most common pitfall: **FP32 inputs silently force fallback to the Math backend** (O(N²) memory), emitting only a UserWarning [F3:4]. For a model using bf16 forward pass, this should not occur — but any accidental fp32 cast (e.g., from an intermediate operation) would silently degrade performance.

Additional constraints:
- Flash Attention requires head_dim ≤ 512; cuDNN backend restricts to head_dim ≤ 128 on consumer GPUs [F3:5]
- `is_causal=True` is mutually exclusive with `attn_mask` [F3:6]
- ROCm (AMD GPUs) silently upcasted all SDPA to fp32 in PyTorch 2.5.0, causing 2× slowdown [F3:8] — relevant if using non-NVIDIA hardware
- cuDNN backend requires dropout_p to be a multiple of 1/16 [F3:10]

**Gap**: No benchmark data exists for SDPA memory usage specifically at seq-len 256 and 512 on 4GB consumer GPUs (RTX 2050/GTX 1650 class).

## 4. Datasets & Training Data

### 4.1 FineWeb-Edu Is the Strongest Candidate

FineWeb-Edu sample-10BT contains ~9.7B GPT-2 tokens of educational web pages, filtered by a BERT-like classifier (F1=82%) trained on Llama3-70B-Instruct annotations, retaining pages scoring ≥3 on a 0–5 scale [F4:1][F4:2]. This quality filtering means each token carries more learning signal than unfiltered web crawls [F4:10].

The 10B token sample is a reasonable starting point for a 30M-param model. Scaling laws suggest diminishing returns beyond ~10–50B tokens for sub-300M models [F4:6], so the 100B sample would offer only marginal gains.

### 4.2 Alternative Datasets

| Dataset | Tokens | Quality Filtering | Suitability for 30M |
|---------|--------|-------------------|---------------------|
| FineWeb-Edu sample-10BT | ~9.7B | Educational classifier (F1=82%) | **Best choice** |
| C4 (English) | ~175B | Basic deduplication only | Noisy; less signal per token [F4:7] |
| SlimPajama | 627B | Deduplicated from RedPajama | Designed for ≥1B models; excessive diversity [F4:5] |
| The Pile | 825 GiB | 22 diverse sub-datasets | Token-per-domain ratio too diluted for 30M [F4:8] |

**Gap**: No direct benchmarks comparing these datasets on models ≤300M params exist in the literature [F4: Dead ends].

### 4.3 Training Token Budget

Chinchilla suggests ~20 tokens per parameter for compute-optimal training [4], implying ~600M tokens for a 30M model. The 9.7B FineWeb-Edu sample is ~16× this minimum — likely sufficient to reach the model's loss floor well before convergence. However, the model's capacity ceiling (30M params) means it cannot absorb the full dataset's information regardless of token count.

## 5. Coherence Evaluation

### 5.1 The Loss-to-Readability Mapping

The Mirostat paper (2020) provides the only published calibration between cross-entropy and human-judged text quality [F5:1]:

| Cross-Entropy (nats/token) | Quality |
|---------------------------|---------|
| < 2.5 | "Boredom trap" — excessive repetition [F5:2] |
| ~3.0 | Sweet spot — best fluency/coherence; >50% human-raters fooled [F5:1] |
| > 5.0 | "Confusion trap" — increasing incoherence [F5:3] |
| 5.13 | Human-written text scored by GPT-2 [F5:4] |

**Critical caveat**: This measures *sampling* cross-entropy (how predictable generated text is to the model), not *training* loss (how well the model fits data). A model with higher training loss can still produce high-quality text if sampling is controlled well [F5:12].

### 5.2 Known Baselines

- GPT-2 124M achieves val loss ~3.12 nats/token on OpenWebText [F5:7]
- GPT-2 XL 1.5B achieves ~2.54 [F5:7]
- GPT-2 124M fine-tuned on OWT reaches ~2.85 [F5:8]

For a 30M-param model, the training loss floor will be significantly higher than GPT-2 124M due to capacity constraints. Whether it can reach the ~3 nats/token sweet spot for sampling cross-entropy is unknown.

### 5.3 Perplexity Is Necessary But Not Sufficient

A 2026 workshop paper demonstrated that zero-parameter, deliberately naive samplers can achieve "SOTA" generative perplexity while producing completely incoherent text [F5:9]. A periodic top-k sampler achieved gen-PPL of 29.4 (lower than real models) while being incoherent by construction [F5:10]. This confirms that perplexity alone does not guarantee coherence — human evaluation remains essential.

## Open Questions

1. **No loss-to-coherence calibration exists for small models.** We cannot map a cross-entropy loss value to human-judged readability for 30M-param models, making it impossible to predict whether training will produce coherent text based on loss alone.

2. **The brief's CPU offloading premise may be false.** F2[7] and F7[7] show 30M params fits entirely in 4GB VRAM without offloading. The investigation should resolve whether offloading is actually needed or if this was based on incorrect assumptions about model/optimizer size.

3. **No empirical training-run data exists at 30M scale.** All claims about minimum viable size are extrapolations from scaling laws or constrained-domain experiments (TinyStories), not direct measurements on web-scale data.

4. **Dataset comparison at sub-300M scale is absent.** No benchmarks exist comparing FineWeb-Edu vs C4 vs SlimPajama on models this small.

5. **TinyStories' 125M claim may be outdated.** The assertion that 125M models "rarely generate coherent English" is single-source from 2023 [F1:6] and may not reflect current best practices (better tokenization, higher-quality data, longer training).

6. **Tokenization effects at sub-300M scale are unaddressed.** GPT-2 BPE vocab size (50,257) vs SentencePiece alternatives could significantly affect training efficiency and loss-to-quality mapping at this scale.

7. **What sampling strategy bridges training loss to coherent output?** Even if the model reaches acceptable training loss, the sampling method (temperature, top-k, top-p, Mirostat) critically determines whether output is readable.

8. **What is the actual VRAM usage at seq-len 256/512 on 4GB consumer GPUs?** F7 provides estimates but no empirical measurements on RTX 2050/GTX 1650 class hardware.

## Sources

1. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361. https://arxiv.org/abs/2001.08361
2. Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models." arXiv:2203.15556. https://arxiv.org/abs/2203.15556
3. Eldan, R. & Li, L. (2023). "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" arXiv:2305.07759. https://arxiv.org/abs/2305.07759
4. Rajbhandari, S. et al. (2021). "ZeRO-Offload: Democratizing Billion-Scale Model Training." arXiv:2101.06840. https://arxiv.org/abs/2101.06840
5. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." arXiv:2205.14135. https://arxiv.org/abs/2205.14135
6. Mirostat — An Adaptive Text Sampler (2020). "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity." arXiv:2007.14966v2. https://arxiv.org/abs/2007.14966v2
7. FineWeb-Edu Dataset Card. HuggingFace. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
8. nanoGPT. Karpathy. https://github.com/karpathy/nanoGPT
9. nanoGPT Speedrun. nilmamano. https://nilmamano.com/blog/nanogpt-speedrun
10. Cerebras-GPT. (2023). arXiv:2304.03208. https://arxiv.org/abs/2304.03208
11. SlimPajama. (2023). arXiv:2309.10818. https://arxiv.org/abs/2309.10818
12. PyTorch SDPA Documentation. https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
13. PyTorch Gradient Checkpointing. https://docs.pytorch.org/docs/2.13/checkpoint.html
14. DeepSpeed ZeRO-Offload Tutorial. https://www.deepspeed.ai/tutorials/zero-offload/
15. "Do Not Be Fooled by Perplexity" (2026). arXiv:2606.08417v1. https://arxiv.org/abs/2606.08417v1
16. microgpt. Karpathy (2026). https://karpathy.github.io/2026/02/12/microgpt/
17. Tiny Language Models (2025). arXiv:2507.14871. https://arxiv.org/abs/2507.14871
18. SDPA FlashAttention Source Analysis. https://gist.github.com/rkayaith/9eb401fddaad27d6b2edc1a496bea7fc
19. FSDP2 CPU Offload Convergence Issue. https://github.com/pytorch/pytorch/issues/154984
20. nanoGPT Scaling Laws Analysis. https://deepwiki.com/karpathy/nanogpt/7-scaling-laws-analysis
