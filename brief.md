# Research Brief

**Date**: 2026-07-20 · **Depth**: standard

## Question

Can a ~30M parameter dense GPT-style autoregressive transformer, trained on consumer GPUs (4 GB VRAM) using CPU-offloaded AdamW and FlashAttention (via PyTorch SDPA), produce coherent English text? Specifically:

1. What is the minimum viable model size for a GPT-style transformer to generate coherent English, and what do scaling-law studies say about the 30M–300M parameter range?
2. Is CPU-offloaded AdamW (fp32 master weights on CPU, bf16 model + grads on GPU) a proven technique for 4 GB VRAM training, or are there hidden failure modes (NaN, divergence, slow convergence)?
3. Does PyTorch's `F.scaled_dot_product_attention` with `is_causal=True` deliver the claimed O(N) memory/compute benefits for sequence lengths 256–512, and are there edge cases where it degrades?
4. What datasets (FineWeb-Edu sample-10BT or alternatives) are suitable for training a tiny GPT, and how many tokens are needed to reach reasonable loss at ~30M parameters?
5. What are the known limitations of tiny language models compared to larger ones, and what tasks can they reasonably accomplish?

## Scope

**In:**
- GPT-style decoder-only autoregressive transformers only (no encoder-decoder, no bidirectional)
- Model sizes: 10M–300M parameters, with emphasis on ~30M
- Training on consumer GPUs with ≤4 GB VRAM (RTX 2050 / GTX 1650 / similar)
- CPU-offloaded optimizer techniques (DeepSpeed ZeRO, custom offload)
- FlashAttention / SDPA memory/compute tradeoffs at sequence lengths up to 512
- Scaling law literature relevant to sub-1B models (Chinchilla, Kaplan et al., and follow-ups)
- Datasets: FineWeb-Edu, C4, WikiText, OpenWebText, The Pile, SlimPajama
- Coherence evaluation: perplexity, human judgment, simple task benchmarks (e.g., HellaSwag, LAMBADA at small scale)

**Out:**
- Training on CPUs only (CPU-only training without any GPU)
- Mixture-of-Experts or sparse attention architectures
- Quantization (GPTQ, AWQ, GGUF) at inference time
- Fine-tuning pretrained models — this is scratch training
- Inference optimization (vLLM, speculative decoding)
- RLHF or instruction tuning
- Multi-modal or vision-language models

## Assumptions

- **Audience**: A practitioner (the user) building Tiny-GPT in `/home/pragadeesh/Tiny-GPT` who wants evidence-based answers to decide architecture/hyperparameter choices for a 30M dense transformer on a 4 GB GPU. Intermediate-to-advanced ML knowledge assumed.
- **Time frame**: Literature search covers 2018–2026 (Kaplan 2020 through current). Findings should be current as of July 2026.
- **Region / language**: English text only. No multilingual considerations.
- **Hardware baseline**: 4 GB VRAM GPU (GTX 1650 / RTX 2050 class), 16+ GB CPU RAM. PyTorch 2.x with CUDA 11.8+.
- **Precision**: BF16 forward pass, fp32 optimizer states. No mixed-precision FP16 training (loss scaling complexity is out of scope unless directly relevant to CPU-offload).
- **Dataset assumption**: FineWeb-Edu sample-10BT is the primary candidate; alternatives are evaluated only if FineWeb-Edu has documented issues for tiny models.
- **Evaluation of "coherence"**: Subjective readability of generated text plus perplexity on held-out data. Formal benchmarks (LAMBADA, HellaSwag) are mentioned but not required for scoping.
- **Definition of "hidden failure modes"**: Includes NaN/Inf losses, silent gradient corruption, optimizer state desynchronization, and convergence pathologies specific to CPU-GPU weight splitting.
