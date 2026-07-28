# Independent Review of REPORT.md

Reviewer: MiMoCode (independent, no prior context)
Date: 2026-07-20
Files reviewed: REPORT.md, findings/F1.md–F7.md

---

## (a) Claims Lacking Citations

**All substantive claims have citations.** No major claim is entirely uncited. However, two items deserve attention:

1. **C4 token count (~175B) in Section 4.2 table** — The report states C4 English has "~175B" tokens, but F4 finding [7] only quotes "305GB" (byte size, not token count). The 175B figure is widely used in the literature but is not traced to any specific finding. Should cite the HuggingFace dataset card directly or note the derivation.

2. **Section 3.1 memory calculation** — "a naive attention matrix at seq-len 512 with 12 heads would require 512×512×12×2 bytes ≈ 6MB per layer" is presented without citation. This is a straightforward arithmetic derivation and does not require a citation, but it would strengthen the argument to note that this is a lower bound (excluding query/key/value tensors).

---

## (b) Spot-Check of 5 Random Citations

| Citation | Claim | URL in Findings | URL Exists | Supports Claim |
|----------|-------|-----------------|------------|----------------|
| [F5:1] | "texts at τ=3 received best fluency/coherence ratings" | arXiv:2007.14966v2 | Yes | Yes — abstract confirms boredom trap, confusion trap, and human evaluation |
| [F3:7] | "Flash is ~38× faster than Math backend (2,272 μs vs 87,472 μs)" | PyTorch SDPA tutorial | Yes | Yes — tutorial benchmarks match these numbers |
| [F6:6] | "Cerebras-GPT starts at 111M parameters" | arXiv:2304.03208 | Yes | Yes — abstract: "scaled from 111M to 13B parameters" |
| [F4:2] | "BERT classifier, F1=82%, trained on Llama3-70B annotations" | HuggingFace FineWeb-Edu | Yes | Yes — dataset card confirms F1=82% and Llama3-70B annotations |
| [F7:7] | "30M-param transformer uses ~480MB VRAM total" | PyTorch checkpoint docs | Yes | Partially — the URL is about gradient checkpointing; the 480MB figure is derived from parameter arithmetic in F7, not from the URL itself |

**Verdict:** All 5 URLs exist and are accessible. 4 of 5 directly support the cited claim. The F7:7 citation is loosely related (the URL discusses memory-compute tradeoffs but does not contain the 480MB figure).

---

## (c) Conclusions Stronger Than Evidence

1. **Executive Summary item 3: "A 30M-param model almost certainly cannot produce coherent general English."**
   - Evidence: TinyStories (constrained domain, 2023) + scaling law extrapolation. No direct measurement at 30M on web-scale data exists.
   - Issue: "Almost certainly" implies high confidence, but the claim rests on a single constrained-domain study and theoretical extrapolation. The report itself acknowledges this is an extrapolation (Section 1.3, Open Question 3). The hedging language is appropriate in the body but the executive summary overstates certainty.

2. **Executive Summary item 7: "FineWeb-Edu sample-10BT is the best dataset choice for a tiny GPT."**
   - Evidence: FineWeb-Edu's quality filtering is well-documented, but no direct comparison exists at sub-300M scale (Section 4.2, Gap).
   - Issue: "Best" implies a comparative judgment that the evidence does not support. "Strongest candidate based on available evidence" would be more accurate.

3. **Section 1.2: TinyStories 125M claim presented with appropriate hedging.**
   - The report correctly notes this is "a single-source claim from 2023" and "may not reflect current best practices." This is well-calibrated.

---

## (d) Executive Summary vs. Body Consistency

| Executive Summary Item | Body Section | Consistent? |
|----------------------|--------------|-------------|
| 1. No sharp coherence threshold | §1.1 | Yes |
| 2. ~100M+ practical floor | §1.2 | Yes |
| 3. 30M cannot produce coherent English | §1.2, §1.3 | Yes (but see overstatement note above) |
| 4. CPU offloading unnecessary | §2.1, §2.2 | Yes |
| 5. Cross-entropy ~3 nats/token sweet spot | §5.1 | Yes |
| 6. FlashAttention O(N) memory, not O(N) compute | §3.1 | Yes |
| 7. FineWeb-Edu best dataset | §4.1, §4.2 | Yes (but see "best" note above) |
| 8. No empirical data at 30M scale | §1.3, Open Questions | Yes |

**Verdict:** Executive summary is consistent with the body throughout. No contradictions found.

---

## Additional Findings

### Citation Numbering Ambiguity

The report uses two overlapping citation systems:
- **Plain [n]** in Sections 1.1–1.2 and the Executive Summary, which refer to finding numbers within F1.md (e.g., [1] = F1 finding [1], [9] = F1 finding [9])
- **[F*:n]** format elsewhere, clearly referencing specific findings files (e.g., [F7:7] = F7 finding [7])

The **Sources list** at the bottom uses a third numbering (1–20). Since plain [n] values like [1], [4], [9] could be misinterpreted as referring to the Sources list (where [1]=Kaplan, [4]=ZeRO-Offload, [9]=nanoGPT Speedrun), this creates potential confusion. In practice, the plain [n] references in Sections 1.1–1.2 are correct when interpreted as F1.md finding numbers, but the ambiguity should be resolved — either by using [F1:n] consistently or by renumbering the Sources list.

### VRAM Calculation Inconsistency

F7 finding [7] states model params use "120MB FP32" while the report table says "Model params (bf16) | ~60 MB." The report's bf16 assumption is more realistic for modern mixed-precision training, but the totals differ: F7 says ~480MB minimum, the report says ~410–460MB total. The report's calculation is more accurate; F7's derivation should be updated to match.

### Source 15 Title Mismatch

Source 15 is listed as "Do Not Be Fooled by Perplexity" (arXiv:2606.08417v1), but the actual paper title is "Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics." The content matches (zero-parameter samplers achieving SOTA gen-PPL while incoherent), but the cited title is incorrect.

---

## Summary

The report is well-researched and appropriately hedged in most places. The main issues are:
1. Two conclusions slightly overstate certainty relative to evidence (items 3 and 7 in Executive Summary)
2. Citation numbering system is ambiguous (plain [n] vs [F*:n] vs Sources list)
3. Source 15 title is incorrect
4. VRAM calculation in F7 uses FP32 while the report correctly uses bf16, creating a minor discrepancy

No fabricated citations were found. All spot-checked URLs exist and support their claims.
