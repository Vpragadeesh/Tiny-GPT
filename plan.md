That's a great catch by the big pickle. Smaller models will absolutely trip over those edge cases—especially the silent failure of gradient checkpointing and the lingering `argparse` definitions.

Here is the revised, bulletproofed `plan.md`. It standardizes the environment across all four files so you can feed the exact same `TinyGPT` class to the 10B model without it getting confused by file-specific logic.

---

```markdown
# Refactoring Plan: Convert Mixture-of-Experts (MoE) to Dense Transformer

## System Prompt Context for the AI
You are an expert Python developer. Your task is to refactor a codebase by stripping out the Mixture-of-Experts (MoE) architecture and replacing it with a standard, dense autoregressive Transformer.

**Execution Rules:**
1. Process one file at a time. Do not attempt to rewrite the entire project in a single response.
2. Only modify the specific code blocks requested in each phase.
3. Preserve all existing imports, tokenization logic, and DeepSpeed/optimizer setup unless explicitly told to change them.

## Target Files
* `main.py`
* `main_deepspeed.py`
* `kaggle_train.py`
* `run.py`

---

## Phase 1: Imports, Globals, and Argparse Cleanup
**Files to update:** All files.

**Actions:**
1. **In `run.py`:** Add the missing import at the top with the other torch imports:
   `from torch.utils.checkpoint import checkpoint as grad_checkpoint`
2. **In `main.py` and `kaggle_train.py`:** Add `USE_ACTIVATION_CHECKPOINT = True` to the global configuration block (near `BLOCK_SIZE`).
3. **In ALL files:** Delete the following global variables:
   * `NUM_EXPERTS = 8` (or 4)
   * `TOP_K = 2`
   * `AUX_LOSS_W = 0.01`
4. **In `kaggle_train.py` ONLY:** Locate the `parse_args()` function and **delete** the three `parser.add_argument` lines for `--num-experts`, `--top-k`, and `--aux-loss-w`.

---

## Phase 2: Architecture Refactoring (The Core Model)
**Files to update:** `main.py`, `main_deepspeed.py`, `kaggle_train.py`, `run.py`

**Action:**
Locate the section defining the neural network classes (`ExpertFFN`, `MoELayer`, `TransformerBlock`, `MoEGPT`).

1. **Delete** the `ExpertFFN` and `MoELayer` classes completely.
2. **Add** the new `FeedForward` class:
```python
class FeedForward(nn.Module):
    """Standard two-layer FFN with GELU."""
    def __init__(self):
        super().__init__()
        self.w1   = nn.Linear(EMBED_DIM, FFN_DIM)
        self.w2   = nn.Linear(FFN_DIM, EMBED_DIM)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))

```

3. **Replace** the existing `TransformerBlock` with this updated version:

```python
class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + Dense FFN, with residuals."""
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.ffn  = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

```

4. **Replace** the `MoEGPT` class with `TinyGPT`. Note: The `generate()` method remains exactly the same as the original file, just ensure it is indented under the new `TinyGPT` class.

```python
class TinyGPT(nn.Module):
    """Standard Dense GPT model."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * NUM_LAYERS) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        for block in self.blocks:
            if self.training and globals().get("USE_ACTIVATION_CHECKPOINT", False):
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        return logits, loss

    # KEEP EXISTING generate() METHOD HERE

```

---

## Phase 3: Initialization and Parameter Counting Updates

**Files to update:** `main.py`, `main_deepspeed.py`, `kaggle_train.py`, `run.py`

**Action:**
Locate where the model is instantiated (e.g., `model = MoEGPT()`).

1. **Change** the instantiation to: `model = TinyGPT()`
2. **Replace** the parameter counting block with standard dense counting.
**Delete** the lines calculating `_expert1` and `n_active`, and replace with:

```python
n_total = sum(p.numel() for p in model.parameters())
n_active = n_total  # In a dense model, all parameters are active

```

---

## Phase 4: Training Loop & Evaluation Cleanup

**Files to update:** `main.py`, `main_deepspeed.py`, `kaggle_train.py`

**Action for Training Loop:**

1. **Find** the training step inside the main loop.
2. **Update** the forward pass unpacking. Because `TinyGPT.forward()` no longer returns `aux_loss`, remove any references to it.
**Change from:**

```python
x, y = get_batch("train")
with torch.amp.autocast("cuda", ...):
    _, loss = model(x, y) # Ensure total_aux or aux_loss isn't being unpacked or added

```

3. Ensure that `loss` is directly divided by `GRAD_ACCUM` and backpropagated without adding `AUX_LOSS_W * total_aux`.

**Action for Evaluation:**
In the `estimate_loss(model)` function, ensure the forward pass only expects `_, loss`. Do not unpack a third variable.

---

## Phase 5: Verification and run.py Formatting

**File to update:** `run.py`

**Actions:**

1. Check the `apply_model_config_from_state_dict` function.
2. **Remove** the logic that attempts to read `NUM_EXPERTS` from `blocks.0.moe.router.weight`.
3. **Update** the logic for `FFN_DIM` to read from the dense feedforward layer instead:

```python
    ffn_key = "blocks.0.ffn.w1.weight"
    if ffn_key in state_dict:
        FFN_DIM = state_dict[ffn_key].shape[0]

```

4. Locate the print statement inside the `load_model()` function (around line 362). Remove `experts={NUM_EXPERTS}, ` from the formatted string so it prints cleanly.

```

```
