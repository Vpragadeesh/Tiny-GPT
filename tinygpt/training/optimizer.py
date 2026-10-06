"""CPU-offloaded AdamW optimizer."""

import torch


class CPUOffloadAdamW:
    """
    AdamW with ALL state (master weights, momentum, variance) in fp32 on CPU.
    fp16 m/v was the culprit for NaN — Adam variance accumulates squared
    gradients that easily exceed fp16 max (65504) → overflow → NaN.
    GPU holds only fp16 model weights + fp16 gradients (~1 GB VRAM).
    CPU RAM: fp32 master(2 GB) + fp32 m(2 GB) + fp32 v(2 GB) ≈ 6.2 GB.
    Expose param_groups so torch.amp.GradScaler.unscale_() works correctly.
    """

    def __init__(self, gpu_params, lr=3e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.gpu_params = list(gpu_params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0

        # fp32 master copies + fp32 momentum/variance on CPU
        self.master = [p.data.float().cpu() for p in self.gpu_params]
        self.m = [torch.zeros_like(mp) for mp in self.master]   # fp32
        self.v = [torch.zeros_like(mp) for mp in self.master]   # fp32

        # GradScaler compatibility: unscale_() iterates param_groups
        self.param_groups = [{"params": self.gpu_params}]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, gp in enumerate(self.gpu_params):
            if gp.grad is None:
                continue
            g = gp.grad.data.float().cpu()   # fp16 grad → fp32

            # Decoupled weight decay
            self.master[i].mul_(1.0 - self.lr * self.wd)

            # Adam moments (all fp32 — no overflow risk)
            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            # Bias-corrected parameter update
            self.master[i].addcdiv_(
                self.m[i] / bc1,
                (self.v[i] / bc2).sqrt_().add_(self.eps),
                value=-self.lr,
            )

            # Push updated fp32 weights → GPU fp16
            gp.data.copy_(self.master[i])

    def zero_grad(self):
        for gp in self.gpu_params:
            gp.grad = None

    def set_lr(self, lr):
        self.lr = lr

    def state_dict(self):
        return {"t": self.t, "master": self.master, "m": self.m, "v": self.v}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        self.master = sd["master"]
        self.m = sd["m"]
        self.v = sd["v"]
        for gp, mp in zip(self.gpu_params, self.master):
            gp.data.copy_(mp.data)

class OptimizerWrap:
    def __init__(self, o): self.opt = o
    def step(self):       self.opt.step()
    def zero_grad(self):  self.opt.zero_grad(set_to_none=True)
    def set_lr(self, lr):
        for pg in self.opt.param_groups: pg["lr"] = lr
    def state_dict(self):       return self.opt.state_dict()
    def load_state_dict(self, sd): self.opt.load_state_dict(sd)

def make_optimizer(model, lr, device):
    if device == "cuda":
        return CPUOffloadAdamW(model.parameters(), lr=lr)
    else:
        inner = torch.optim.AdamW(model.parameters(), lr=lr)
        return OptimizerWrap(inner)
