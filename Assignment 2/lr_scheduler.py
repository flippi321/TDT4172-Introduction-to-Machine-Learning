import math
from dataclasses import dataclass

@dataclass
class WarmupCosineCfg:
    base_lr: float            # peak LR reached after warmup
    total_steps: int          # total optimizer steps (usually epochs * steps_per_epoch)
    warmup_steps: int = 0     # warmup steps (linear warmup)
    min_lr: float = 0.0       # final LR floor at the end of cosine

class WarmupCosineLRScheduler:
    """
    Manual LR scheduler with linear warmup and cosine decay.
    Call .step() once per optimizer step (usually per batch).
    """
    def __init__(self, optimizer, cfg: WarmupCosineCfg):
        self.optimizer = optimizer
        self.cfg = cfg
        self.step_num = 0  # number of completed steps

        # safety
        self.cfg.total_steps = max(1, int(self.cfg.total_steps))
        self.cfg.warmup_steps = max(0, int(self.cfg.warmup_steps))
        self._apply_lr(self.get_lr_for_step(0))

    def get_lr_for_step(self, step: int) -> float:
        if self.cfg.warmup_steps > 0 and step < self.cfg.warmup_steps:
            # Linear warmup from 0 -> base_lr
            return self.cfg.base_lr * (step / float(self.cfg.warmup_steps))
        # Cosine decay from base_lr -> min_lr
        progress = (step - self.cfg.warmup_steps) / max(1, (self.cfg.total_steps - self.cfg.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)  # clamp to [0,1]
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_lr + (self.cfg.base_lr - self.cfg.min_lr) * cosine

    def _apply_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self):
        """Advance one step and apply new LR."""
        self.step_num += 1
        lr = self.get_lr_for_step(self.step_num)
        self._apply_lr(lr)
        return lr

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "cfg": vars(self.cfg),
        }

    def load_state_dict(self, state):
        self.step_num = state["step_num"]
        self.cfg = WarmupCosineCfg(**state["cfg"])
        # restore LR at current step
        self._apply_lr(self.get_lr_for_step(self.step_num))
