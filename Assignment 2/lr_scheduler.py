import numpy as np
from keras import callbacks

class WarmupCosineLRScheduler(callbacks.Callback):
    def __init__(self, warmup_epochs: int, total_epochs: int, max_lr: float = 1e-3, min_lr: float = 0.0):
        super().__init__()
        assert total_epochs > 0 and warmup_epochs >= 0 and total_epochs >= warmup_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.lr_history = []

    def _compute_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            progress = (epoch + 1) / self.warmup_epochs
            return self.min_lr + (self.max_lr - self.min_lr) * progress
        denom = max(1, self.total_epochs - self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / denom
        cosine = 0.5 * (1.0 + np.cos(np.pi * np.clip(progress, 0.0, 1.0)))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def _set_optimizer_lr(self, lr: float):
        opt = self.model.optimizer
        # Prefer .learning_rate if present (Keras 3)
        if hasattr(opt, "learning_rate"):
            lr_attr = opt.learning_rate
            # If it's a backend variable/tensor (e.g., TF Variable), use .assign
            if hasattr(lr_attr, "assign"):
                lr_attr.assign(lr)
            else:
                # Plain float or schedule — set the attribute directly
                setattr(opt, "learning_rate", lr)
            return
        # Fallback for older APIs
        if hasattr(opt, "lr"):
            if hasattr(opt.lr, "assign"):
                opt.lr.assign(lr)
            else:
                opt.lr = lr

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(self._compute_lr(epoch))
        self.lr_history.append(lr)
        self._set_optimizer_lr(lr)