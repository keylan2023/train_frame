import os
from .base import Callback
from utils.checkpoint import save_checkpoint

class ModelCheckpoint(Callback):
    def __init__(self, save_top_k=1, mode="max"):
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_k = []  # (metric, path)

    def on_validation_end(self, trainer, epoch, logs):
        monitor = trainer.monitor_value
        if monitor is None: return
        ckpt_dir = trainer.ckpt_dir
        name = trainer.monitor_name.replace("/", "_")
        fname = f"epoch{epoch:03d}_{name}={monitor:.4f}.pth"
        path = os.path.join(ckpt_dir, fname)
        save_checkpoint(path, trainer.model, trainer.optimizer, trainer.scheduler,
                        epoch, trainer.best_metric, trainer.config)
        # manage top-k
        self.best_k.append((monitor, path))
        reverse = (self.mode == "max")
        self.best_k.sort(key=lambda x: x[0], reverse=reverse)
        # remove beyond top_k
        for m, p in self.best_k[self.save_top_k:]:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass
        self.best_k = self.best_k[:self.save_top_k]
