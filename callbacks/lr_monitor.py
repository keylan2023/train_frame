from .base import Callback

class LearningRateMonitor(Callback):
    def on_batch_end(self, trainer, step, logs):
        if trainer.optimizer is None: return
        lrs = [group.get("lr", None) for group in trainer.optimizer.param_groups]
        logs["lr"] = lrs[0] if lrs else None
