from .base import Callback
import math

class EarlyStopping(Callback):
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.num_bad = 0
        self.should_stop = False

    def on_validation_end(self, trainer, epoch, logs):
        monitor = trainer.monitor_value
        if monitor is None: return
        improved = (monitor > self.best) if self.mode == "max" else (monitor < self.best)
        if improved:
            self.best = monitor
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
