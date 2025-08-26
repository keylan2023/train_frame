class Callback:
    def on_train_start(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_start(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs: dict): pass
    def on_batch_end(self, trainer, step, logs: dict): pass
    def on_validation_end(self, trainer, epoch, logs: dict): pass
    def on_save_checkpoint(self, trainer, path): pass
