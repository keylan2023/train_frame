import os, csv, torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, metrics: dict, callbacks: list,
                 device="auto", epochs=10, mixed_precision=False, grad_accum_steps=1,
                 grad_clip=None, log_interval=50, ckpt_dir="outputs/exp1",
                 monitor="val/accuracy", monitor_mode="max", config=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.epochs = epochs
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.ckpt_dir = ckpt_dir
        self.monitor_name = monitor
        self.monitor_mode = monitor_mode
        self.best_metric = None
        self.config = config or {}

        # device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # AMP
        self.use_amp = bool(mixed_precision) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._last_val_logs = {}

    @property
    def monitor_value(self):
        return self._last_val_logs.get(self.monitor_name)

    def fit(self, train_loader, val_loader=None):
        for cb in self.callbacks: cb.on_train_start(self)

        csv_path = os.path.join(self.ckpt_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = None

        global_step = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for m in self.metrics.values(): m.reset()

            for step, batch in enumerate(train_loader, start=1):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with autocast(enabled=self.use_amp):
                    logits = self.model(inputs)
                    loss = self.loss_fn(logits, targets) / self.grad_accum_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % self.grad_accum_steps == 0:
                    if self.grad_clip is not None:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()

                running_loss += loss.item() * self.grad_accum_steps

                for name, metric in self.metrics.items():
                    metric.update(logits.detach(), targets)

                if step % self.log_interval == 0:
                    logs = {
                        "epoch": epoch,
                        "step": step,
                        "train/loss": running_loss / step,
                    }
                    for name, metric in self.metrics.items():
                        logs[f"train/{name}"] = metric.compute()
                    for cb in self.callbacks: cb.on_batch_end(self, global_step, logs)

                global_step += 1

            train_logs = {"epoch": epoch, "train/loss": running_loss / max(1, step)}
            for name, metric in self.metrics.items():
                train_logs[f"train/{name}"] = metric.compute()

            val_logs = {}
            if val_loader is not None:
                self.model.eval()
                for m in self.metrics.values(): m.reset()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        logits = self.model(inputs)
                        loss = self.loss_fn(logits, targets)
                        val_loss += loss.item()
                        for name, metric in self.metrics.items():
                            metric.update(logits, targets)
                val_logs["val/loss"] = val_loss / max(1, len(val_loader))
                for name, metric in self.metrics.items():
                    val_logs[f"val/{name}"] = metric.compute()

            logs = {**train_logs, **val_logs}
            if csv_writer is None:
                fieldnames = ["epoch"] + [k for k in logs.keys() if k != "epoch"]
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
            csv_writer.writerow(logs)
            csv_file.flush()

            self._last_val_logs = logs
            monitor_val = logs.get(self.monitor_name)
            improved = False
            if monitor_val is not None:
                if self.best_metric is None:
                    improved = True
                else:
                    if self.monitor_mode == "max":
                        improved = monitor_val > self.best_metric
                    else:
                        improved = monitor_val < self.best_metric
                if improved:
                    self.best_metric = monitor_val

            for cb in self.callbacks: cb.on_validation_end(self, epoch, logs)
            for cb in self.callbacks: cb.on_epoch_end(self, epoch, logs)

            stop = any(getattr(cb, "should_stop", False) for cb in self.callbacks)
            if stop: break

        for cb in self.callbacks: cb.on_train_end(self)
        csv_file.close()
