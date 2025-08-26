import os
from importlib import import_module
from utils.config import parse_args_and_config
from utils.factory import instantiate
from utils.seed import seed_everything
from utils.checkpoint import load_checkpoint
from trainer.loop import Trainer

def main():
    cfg = parse_args_and_config()
    seed_everything(cfg["training"].get("seed"))

    data = instantiate(cfg.get("data"))
    model = instantiate(cfg.get("model"))
    loss_fn = instantiate(cfg.get("loss"))

    optim_cfg = cfg.get("optimizer")
    mod_name, cls_name = optim_cfg["target"].rsplit(".", 1)
    Optim = getattr(import_module(mod_name), cls_name)
    optimizer = Optim(model.parameters(), **(optim_cfg.get("params") or {}))

    scheduler = None
    if cfg.get("scheduler"):
        sch_cfg = cfg["scheduler"]
        mod_name, cls_name = sch_cfg["target"].rsplit(".", 1)
        Sch = getattr(import_module(mod_name), cls_name)
        scheduler = Sch(optimizer, **(sch_cfg.get("params") or {}))

    metrics = {}
    for name, block in (cfg.get("metrics") or {}).items():
        metrics[name] = instantiate(block)

    callbacks = []
    for block in (cfg.get("callbacks") or []):
        callbacks.append(instantiate(block))

    tcfg = cfg["training"]
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        callbacks=callbacks,
        device=tcfg.get("device", "auto"),
        epochs=int(tcfg.get("epochs", 10)),
        mixed_precision=bool(tcfg.get("mixed_precision", False)),
        grad_accum_steps=int(tcfg.get("grad_accum_steps", 1)),
        grad_clip=tcfg.get("grad_clip", None),
        log_interval=int(tcfg.get("log_interval", 50)),
        ckpt_dir=tcfg.get("ckpt_dir", "outputs/exp1"),
        monitor=tcfg.get("monitor", "val/accuracy"),
        monitor_mode=tcfg.get("monitor_mode", "max"),
        config=cfg
    )

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    if tcfg.get("resume_from"):
        ckpt = load_checkpoint(tcfg["resume_from"], model, optimizer, scheduler,
                               map_location=trainer.device)
        print(f"Resumed from {tcfg['resume_from']} at epoch={ckpt.get('epoch')}")

    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
