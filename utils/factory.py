import importlib

def instantiate(cfg: dict):
    """Instantiate an object from a block with keys {target, params}."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict) or "target" not in cfg:
        raise ValueError(f"Invalid config for instantiate: {cfg}")
    target = cfg["target"]
    params = cfg.get("params") or {}
    module_path, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, attr_name)
    return cls(**params)
