import yaml, os

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # auto create out_dir
    out_dir = cfg.get("train", {}).get("out_dir", "out/tmp")
    os.makedirs(out_dir, exist_ok=True)
    return cfg