# -*- coding: utf-8 -*-
"""
Robust train_ddp.py
- 统一数据清洗：去 NaN/Inf、裁剪极端值
- AMP 可选（默认关闭）；梯度裁剪；更鲁棒的回归损失（Huber）
- 支持 DDP（--ddp 1），否则单进程
- 训练/验证/保存 ckpt + meta（含 feature_cols/window）
- 兼容你项目的特征与数据集生成：compute_tech_indicators / make_labels / make_sequence_dataset（仅在需要时加载）
"""

import os
import sys
import json
import math
import time
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist  # Added for DDP
from torch.nn.parallel import DistributedDataParallel as DDP  # Added for DDP
from torch.utils.data.distributed import DistributedSampler  # Added for DDP

from src.utils.config import load_config
from src.datasets import make_sequence_dataset
from src.feature_engineering import compute_tech_indicators
from src.labeling import make_labels

# models
from src.models.hybrid_transformer import HybridTransformerMTL
from src.models.multitask_lstm import MultiTaskLSTM
from src.models.multitask_transformer import MultiTaskTransformer

try:
    from src.models.multitask_transformer import MultiTaskTransformer  # 如果没有，会在构建时报错并提示忽略
    HAS_MultiTaskTransformer = True
except Exception:
    MultiTaskTransformer = None
    HAS_MultiTaskTransformer = False


# ---------------- Utils ----------------
def set_seed(seed: int = 123):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pick_device(req: str = "auto") -> str:
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if req == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "cpu"
    return req

def sanitize_array_(arr: np.ndarray, clip: float = 10.0):
    """就地清洗：替换 NaN/Inf，裁剪极端值"""
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None and clip > 0:
        np.clip(arr, -clip, clip, out=arr)

def batch_sanitize_(x: torch.Tensor, *ys: torch.Tensor, clip: float = 10.0) -> Tuple[torch.Tensor, ...]:
    xs = [x, *ys]
    outs = []
    for t in xs:
        if t is None: 
            outs.append(None); continue
        if torch.isnan(t).any() or torch.isinf(t).any():
            t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
        if clip is not None and clip > 0:
            t = torch.clamp(t, -clip, clip)
        outs.append(t)
    return tuple(outs)

def safe_mean(x: List[float]) -> float:
    vals = [v for v in x if v == v and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------- Data ----------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cls: Optional[np.ndarray], y_reg: Optional[np.ndarray]):
        '''
        # 保证 np.float32 且无 NaN/Inf
        '''
        self.X = X.astype(np.float32, copy=False)
        sanitize_array_(self.X)
        self.X = torch.from_numpy(self.X).float()  # Convert to tensor early
        self.yc = y_cls.astype(np.float32, copy=False) if y_cls is not None else None
        if self.yc is not None:
            sanitize_array_(self.yc)
            self.yc = torch.from_numpy(self.yc).float().view(-1, 1)  # Convert to tensor
        self.yr = y_reg.astype(np.float32, copy=False) if y_reg is not None else None
        if self.yr is not None:
            sanitize_array_(self.yr)
            self.yr = torch.from_numpy(self.yr).float().view(-1, 1)  # Convert to tensor

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = self.X[i]
        yc = self.yc[i] if self.yc is not None else torch.tensor(0.0)
        yr = self.yr[i] if self.yr is not None else torch.tensor(0.0)
        return x, yc, yr

def load_dataset_from_csv(cfg: Dict[str, Any], symbol: str, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    优先读取 cfg['data']['dataset_csv']；否则尝试 {out_dir}/{SYMBOL}_dataset.csv
    若没有就现场用 compute_tech_indicators/make_labels 做一次（只为跑通；推荐仍然事先离线生成）
    """
    ds_path = (cfg.get("data", {}) or {}).get("dataset_csv", "")
    if not ds_path:
        out_dir = (cfg.get("train", {}) or {}).get("out_dir", "out/mid/models")
        cand = [
            os.path.join(out_dir, f"{symbol}_dataset.csv"),
            os.path.join(out_dir, "dataset.csv"),
            os.path.join("out", "mid", symbol, "dataset.csv"),
        ]
        for p in cand:
            if os.path.exists(p):
                ds_path = p; break

    if ds_path and os.path.exists(ds_path):
        df = pd.read_csv(ds_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        # 猜测 feature 列：数值列且排除标签列
        label_cols = {"y_cls", "y_reg", "label", "target"}
        num_cols = [c for c in df.columns if c not in ("timestamp","ticker") and df[c].dtype != "object"]
        feat_cols = [c for c in df.columns if c not in ("id","timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight","insert_time","update_time","interface_id","fetch_time")]
        # 清洗
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols, how="any")
        X, y1, y2 = make_sequence_dataset(df, feat_cols, "y_reg" if "y_reg" in df.columns else None, window)
        sanitize_array_(X); 
        if y1 is not None: sanitize_array_(y1)
        if y2 is not None: sanitize_array_(y2)
        return X, y1, y2, feat_cols

    # 兜底：临时构建一次
    print("[data] dataset csv not found; build on the fly...")
    # 允许用户在 cfg['data'] 定义原始路径；否则用 akshare 现抓（避免复杂性，这里从 examples 兜底）
    raw_csv = (cfg.get("data", {}) or {}).get("raw_csv", "")
    if raw_csv and os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # 尝试 examples/sample_data.csv
        sample = os.path.join("examples", "sample_data.csv")
        if not os.path.exists(sample):
            raise FileNotFoundError("No dataset_csv found and no examples/sample_data.csv to fall back.")
        df = pd.read_csv(sample)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = compute_tech_indicators(df, cfg)
    df = make_labels(df, cfg)
    label_cols = {"y_cls", "y_reg", "label", "target"}
    num_cols = [c for c in df.columns if c not in ("timestamp","ticker") and df[c].dtype != "object"]
    feat_cols = [c for c in df.columns if c not in ("timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight","insert_time","update_time","interface_id","fetch_time")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols, how="any")
    X, y1, y2 = make_sequence_dataset(df, feat_cols, "y_reg" if "y_reg" in df.columns else None, window)
    sanitize_array_(X); 
    if y1 is not None: sanitize_array_(y1)
    if y2 is not None: sanitize_array_(y2)
    return X, y1, y2, feat_cols


# ---------------- Model builder ----------------
def build_model(model_name: str, input_dim: int, cfg: Dict[str, Any], device: str):
    m = model_name.lower()
    if m == "hybrid":
        model = HybridTransformerMTL(input_dim=input_dim, cfg=cfg)
    elif m == "lstm":
        model = MultiTaskLSTM(input_dim=input_dim, cfg=cfg)
    elif m == "transformer":
        if not HAS_MultiTaskTransformer:
            raise RuntimeError("Transformer model not available in this repo.")
        model = MultiTaskTransformer(input_dim=input_dim, cfg=cfg)
    else:
        raise ValueError(f"unknown model={model_name}")
    model.to(device)
    return model


# ---------------- Train/Eval ----------------
def compute_loss(outputs, yc, yr, class_weight=1.0):
    """
    兼容：返回可能是 dict/tuple
      - dict: {'logits':..., 'reg':...} 或 {'prob':..., 'reg':...}
      - tuple: (logits, reg)
    """
    if isinstance(outputs, dict):
        logits = outputs.get("logits", None)
        prob   = outputs.get("prob", None)
        reg    = outputs.get("reg", None)
        if logits is None and prob is not None:
            # 如果只有概率，转成 logits
            eps = 1e-6
            prob = torch.clamp(prob, eps, 1-eps)
            logits = torch.log(prob/(1-prob))
    else:
        logits, reg = outputs

    # 清洗
    logits, yc = batch_sanitize_(logits, yc, clip=20.0)[:2]
    reg, yr = batch_sanitize_(reg, yr, clip=20.0)[:2]

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight], device=logits.device)) if logits is not None else None
    huber = nn.SmoothL1Loss(beta=0.01) if reg is not None else None  # Huber 更稳

    loss_cls = bce(logits.view_as(yc), yc) if (bce is not None and yc is not None) else torch.tensor(0.0, device=logits.device if logits is not None else reg.device)
    loss_reg = huber(reg.view_as(yr), yr) if (huber is not None and yr is not None) else torch.tensor(0.0, device=reg.device if reg is not None else logits.device)
    loss = loss_cls + loss_reg
    return loss, loss_cls.detach(), loss_reg.detach()

def evaluate(model, dl, device: str) -> Dict[str, float]:
    model.eval()
    losses, lcs, lrs = [], [], []
    with torch.no_grad():
        for x, yc, yr in dl:
            x = x.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            x, yc, yr = batch_sanitize_(x, yc, yr, clip=20.0)
            out = model(x)
            loss, lc, lr = compute_loss(out, yc, yr)
            losses.append(float(loss.item()))
            lcs.append(float(lc.item()))
            lrs.append(float(lr.item()))
    return {"loss": safe_mean(losses), "loss_cls": safe_mean(lcs), "loss_reg": safe_mean(lrs)}

def train_loop(model, train_dl, valid_dl, cfg: Dict[str, Any], device: str, amp: str = "off", out_dir: str = "out/mid/models", prefix: str = "SP0_hybrid", rank: int = 0):
    epochs = int(cfg.get("train", {}).get("epochs", 20))
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    wd = float(cfg.get("train", {}).get("l2", 1e-4))
    class_weight = float(cfg.get("train", {}).get("class_weight", 1.0))
    patience = int(cfg.get("train", {}).get("early_stop_patience", 20))
    os.makedirs(out_dir, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler('cuda', enabled=(amp in ("fp16","bf16") and device.startswith("cuda")))  # Updated for deprecation
    use_autocast = (amp in ("fp16","bf16") and device.startswith("cuda"))
    autocast_dtype = torch.float16 if amp == "fp16" else torch.bfloat16

    best = float("inf")
    no_improve = 0
    ckpt_path = os.path.join(out_dir, f"{prefix}_best.pt")
    meta_path = os.path.join(out_dir, f"{prefix}_best_meta.json")

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for x, yc, yr in train_dl:
            x = x.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            x, yc, yr = batch_sanitize_(x, yc, yr, clip=20.0)

            opt.zero_grad(set_to_none=True)
            try:
                if use_autocast:
                    with torch.amp.autocast('cuda', dtype=autocast_dtype):  # Updated context
                        out = model(x)
                        loss, _, _ = compute_loss(out, yc, yr, class_weight=class_weight)
                else:
                    out = model(x)
                    loss, _, _ = compute_loss(out, yc, yr, class_weight=class_weight)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise FloatingPointError("NaN/Inf loss detected.")

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()

                losses.append(float(loss.item()))
            except Exception as e:
                # 降级处理：关闭 AMP，并记录一次
                print(f"[warn] step failed due to {e}; try disable AMP and sanitize.")
                if scaler.is_enabled():
                    scaler = torch.amp.GradScaler('cuda', enabled=False)

        # 验证
        metrics = evaluate(model, valid_dl, device)
        tr_loss = safe_mean(losses)
        if rank == 0:
            print(f"[{ep}/{epochs}] train={tr_loss:.6f} valid={metrics['loss']:.6f} (cls={metrics['loss_cls']:.6f}, reg={metrics['loss_reg']:.6f})")

        if metrics["loss"] < best - 1e-6:
            best = metrics["loss"]; no_improve = 0
            if rank == 0:
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(state_dict, ckpt_path)
                #torch.save(model.state_dict(), ckpt_path)
                # 保存 meta（特征与窗口用于线上推理）
                # 从 DataLoader 中推断
                feat_cols = getattr(train_dl.dataset, "feat_cols", None)
                if feat_cols is None and hasattr(train_dl.dataset, "X"):
                    # 无法直接拿列名，就写个占位：输入维度
                    feat_cols = [f"f{i}" for i in range(train_dl.dataset.X.shape[-1])]
                meta = {
                    "feature_cols": feat_cols,
                    "window": train_dl.dataset.X.shape[1],
                    "model": prefix.split("_")[-1]
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            no_improve += 1
            if no_improve >= patience:
                if rank == 0:
                    print(f"[early-stop] patience={patience} reached.")
                break

    return ckpt_path, meta_path


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["hybrid","lstm","transformer"], default="hybrid")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", default="out/mid/models")
    ap.add_argument("--ddp", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--amp", default="off", choices=["off","fp16","bf16"])
    args = ap.parse_args()

    # DDP init if enabled
    rank = 0
    world_size = 1
    if args.ddp:
        if os.environ.get('RANK') is None:
            print("[info] DDP requested but not launched with torchrun; setting up single-process DDP.")
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'  # Choose a free port
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        if rank == 0:
            print(f"[DDP] Initialized with world_size={world_size}")

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 123)))

    device = pick_device(args.device)
    if rank == 0:
        print(f"[info] Using device: {device}")

    window = int(cfg.get("train", {}).get("window", 64))

    # 加载数据 (only rank 0 loads, then broadcast if needed; but for simplicity, all load since data is small)
    X, y_cls, y_reg, feat_cols = load_dataset_from_csv(cfg, args.symbol.upper(), window)
    if X.shape[0] < 2:
        raise RuntimeError("Not enough training samples after windowing.")

    # 拆分：按 cfg['train_test_split_date'] 或 8:2
    split_date = cfg.get("train_test_split_date", None)
    if split_date:
        # 假设 CSV 中有 timestamp，重新从原 CSV 读取一遍用于时间切分
        ds_path = (cfg.get("data", {}) or {}).get("dataset_csv", "")
        if not ds_path:
            ds_path = os.path.join(cfg.get("train", {}).get("out_dir","out/mid/models"), "dataset.csv")
        df_all = pd.read_csv(ds_path)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        idx = (df_all["timestamp"] >= pd.to_datetime(split_date)).sum()
        # 近似估算窗口对齐后的切分点
        split_idx = max(1, min(X.shape[0]-1, idx - window))
    else:
        split_idx = int(X.shape[0]*0.8)

    Xtr, Xv = X[:split_idx], X[split_idx:]
    yctr = y_cls[:split_idx] if y_cls is not None else None
    ycv  = y_cls[split_idx:] if y_cls is not None else None
    yrtr = y_reg[:split_idx] if y_reg is not None else None
    yrv  = y_reg[split_idx:] if y_reg is not None else None

    if Xv.shape[0] == 0:
        # 保证验证集非空
        Xtr, Xv = X[:-1], X[-1:]
        if yctr is not None: yctr, ycv = yctr[:-1], yctr[-1:]
        if yrtr is not None: yrtr, yrv = yrtr[:-1], yrv[-1:]

    tr_ds = SeqDataset(Xtr, yctr, yrtr); tr_ds.feat_cols = feat_cols
    v_ds  = SeqDataset(Xv,  ycv,  yrv ); v_ds.feat_cols  = feat_cols
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))

    # Use DistributedSampler if DDP
    train_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if args.ddp else None
    valid_sampler = DistributedSampler(v_ds, num_replicas=world_size, rank=rank, shuffle=False) if args.ddp else None

    tr_dl = DataLoader(tr_ds, batch_size=int(cfg.get("train", {}).get("batch_size", 256)),
                       shuffle=(train_sampler is None), sampler=train_sampler, drop_last=False, num_workers=num_workers, pin_memory=(device.startswith("cuda")))
    v_dl  = DataLoader(v_ds,  batch_size=int(cfg.get("train", {}).get("batch_size", 256)),
                       shuffle=False, sampler=valid_sampler, drop_last=False, num_workers=num_workers, pin_memory=(device.startswith("cuda")))

    # 构建模型
    if args.model == "lstm":
        net = MultiTaskLSTM(input_dim=Xtr.shape[2], hidden=256, num_layers=3, dropout=0.2)
    elif args.model == "transformer":
        net = MultiTaskTransformer(input_dim=Xtr.shape[2], d_model=256, nhead=8, num_layers=3)
    else:  # hybrid 大模型
        from src.models.hybrid_transformer import HybridTransformerMTL
        net = HybridTransformerMTL(input_dim=Xtr.shape[2], d_model=384, nhead=8, num_layers=6, dim_ff=768, dropout=0.1)

    net.to(device)  # Critical fix: move model to device

    if args.ddp:
        net = DDP(net, device_ids=[rank % torch.cuda.device_count()])

    # 训练
    prefix = f"{args.symbol.upper()}_{args.model}"
    ckpt, meta = train_loop(net, tr_dl, v_dl, cfg, device=device, amp=args.amp, out_dir=args.out, prefix=prefix, rank=rank)
    if rank == 0:
        print(f"[save] ckpt={ckpt}\n[save] meta={meta}")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()