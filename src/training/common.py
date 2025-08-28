import os, time, json, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from ..utils.logger import get_logger

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def setup_device(cfg):
    dev = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def train_torch_model(model, X_train, y_train, w_train, X_valid, y_valid, task="binary", cfg=None):
    logger = get_logger("train")
    device = setup_device(cfg)
    model.to(device)

    bs = int(cfg["train"].get("batch_size", 256))
    epochs = int(cfg["train"].get("epochs", 20))
    lr = float(cfg["train"].get("lr", 1e-3))
    wd = float(cfg["train"].get("l2", 0.0))

    #Xtr = torch.tensor(X_train, dtype=torch.float32); ytr = torch.tensor(y_train); wtr = torch.tensor(w_train, dtype=torch.float32)
    Xtr = torch.tensor(X_train, dtype=torch.float32); ytr = torch.tensor(y_train, dtype=torch.float32); wtr = torch.tensor(w_train, dtype=torch.float32)
    Xva = torch.tensor(X_valid, dtype=torch.float32); yva = torch.tensor(y_valid, dtype=torch.float32)

    train_ds = TensorDataset(Xtr, ytr, wtr)
    valid_ds = TensorDataset(Xva, yva)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(cfg["train"].get("num_workers", 0)))
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=int(cfg["train"].get("num_workers", 0)))
    
    if task == "binary":
        crit = nn.BCEWithLogitsLoss(reduction="none")
    else:
        crit = nn.MSELoss(reduction="none")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_metric, best_path = -1e9, os.path.join(cfg["train"]["out_dir"], "best.pt")
    patience = int(cfg["train"].get("early_stop_patience", 5))
    bad = 0
    
    for ep in range(1, epochs+1):
        model.train()
        loss_sum, w_sum = 0.0, 0.0
        '''
        # 限制线程避免 BLAS/OMP 线程干扰（调试时先开启）
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        it = iter(train_dl)
        print("DEBUG: about to fetch first batch via next(it)")
        t0 = time.time()
        try:
            xb, yb, wb = next(it)
            t1 = time.time()
            print(f"DEBUG: fetched batch in {t1-t0:.3f}s")
            print("xb shape, dtype, device:", getattr(xb, "shape", None), xb.dtype, xb.device)
            print("yb shape, dtype, device:", getattr(yb, "shape", None), yb.dtype, yb.device)
            print("wb shape, dtype, device:", getattr(wb, "shape", None), wb.dtype, wb.device)

            # 1) test .float() and .to(device) separately with prints
            t0 = time.time()
            xb2 = xb.to(device)
            t1 = time.time()
            print(f"DEBUG: xb.to(device) finished in {t1-t0:.3f}s, xb2 device: {xb2.device}")

            t0 = time.time()
            yb2 = yb.float().to(device)
            t1 = time.time()
            print(f"DEBUG: yb.float().to(device) finished in {t1-t0:.3f}s, yb2 device: {yb2.device}")

            t0 = time.time()
            wb2 = wb.to(device)
            t1 = time.time()
            print(f"DEBUG: wb.to(device) finished in {t1-t0:.3f}s, wb2 device: {wb2.device}")

            # 2) inspect where model lives
            try:
                p = next(model.parameters())
                print("model param device:", p.device)
            except StopIteration:
                print("DEBUG: model has no parameters (?)")

            # 3) do a single forward with timing and catch exceptions
            try:
                t0 = time.time()
                out = model(xb2)
                t1 = time.time()
                print(f"DEBUG: forward finished in {t1-t0:.3f}s, out shape: {getattr(out, 'shape', None)}")
            except Exception as e:
                print("ERROR during forward:", repr(e))
                raise

            # 4) compute loss (just to test)
            try:
                out_s = out.squeeze(-1)
                loss_term = crit(out_s, yb2 if task!="binary" else yb2) * wb2
                print("DEBUG: loss_term shape:", loss_term.shape)
                lmean = loss_term.mean()
                print("DEBUG: mean loss computed:", float(lmean))
            except Exception as e:
                print("ERROR during loss computation:", repr(e))
                raise

        except Exception as e:
            print("ERROR fetching first batch or iterating:", repr(e))
            raise
        '''
        for xb, yb, wb in train_dl:
            xb = xb.to(device); yb = yb.float().to(device); wb = wb.to(device)
            out = model(xb).squeeze(-1)
            loss = crit(out, yb if task!="binary" else yb) * wb
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            w_sum += len(xb)
        # validation
        model.eval()

        #original code
        with torch.no_grad():
            ov = model(Xva.to(device)).squeeze(-1).cpu().numpy()
        
        if task == "binary":
            prob = 1 / (1 + np.exp(-ov))
            pred = (prob >= 0.5).astype(int)
            metric = (pred == y_valid).mean()
        else:
            rmse = ((ov - y_valid) ** 2).mean() ** 0.5
            metric = -rmse  # 越大越好
        
        '''
        #修改后的代码：
        with torch.no_grad():
            ov_tensor = model(Xva.to(device)).squeeze(-1)      # tensor on device (cpu in your case)
            prob = torch.sigmoid(ov_tensor).cpu().numpy()      # numpy array of probabilities

        if task == "binary":
            pred = (prob >= 0.5).astype(int)
            # 强制把 y_valid 转成标准 numpy ndarray（避免自定义包装器的 __eq__ / __array_ufunc__ 等）
            yv = np.asarray(y_valid)
            # 使用 numpy 的等价函数并取 float 返回，避免调用可被覆盖的 .mean() 方法
            metric = float(np.mean(np.equal(pred, yv)))
        else:
            ov = ov_tensor.cpu().numpy().astype(np.float32)
            rmse = ((ov - y_valid) ** 2).mean() ** 0.5
            metric = -rmse
        '''

        logger.info(f"epoch {ep}: train_loss={loss_sum/max(w_sum,1):.6f} valid_metric={metric:.6f}")
        if metric > best_metric:
            best_metric, bad = metric, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                logger.info("early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model
