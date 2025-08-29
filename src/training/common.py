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
    
    '''
    #单元测试代码：
    # 在 model.to(device) 后加入检查
    print("Model device after to:", next(model.parameters()).device)
    # 快速单次前向样例检查（防止第一步就卡）
    with torch.no_grad():
        small_x = Xtr[:min(8, len(Xtr))]
        try:
            _ = model(small_x.to(next(model.parameters()).device))
            print("quick forward check passed")
        except Exception as e:
            print("quick forward check failed:", repr(e))
            raise
    '''

    for ep in range(1, epochs+1):
        model.train()
        loss_sum, w_sum = 0.0, 0.0

        for xb, yb, wb in train_dl:
            xb = xb.to(device); yb = yb.float().to(device); wb = wb.to(device)
            out = model(xb).squeeze(-1)
            loss = crit(out, yb if task!="binary" else yb) * wb
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            '''
            #测试代码：检查损失/梯度是否在更新（可在训练循环打印梯度范数）
            #在训练循环里 loss.backward() 之后、opt.step() 之前插入：
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is None: continue
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"grad_norm: {total_norm:.6f}")
            # 也可打印参数范数
            param_norm = 0.0
            for p in model.parameters():
                param_norm += float(p.data.norm(2).item()**2)
            print("param_norm:", param_norm**0.5)
            #测试代码结束
            '''
            opt.step()
            loss_sum += loss.item() * len(xb)
            w_sum += len(xb)
        # validation
        model.eval()

        '''
        #original code
        #原始代码报错np.exp
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
        #newcode:
        all_preds = []
        with torch.no_grad():
            for xb_v, _ in valid_dl:
                xb_v = xb_v.to(device)
                out_v = model(xb_v).squeeze(-1)
                if task == "binary":
                    # compute probabilities with torch.sigmoid (safe)
                    probs_v = torch.sigmoid(out_v)
                    all_preds.append(probs_v.cpu().numpy())
                else:
                    all_preds.append(out_v.cpu().numpy())
        if len(all_preds) > 0:
            ov = np.concatenate(all_preds, axis=0)
            '''
            #单元测试代码：
            #测试代码：验证预测是否恒定（在 train_torch_model 验证后打印 ov 分布）在 train_torch_model 的验证段（我们之前改好的那段）末尾，加入：
            # after ov = np.concatenate(all_preds, axis=0)
            print("VALIDATION: ov shape", ov.shape)
            print("ov sample:", ov[:10])
            print("ov mean/std:", np.nanmean(ov), np.nanstd(ov))
            # 如果二分类，输出 pred distribution:
            if task == "binary":
                preds = (ov >= 0.5).astype(int)
                vals, cnts = np.unique(preds, return_counts=True)
                print("pred dist:", dict(zip(vals.tolist(), cnts.tolist())))
                # also compare to y_valid
                yv = np.asarray(y_valid).ravel()
                if len(yv) == len(preds):
                    print("accuracy check:", (preds == yv).mean())
                else:
                    print("WARNING: y_valid length mismatch", len(yv), len(preds))
            '''

        else:
            ov = np.array([])

        if task == "binary":
            prob = ov  # ov already probabilities in [0,1]
            pred = (prob >= 0.5).astype(int)
            metric = (pred == y_valid).mean()
        else:
            # ov are raw predictions
            rmse = ((ov - y_valid) ** 2).mean() ** 0.5
            metric = -rmse  # 越大越好

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
