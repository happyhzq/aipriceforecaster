# src/training/train.py
import os, argparse, numpy as np, pandas as pd, torch
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.feature_engineering import compute_tech_indicators
from src.labeling import make_labels
from src.datasets import make_sequence_dataset
from sklearn.metrics import accuracy_score, mean_squared_error

logger = get_logger('train')

def train_single(cfg_path, model_family='lstm'):
    cfg = load_config(cfg_path)
    out = cfg['train']['out_dir']
    df = pd.read_csv(os.path.join(out, 'dataset.csv'), parse_dates=['timestamp'])
    feat_cols = [c for c in df.columns if c not in ("timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight","insert_time","update_time","interface_id","fetch_time")]
    split = cfg['data']['train_test_split_date']
    train = df[df['timestamp'] < split].copy()
    valid = df[df['timestamp'] >= split].copy()
    window = cfg.get('train', {}).get('window', 32)
    Xtr, ytr = make_sequence_dataset(train, feat_cols, 'label_cls', window)
    _, ytr_reg = make_sequence_dataset(train, feat_cols, 'y_reg', window)
    Xva, yva = make_sequence_dataset(valid, feat_cols, 'label_cls', window)
    _, yva_reg = make_sequence_dataset(valid, feat_cols, 'y_reg', window)
    device = torch.device(cfg['train'].get('device', 'cpu'))
    if model_family == 'lstm':
        from src.models.multitask_lstm import MultiTaskLSTM
        model = MultiTaskLSTM(input_dim=len(feat_cols), hidden=256, num_layers=3).to(device)
    else:
        from src.models.multitask_transformer_mixed import MultiTaskTransformerMixed
        model = MultiTaskTransformerMixed(input_dim=len(feat_cols), d_model=256, nhead=8, num_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['train'].get('lr', 1e-3)))
    bs = int(cfg['train'].get('batch_size', 64))
    epochs = int(cfg['train'].get('epochs', 5))
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for i in range(0, Xtr.shape[0], bs):
            xb = torch.tensor(Xtr[i:i+bs], dtype=torch.float32, device=device)
            yb = torch.tensor(ytr[i:i+bs], dtype=torch.float32, device=device)
            yb_reg = torch.tensor(ytr_reg[i:i+bs], dtype=torch.float32, device=device)
            logit, reg = model(xb)
            loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)
            loss_reg = torch.nn.functional.mse_loss(reg, yb_reg)
            loss = loss_cls + 0.5 * loss_reg
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # validate
        model.eval()
        with torch.no_grad():
            if Xva.shape[0] > 0:
                ov, orr = [], []
                for i in range(0, Xva.shape[0], bs):
                    xb = torch.tensor(Xva[i:i+bs], dtype=torch.float32, device=device)
                    l, r = model(xb)
                    ov.append(l.cpu().numpy()); orr.append(r.cpu().numpy())
                ov = np.concatenate(ov); orr = np.concatenate(orr)
                prob = 1/(1+np.exp(-ov))
                acc = accuracy_score(yva, (prob>=0.5).astype(int))
                rmse = mean_squared_error(yva_reg, orr, squared=False)
            else:
                acc = 0.0; rmse = 0.0
        logger.info(f"ep {ep} loss {np.mean(losses):.6f} val_acc {acc:.4f} val_rmse {rmse:.6f}")
    # save
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{model_family}_best.pt")
    torch.save(model.state_dict(), path)
    logger.info("saved model to %s", path)
