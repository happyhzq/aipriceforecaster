import os, numpy as np, pandas as pd, torch, argparse, json
from ..utils.config import load_config
from ..utils.logger import get_logger
from ..feature_engineering import compute_tech_indicators
from ..labeling import make_labels
from ..datasets import make_sequence_dataset
from ..models.multitask_lstm import MultiTaskLSTM
from ..models.multitask_transformer import MultiTaskTransformer
from ..models.xgb_multitask import XGBMulti
from sklearn.metrics import accuracy_score, mean_squared_error
import optuna

def set_seed(s=42):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)

def walk_forward_splits(df, n_splits=3, train_window_days=None):
    # simple time-based splits - user can refine to trading days
    df = df.sort_values('timestamp')
    times = df['timestamp'].unique()
    N = len(times)
    step = max(1, N//(n_splits+1))
    splits = []
    for i in range(1, n_splits+1):
        cut = times[i*step]
        splits.append(cut)
    return splits

def train_one(cfg, model_family='lstm'):
    logger = get_logger('train')
    cfg = cfg
    df = pd.read_csv(os.path.join(cfg['train']['out_dir'],'dataset.csv'), parse_dates=['timestamp'])
    feat_cols = [c for c in df.columns if c not in ("timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight","insert_time","update_time","interface_id","fetch_time")]
    target_col = "y_cls" if cfg["label"]["label_type"]=="binary" else ("y_reg" if cfg["label"]["label_type"]=="regression" else "y_tri")
    # simple single split
    split_time = cfg['data']['train_test_split_date']
    train = df[df['timestamp'] < split_time].copy()
    valid = df[df['timestamp'] >= split_time].copy()
    window = 32
    Xtr, ytr, wtr = make_sequence_dataset(train, feat_cols, target_col, window)
    _, ytr_reg, wtr_reg = make_sequence_dataset(train, feat_cols, 'y_reg', window)
    Xva, yva, wva = make_sequence_dataset(valid, feat_cols, target_col, window)
    _, yva_reg, wva_reg = make_sequence_dataset(valid, feat_cols, 'y_reg', window)
    device = torch.device(cfg['train'].get('device','cpu'))
    if model_family == 'xgboost':
        xgb = XGBMulti()
        xgb.fit(Xtr.reshape(Xtr.shape[0], -1), ytr, ytr_reg)
        pcls, preg = xgb.predict(Xva.reshape(Xva.shape[0], -1))
        logger.info('XGB cls acc: %.4f reg rmse: %.6f' % (accuracy_score(yva, (pcls>=0.5).astype(int)) if len(pcls)>0 else 0.0, mean_squared_error(yva_reg, preg, squared=False) if len(preg)>0 else 0.0))
        pred_df = valid[['timestamp','ticker']].tail(len(pcls)).copy()
        pred_df['pred'] = pcls; pred_df['pred_reg'] = preg
        pred_df.to_csv(os.path.join(cfg['train']['out_dir'],'preds.csv'), index=False)
        return
    # pytorch models
    if model_family == 'lstm':
        model = MultiTaskLSTM(input_dim=Xtr.shape[2], hidden=256, num_layers=3).to(device)
    else:
        model = MultiTaskTransformer(input_dim=Xtr.shape[2], d_model=256, nhead=8, num_layers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['train'].get('lr',1e-3)), weight_decay=1e-5)
    bs = int(cfg['train'].get('batch_size',64))
    epochs = int(cfg['train'].get('epochs',10))
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for i in range(0, Xtr.shape[0], bs):
            xb = torch.tensor(Xtr[i:i+bs], dtype=torch.float32).to(device)
            yb = torch.tensor(ytr[i:i+bs], dtype=torch.float32).to(device)
            yb_reg = torch.tensor(ytr_reg[i:i+bs], dtype=torch.float32).to(device)
            logit, reg = model(xb)
            loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)
            loss_reg = torch.nn.functional.mse_loss(reg, yb_reg)
            loss = loss_cls + 0.5 * loss_reg
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # validation
        model.eval()
        with torch.no_grad():
            ov = []
            orr = []
            for i in range(0, Xva.shape[0], bs):
                xb = torch.tensor(Xva[i:i+bs], dtype=torch.float32).to(device)
                l, r = model(xb)
                ov.append(l.cpu().numpy()); orr.append(r.cpu().numpy())
            ov = np.concatenate(ov) if len(ov)>0 else np.array([])
            orr = np.concatenate(orr) if len(orr)>0 else np.array([])
        if len(ov)>0:
            prob = 1/(1+np.exp(-ov))
            acc = accuracy_score(yva, (prob>=0.5).astype(int))
            rmse = mean_squared_error(yva_reg, orr)
            logger.info(f'ep {ep} train_loss {np.mean(losses):.6f} val_acc {acc:.4f} val_rmse {rmse:.6f}')
    torch.save(model.state_dict(), os.path.join(cfg['train']['out_dir'], f'{model_family}_best.pt'))
    prob = 1/(1+np.exp(-ov)) if len(ov)>0 else np.array([])
    pred_df = valid[['timestamp','ticker']].tail(len(prob)).copy()
    pred_df['pred'] = prob; pred_df['pred_reg'] = orr
    pred_df.to_csv(os.path.join(cfg['train']['out_dir'],'preds.csv'), index=False)

if __name__=='__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--model', default='lstm'); args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg['train'].get('seed',42)); train_one(cfg, args.model)