# src/training/hpo_optuna.py
import os, json, argparse, optuna, numpy as np, pandas as pd, plotly
from sklearn.metrics import accuracy_score, mean_squared_error
import torch
from src.utils.config import load_config
from src.feature_engineering import compute_tech_indicators
from src.labeling import make_labels
from src.datasets import make_sequence_dataset
from src.models.hybrid_transformer import HybridTransformerMTL

def objective(trial, cfg, symbol):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [256, 384, 512])
    num_layers = trial.suggest_int("num_layers", 3, 8)
    nhead = trial.suggest_categorical("nhead", [4,8])
    dim_ff = trial.suggest_categorical("dim_ff", [512, 768, 1024])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    df = pd.read_csv(cfg['data']['input_csv'])
    df[cfg['data']['timestamp_col']] = pd.to_datetime(df[cfg['data']['timestamp_col']])
    df = df.rename(columns={cfg['data']['timestamp_col']:'timestamp', cfg['data']['ticker_col']:'ticker'})
    df = df[df['ticker']==symbol].sort_values(['ticker','timestamp'])

    df = compute_tech_indicators(df, cfg)
    df = make_labels(df, cfg)
    feature_cols = [c for c in df.columns if c not in ('timestamp','ticker','open','high','low','close','volume','fwd_close','y_reg','label_cls','label_tri','label_reg')]
    X, y = make_sequence_dataset(df, feature_cols, 'label_cls', 64)
    _, yr = make_sequence_dataset(df, feature_cols, 'y_reg', 64)
    # 简单划分：最后10% 作为验证
    n = X.shape[0]; k = max(1, int(n*0.1))
    Xtr, Xva = X[:-k], X[-k:]; ytr, yva = y[:-k], y[-k:]; yrtr, yrva = yr[:-k], yr[-k:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = HybridTransformerMTL(input_dim=X.shape[2], d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, dropout=dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    bs = 128; epochs = 6
    for ep in range(epochs):
        net.train()
        for i in range(0, Xtr.shape[0], bs):
            xb = torch.tensor(Xtr[i:i+bs], dtype=torch.float32, device=device)
            yb = torch.tensor(ytr[i:i+bs], dtype=torch.float32, device=device)
            ybr = torch.tensor(yrtr[i:i+bs], dtype=torch.float32, device=device)
            logit, reg = net(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb) + 0.5*torch.nn.functional.mse_loss(reg, ybr)
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        xb = torch.tensor(Xva, dtype=torch.float32, device=device)
        lg, rr = net(xb)
        prob = torch.sigmoid(lg).cpu().numpy()
        rmse = mean_squared_error(yrva, rr.cpu().numpy(), squared=False)
        acc = accuracy_score(yva, (prob>=0.5).astype(int))
    # maximize: acc - alpha*rmse
    score = float(acc - 0.1*rmse)
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="out/hpo")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, cfg, args.symbol), n_trials=args.trials)
    with open(os.path.join(args.out, f"{args.symbol}_best_params.json"),'w',encoding='utf-8') as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    # 可视化
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study); fig1.write_html(os.path.join(args.out, f"{args.symbol}_history.html"))
        fig2 = plot_param_importances(study); fig2.write_html(os.path.join(args.out, f"{args.symbol}_importance.html"))
    except Exception as e:
        print("plot fail:", e)

if __name__ == "__main__":
    main()
