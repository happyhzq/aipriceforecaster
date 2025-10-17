# src/training/optuna_search.py
import optuna, os, json
from src.utils.config import load_config
from src.utils.logger import get_logger
logger = get_logger('optuna')

def objective(trial, cfg):
    # sample hyperparams
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    nlayer = trial.suggest_int('nlayer', 2, 6)
    # integrate these into training flow & return val metric (e.g. accuracy)
    # For brevity, call training function with these params (user should implement train_with_params)
    from src.training.train import train_single
    # train_single currently doesn't accept dynamic params; in practice you'd factor training into train_with_params.
    # Here is just a skeleton showing how you'd call it.
    val_metric = 0.0
    # TODO: implement train_with_params to return validation metric
    return val_metric

def run_optuna(cfg_path, n_trials=20):
    cfg = load_config(cfg_path)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, cfg), n_trials=n_trials)
    logger.info("Best params: %s", study.best_params)
    # save
    out = cfg['train']['out_dir']
    with open(os.path.join(out, 'optuna_best.json'), 'w') as f:
        json.dump(study.best_params, f)
