# src/training/ddp_train.py
"""
DDP training script.
Run with:
  torchrun --nproc_per_node=4 src/training/ddp_train.py --config configs/short.yaml --model transformer
"""
import os, argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("ddp_train")

def setup(rank, world_size, master_addr="127.0.0.1", master_port="12355"):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def ddp_worker(rank, world_size, args):
    setup(rank, world_size)
    cfg = load_config(args.config)
    # build dataset and model like in training.train_single but use DistributedSampler
    import pandas as pd, numpy as np
    from src.datasets import make_sequence_dataset
    from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
    out = cfg['train']['out_dir']
    df = pd.read_csv(os.path.join(out, 'dataset.csv'), parse_dates=['timestamp'])
    feat_cols = [c for c in df.columns if c not in ('timestamp','ticker','open','high','low','close','volume','fwd_close','y_reg','label_cls')]
    split = cfg['data']['train_test_split_date']
    train_df = df[df['timestamp'] < split].copy()
    window = cfg['train'].get('window', 32)
    Xtr, ytr = make_sequence_dataset(train_df, feat_cols, 'label_cls', window)
    # convert to tensors and sampler
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    ds = TensorDataset(Xtr_t, ytr_t)
    sampler = DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=cfg['train'].get('batch_size', 64), sampler=sampler)
    # model
    device = torch.device(f"cuda:{rank}")
    if args.model == 'transformer':
        from src.models.multitask_transformer_mixed import MultiTaskTransformerMixed as ModelClass
        model = ModelClass(input_dim=len(feat_cols), d_model=256, nhead=8, num_layers=4).to(device)
    else:
        from src.models.multitask_lstm import MultiTaskLSTM as ModelClass
        model = ModelClass(input_dim=len(feat_cols), hidden=256, num_layers=3).to(device)
    model = torch.nn.parallel.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['train'].get('lr', 1e-3)))
    # train loop
    for ep in range(int(cfg['train'].get('epochs', 5))):
        sampler.set_epoch(ep)
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            logit, reg = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if rank == 0:
            logger.info(f"ep {ep} done")
    # save from rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(out, f"{args.model}_ddp_best.pt"))
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', default='transformer')
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    # torchrun will spawn processes; here use rank from env if present
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        ddp_worker(rank, world_size, args)
    else:
        mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
