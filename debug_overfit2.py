"""Quick overfit test AFTER dataset fix: can model learn on 1 diverse batch?"""
import sys
import numpy as np
import torch
import toml

from src.dataset_libri import make_dataloader
from src.trainer import SiSnrTrainer, get_mix_stft, load_obj
from src.utils import cal_sisnr, cal_sdr, get_logger
from model import pTFGridNet

configs = toml.load("configs/train_libri_config.toml")

train_loader = make_dataloader(
    train=True,
    data_dirs=configs['path']['train']['data_dirs'],
    chunk_length=configs['signal']['chunk_length'],
    sample_rate=configs['signal']['sr'],
    batch_size=configs['dataloader']['batch_size'],
    pin_memory=configs['dataloader']['pin_memory'],
    num_workers=0,
)
batch = next(iter(train_loader))

mix_np = batch['mix'].numpy()
ref_np = batch['ref'].numpy()
aux_lens = batch['aux_len'].tolist()

# Check aux diversity
uniq = len(set(aux_lens))
print(f"aux_len unique: {uniq}/4")
if uniq < 2:
    print("SKIP: still got non-diverse batch, try again")
    sys.exit(1)

gpuids = tuple(configs['gpu']['gpu_ids'])
device = torch.device(f"cuda:{gpuids[0]}")
net = pTFGridNet(n_fft=configs['signal']['fft_num'], n_layers=configs['net']['n_layers'],
                 lstm_hidden_units=configs['net']['lstm_hidden_units'],
                 attn_n_head=configs['net']['attn_n_head'],
                 attn_approx_qk_dim=configs['net']['attn_approx_qk_dim'],
                 emb_dim=configs['net']['emb_dim'], emb_ks=configs['net']['emb_ks'],
                 emb_hs=configs['net']['emb_hs'], activation=configs['net']['activation'],
                 eps=configs['net']['eps'])

trainer = SiSnrTrainer(net, gpuid=gpuids, configs=configs)

# Baseline
unpro_sisnr = np.mean([cal_sisnr(i, mix_np, ref_np, batch['valid_len']) for i in range(4)])
print(f"\nBaseline SI-SDR(mix,ref) = {unpro_sisnr:+.2f} dB")

# Overfit for 500 steps
batch_cuda = load_obj(batch, device)
batch_cuda = get_mix_stft(batch_cuda, trainer.configs_signal, device)

trainer.net.train()
optimizer = trainer.optimizer

print(f"\n{'step':>5s} {'loss':>8s} {'SI-SDR(pro)':>12s} {'delta':>8s}")
best_sisnr = unpro_sisnr
for step in range(1, 501):
    optimizer.zero_grad()
    batch_est_wav, loss = trainer.compute_loss(batch_cuda)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), configs['optimizer']['gradient_norm'])
    optimizer.step()

    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            est_np = batch_est_wav.detach().cpu().numpy()
            pro_sisnr = np.mean([cal_sisnr(i, est_np, ref_np, batch['valid_len']) for i in range(4)])
            delta = pro_sisnr - unpro_sisnr
            print(f" {step:>5d} {loss.item():>+8.2f} {pro_sisnr:>+12.2f} {delta:>+8.2f}")
            if pro_sisnr > best_sisnr:
                best_sisnr = pro_sisnr

# Check alpha
print(f"\nBest SI-SDR: {best_sisnr:+.2f}")
print("Fusion alpha after 500 steps:")
for i, fb in enumerate(trainer.net.fusion_blocks):
    alpha = fb.alpha.detach().cpu().numpy().flatten()
    print(f"  block {i}: mean={alpha.mean():.6f}, max={alpha.max():.6f}")
