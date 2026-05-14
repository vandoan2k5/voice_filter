"""Test overfit on 1 batch + speaker embedding analysis."""
import sys
import numpy as np
import torch
import toml

from src.dataset_libri import make_dataloader
from src.trainer import SiSnrTrainer, get_mix_stft, load_obj
from src.utils import cal_sisnr, cal_sdr, get_logger
from model import pTFGridNet

configs = toml.load("configs/train_libri_config.toml")

logger = get_logger(__name__)

# ── Create dataloader and grab 1 batch ──
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
print(f"\nBatch keys: {batch.keys()}")
print(f"mix shape: {batch['mix'].shape}")
print(f"ref shape: {batch['ref'].shape}")
print(f"aux shape: {batch['aux'].shape}")
print(f"valid_len: {batch['valid_len']}")

# ── Speaker embedding analysis ──
gpuids = tuple(configs['gpu']['gpu_ids'])
device = torch.device(f"cuda:{gpuids[0]}")

net = pTFGridNet(n_fft=configs['signal']['fft_num'],
                 n_layers=configs['net']['n_layers'],
                 lstm_hidden_units=configs['net']['lstm_hidden_units'],
                 attn_n_head=configs['net']['attn_n_head'],
                 attn_approx_qk_dim=configs['net']['attn_approx_qk_dim'],
                 emb_dim=configs['net']['emb_dim'],
                 emb_ks=configs['net']['emb_ks'],
                 emb_hs=configs['net']['emb_hs'],
                 activation=configs['net']['activation'],
                 eps=configs['net']['eps'])

trainer = SiSnrTrainer(net, gpuid=gpuids, configs=configs)

# Extract speaker embeddings for the batch and compare
with torch.no_grad():
    aux_t = batch['aux'].to(device)
    spk_emb = trainer.spk_classifier.encode_batch(aux_t).squeeze(1).cpu().numpy()
    
    cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    print("\n=== SPEAKER EMBEDDING ANALYSIS ===")
    for i in range(len(spk_emb)):
        for j in range(i+1, len(spk_emb)):
            sim = cos_sim(spk_emb[i], spk_emb[j])
            print(f"  cos_sim(spk_emb[{i}], spk_emb[{j}]) = {sim:+.4f}")
    
    print(f"\n  spk_emb std per dim: min={spk_emb.std(axis=0).min():.4f}, "
          f"max={spk_emb.std(axis=0).max():.4f}, "
          f"mean={spk_emb.std(axis=0).mean():.4f}")
    print(f"  spk_emb magnitude per sample: {[f'{np.linalg.norm(spk_emb[i]):.2f}' for i in range(len(spk_emb))]}")

# ── Compute baseline metrics ──
mix_wav = mix_np[0, :batch['valid_len'][0]]
ref_wav = ref_np[0, :batch['valid_len'][0]]
print(f"\n=== BASELINE ===")
print(f"  SI-SDR(mix, ref) = {cal_sisnr(0, mix_np, ref_np, batch['valid_len']):+.2f} dB")
print(f"  SDR(mix, ref) = {cal_sdr(0, mix_np, ref_np, batch['valid_len']):+.2f} dB")
print(f"  mix std = {np.std(mix_wav):.4f}, ref std = {np.std(ref_wav):.4f}")

# ── Overfit test: train on same batch for 100 steps ──
print("\n=== OVERFIT TEST (100 steps on same batch) ===")
print(f"{'step':>5s} {'loss':>8s} {'SI-SDR(pro)':>12s} {'SI-SDR(i)':>12s} {'SDR(i)':>10s}")

trainer.net.train()
optimizer = trainer.optimizer

batch_cuda = load_obj(batch, device)
batch_cuda = get_mix_stft(batch_cuda, trainer.configs_signal, device)

for step in range(1, 101):
    optimizer.zero_grad()
    batch_est_wav, loss = trainer.compute_loss(batch_cuda)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), configs['optimizer']['gradient_norm'])
    optimizer.step()

    if step % 10 == 0 or step == 1:
        with torch.no_grad():
            est_np = batch_est_wav.detach().cpu().numpy()
            pro_sisnr = np.mean([cal_sisnr(i, est_np, ref_np, batch['valid_len']) for i in range(min(2, len(batch['valid_len'])))])
            unpro_sisnr = np.mean([cal_sisnr(i, mix_np, ref_np, batch['valid_len']) for i in range(min(2, len(batch['valid_len'])))])
            pro_sdr = np.mean([cal_sdr(i, est_np, ref_np, batch['valid_len']) for i in range(min(2, len(batch['valid_len'])))])
            unpro_sdr = np.mean([cal_sdr(i, mix_np, ref_np, batch['valid_len']) for i in range(min(2, len(batch['valid_len'])))])
            print(f" {step:>5d} {loss.item():>+8.2f} {pro_sisnr:>+12.2f} {pro_sisnr - unpro_sisnr:>+12.2f} {pro_sdr - unpro_sdr:>+10.2f}")

# ── Check FusionModule alpha values ──
print("\n=== FUSION MODULE ALPHA VALUES ===")
for i, fb in enumerate(trainer.net.fusion_blocks):
    alpha = fb.alpha.detach().cpu().numpy().flatten()
    print(f"  block {i}: alpha min={alpha.min():.6f}, max={alpha.max():.6f}, mean={alpha.mean():.6f}")

# ── Check correlation: output vs mix, output vs ref ──
with torch.no_grad():
    batch_est_wav, _ = trainer.compute_loss(batch_cuda)
    est_np = batch_est_wav.cpu().numpy()[0, :batch['valid_len'][0].item()]
    mix_w = mix_np[0, :batch['valid_len'][0].item()]
    ref_w = ref_np[0, :batch['valid_len'][0].item()]
    
    corr_est_mix = np.corrcoef(est_np, mix_w)[0, 1]
    corr_est_ref = np.corrcoef(est_np, ref_w)[0, 1]
    print(f"\n=== OUTPUT CORRELATION (after 100 steps) ===")
    print(f"  corr(output, mix) = {corr_est_mix:+.4f}")
    print(f"  corr(output, ref) = {corr_est_ref:+.4f}")
    print(f"  → model output is {'dominantly the MIX' if corr_est_mix > corr_est_ref else 'closer to REF'}")

print("\nDone.")
