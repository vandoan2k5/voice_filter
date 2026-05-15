"""Overfit test: 1 batch, 1 GPU, batch_size=2"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys, numpy as np, torch, toml
from src.dataset_libri import make_dataloader
from src.trainer import SiSnrTrainer, get_mix_stft, load_obj
from src.utils import cal_sisnr, cal_sdr
from model import pTFGridNet

configs = toml.load("configs/train_libri_config.toml")
configs['gpu']['gpu_ids'] = [0]
configs['dataloader']['batch_size'] = 2

device = torch.device("cuda:0")

train_loader = make_dataloader(train=True, data_dirs=configs['path']['train']['data_dirs'],
    chunk_length=3, sample_rate=16000, batch_size=2, pin_memory=True, num_workers=0)

batch = next(iter(train_loader))
mix_np = batch['mix'].numpy()
ref_np = batch['ref'].numpy()
aux_lens = batch['aux_len'].tolist()
print(f"aux unique: {len(set(aux_lens))}/2")

net = pTFGridNet(n_fft=configs['signal']['fft_num'], n_layers=3,
                 lstm_hidden_units=256, attn_n_head=4, attn_approx_qk_dim=512,
                 emb_dim=64, emb_ks=4, emb_hs=2, activation='prelu', eps=1e-5)

trainer = SiSnrTrainer(net, gpuid=(0,), configs=configs)

unpro_sisnr = np.mean([cal_sisnr(i, mix_np, ref_np, batch['valid_len']) for i in range(2)])
print(f"Baseline SI-SDR(mix,ref) = {unpro_sisnr:+.2f} dB")

batch_cuda = load_obj(batch, device)
batch_cuda = get_mix_stft(batch_cuda, trainer.configs_signal, device)

trainer.net.train()
optimizer = trainer.optimizer

print(f"\n{'step':>5s} {'loss':>8s} {'pro SI-SDR':>11s} {'delta':>8s} {'pro SDR':>8s}")
for step in range(1, 301):
    optimizer.zero_grad()
    batch_est_wav, loss = trainer.compute_loss(batch_cuda)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()

    if step % 30 == 0 or step == 1:
        with torch.no_grad():
            est_np = batch_est_wav.detach().cpu().numpy()
            pro_sisnr = np.mean([cal_sisnr(i, est_np, ref_np, batch['valid_len']) for i in range(2)])
            pro_sdr = np.mean([cal_sdr(i, est_np, ref_np, batch['valid_len']) for i in range(2)])
            print(f" {step:>5d} {loss.item():>+8.2f} {pro_sisnr:>+11.2f} {pro_sisnr - unpro_sisnr:>+8.2f} {pro_sdr:>+8.2f}")

print("\nFusion alpha:")
for i, fb in enumerate(trainer.net.fusion_blocks):
    a = fb.alpha.detach().cpu().numpy().flatten()
    print(f"  block {i}: mean={a.mean():.6f}, max={a.max():.6f}")

# Correlation check
with torch.no_grad():
    est, _ = trainer.compute_loss(batch_cuda)
    est_np = est.cpu().numpy()[0, :batch['valid_len'][0].item()]
    mx = mix_np[0, :batch['valid_len'][0].item()]
    rf = ref_np[0, :batch['valid_len'][0].item()]
    corr_est_mix = np.corrcoef(est_np, mx)[0,1]
    corr_est_ref = np.corrcoef(est_np, rf)[0,1]
    print(f"\ncorr(output,mix)={corr_est_mix:+.4f}  corr(output,ref)={corr_est_ref:+.4f}")
