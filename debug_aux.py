"""Check: are aux signals identical? + ECAPA embedding sanity check."""
import sys
import numpy as np
import torch
import toml
import soundfile as sf

from src.dataset_libri import make_dataloader

configs = toml.load("configs/train_libri_config.toml")
device = torch.device(f"cuda:{configs['gpu']['gpu_ids'][0]}")

train_loader = make_dataloader(
    train=True,
    data_dirs=configs['path']['train']['data_dirs'],
    chunk_length=configs['signal']['chunk_length'],
    sample_rate=configs['signal']['sr'],
    batch_size=4,
    pin_memory=configs['dataloader']['pin_memory'],
    num_workers=0,
)
batch = next(iter(train_loader))

# ── PART 1: Check if aux signals are actually different ──
print("=== AUX SIGNAL DIVERSITY ===")
aux = batch['aux'].numpy()
for i in range(4):
    for j in range(i+1, 4):
        li, lj = batch['aux_len'][i].item(), batch['aux_len'][j].item()
        min_len = min(li, lj)
        diff = np.max(np.abs(aux[i, :min_len] - aux[j, :min_len]))
        corr = np.corrcoef(aux[i, :min_len], aux[j, :min_len])[0, 1]
        print(f"  aux[{i}] vs aux[{j}]: max_diff={diff:.4f}, corr={corr:+.4f}  (lens: {li}, {lj})")

# ── PART 2: Quick ECAPA sanity check (different audio files) ──
print("\n=== ECAPA SANITY CHECK ===")
from speechbrain.inference.speaker import EncoderClassifier
ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": str(device)}
)

# Load 3 different WAV files and test
import glob, random
files = sorted(glob.glob("LibriSpeech/dev-clean/*/*/*.flac"))
if len(files) >= 3:
    test_files = random.sample(files, 3)
    wavs = []
    for f in test_files:
        data, sr = sf.read(f)
        if sr != 16000:
            import torchaudioudio
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            data = torch.from_numpy(data).unsqueeze(0)
            data = resampler(data).squeeze(0).numpy()
        wavs.append(data.astype(np.float32))
    
    # Pad to same length
    max_len = max(len(w) for w in wavs)
    wavs_padded = np.stack([np.pad(w, (0, max_len - len(w))) for w in wavs], axis=0)
    wav_t = torch.from_numpy(wavs_padded).to(device)
    
    with torch.no_grad():
        embs = ecapa.encode_batch(wav_t).squeeze(1).cpu().numpy()
    
    for i in range(3):
        for j in range(i+1, 3):
            sim = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8)
            print(f"  file[{i}] vs file[{j}]: cos_sim={sim:+.4f}, "
                  f"same speaker={os.path.dirname(os.path.dirname(test_files[i])) == os.path.dirname(os.path.dirname(test_files[j]))}")

# ── PART 3: Check dataset aux embedding again ──
print("\n=== DATASET AUX EMBEDDING (re-check) ===")
aux_t = batch['aux'].to(device)
with torch.no_grad():
    embs = ecapa.encode_batch(aux_t).squeeze(1).cpu().numpy()

for i in range(4):
    for j in range(i+1, 4):
        sim = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8)
        print(f"  emb[{i}] vs emb[{j}]: cos_sim={sim:+.4f}")

print(f"\n  emb std per dim: {embs.std(axis=0).mean():.6f}")

# ── PART 4: Test with shorter aux (first 3s only) ──
print("\n=== ECAPA on shorter aux (truncated to 3s) ===")
short_aux = batch['aux'][:, :48000].to(device)
with torch.no_grad():
    short_embs = ecapa.encode_batch(short_aux).squeeze(1).cpu().numpy()
for i in range(4):
    for j in range(i+1, 4):
        sim = np.dot(short_embs[i], short_embs[j]) / (np.linalg.norm(short_embs[i]) * np.linalg.norm(short_embs[j]) + 1e-8)
        print(f"  short_emb[{i}] vs short_emb[{j}]: cos_sim={sim:+.4f}")
