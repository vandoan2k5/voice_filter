import os
import glob
import time
import torch
import random
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio


def worker_init_fn(worker_id):
    np.random.seed(int(time.time() * 1000) % (2**31) + worker_id)
    random.seed(int(time.time() * 1000) % (2**31) + worker_id)


def create_online_dataloader(hp, args, train):
    def train_collate_fn(batch):
        dvec_wav_list = list()
        target_mag_list = list()
        mixed_mag_list = list()

        for dvec_wav, target_mag, mixed_mag in batch:
            dvec_wav_list.append(dvec_wav)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

        return dvec_wav_list, target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(
            dataset=OnlineVFDataset(hp, args, train=True),
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            collate_fn=train_collate_fn,
            pin_memory=True,
            drop_last=True,
            sampler=None,
            worker_init_fn=worker_init_fn,
        )
    else:
        return DataLoader(
            dataset=OnlineVFDataset(hp, args, train=False),
            collate_fn=test_collate_fn,
            batch_size=1, shuffle=False, num_workers=0
        )


class OnlineVFDataset(Dataset):
    """
    On-the-fly dataset that reads raw FLAC files directly from LibriSpeech
    directory structure. No intermediate files are saved to disk.

    Processing pipeline (mirrors generator.py logic):
      1. Load raw FLAC
      2. Trim silence (librosa.effects.trim, top_db=20)
      3. Mix target + interference
      4. Peak normalize to prevent clipping
      5. Compute magnitude spectrogram via STFT
    """
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.srate = hp.audio.sample_rate
        self.audio_len_samples = int(self.srate * hp.data.audio_len)
        self.min_dvec_len = int(1.1 * hp.embedder.window * hp.audio.hop_length)

        data_dirs = hp.data.raw_train_dirs if train else hp.data.raw_test_dirs

        self.speakers = []
        for data_dir in data_dirs:
            speaker_folders = sorted([
                x for x in glob.glob(os.path.join(data_dir, '*', '*'))
                if os.path.isdir(x)
            ])
            for spk_dir in speaker_folders:
                flac_files = sorted(glob.glob(os.path.join(spk_dir, '*.flac')))
                if len(flac_files) >= 2:
                    self.speakers.append(flac_files)

        if len(self.speakers) < 2:
            raise RuntimeError(
                "Need at least 2 speakers with >=2 utterances each. "
                f"Found {len(self.speakers)} speakers."
            )

        self.num_samples = hp.data.num_train_samples if train else hp.data.num_test_samples

        self.samples = []
        rng = random.Random(42)
        for i in range(self.num_samples):
            spk1_idx = rng.randint(0, len(self.speakers) - 1)
            spk2_idx = rng.randint(0, len(self.speakers) - 1)
            while spk2_idx == spk1_idx:
                spk2_idx = rng.randint(0, len(self.speakers) - 1)

            spk1_files = self.speakers[spk1_idx]
            spk2_files = self.speakers[spk2_idx]

            s1_dvec, s1_target = rng.sample(spk1_files, 2)
            s2 = rng.choice(spk2_files)

            self.samples.append((s1_dvec, s1_target, s2))

        self.audio = Audio(hp)
        self._fallback_idx = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retries = 20
        for attempt in range(max_retries):
            try:
                return self._load_sample(idx)
            except Exception:
                idx = (idx + self._fallback_idx + attempt) % len(self.samples)
        self._fallback_idx = (self._fallback_idx + 1) % len(self.samples)
        return self._load_sample(0)

    def _load_sample(self, idx):
        dvec_path, s1_target_path, s2_path = self.samples[idx]

        d, _ = librosa.load(dvec_path, sr=self.srate)
        w1, _ = librosa.load(s1_target_path, sr=self.srate)
        w2, _ = librosa.load(s2_path, sr=self.srate)

        d, _ = librosa.effects.trim(d, top_db=20)
        w1, _ = librosa.effects.trim(w1, top_db=20)
        w2, _ = librosa.effects.trim(w2, top_db=20)

        if d.shape[0] < self.min_dvec_len:
            raise RuntimeError("dvec too short")

        if w1.shape[0] < self.audio_len_samples or w2.shape[0] < self.audio_len_samples:
            raise RuntimeError("audio too short")

        # Random start offset for variety
        if w1.shape[0] > self.audio_len_samples:
            start = random.randint(0, w1.shape[0] - self.audio_len_samples)
            w1 = w1[start:start + self.audio_len_samples]
        else:
            w1 = w1[:self.audio_len_samples]

        if w2.shape[0] > self.audio_len_samples:
            start = random.randint(0, w2.shape[0] - self.audio_len_samples)
            w2 = w2[start:start + self.audio_len_samples]
        else:
            w2 = w2[:self.audio_len_samples]

        mixed = w1 + w2

        norm = np.max(np.abs(mixed)) * 1.1
        if norm < 1e-8:
            raise RuntimeError("silent mixed")
        w1 = w1 / norm
        mixed = mixed / norm

        if self.train:
            target_mag, _ = self.audio.wav2spec(w1)
            mixed_mag, _ = self.audio.wav2spec(mixed)
            target_mag = torch.from_numpy(target_mag).float()
            mixed_mag = torch.from_numpy(mixed_mag).float()
            return d, target_mag, mixed_mag
        else:
            target_mag, _ = self.audio.wav2spec(w1)
            mixed_mag, mixed_phase = self.audio.wav2spec(mixed)
            target_mag = torch.from_numpy(target_mag).float()
            mixed_mag = torch.from_numpy(mixed_mag).float()
            return d, w1, mixed, target_mag, mixed_mag, mixed_phase
