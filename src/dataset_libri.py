import os
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
from collections import defaultdict
from torch.utils.data.dataloader import default_collate


def make_dataloader(train=True,
                    data_dirs=None,
                    num_workers=4,
                    chunk_length=4,
                    sample_rate=8000,
                    pin_memory=True,
                    batch_size=16):
    chunk_size = sample_rate * chunk_length

    dataset = LibriSpeechDataset(data_dirs=data_dirs, sample_rate=sample_rate)

    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      pin_memory=pin_memory,
                      num_workers=num_workers)


class LibriSpeechDataset:
    def __init__(self,
                 data_dirs,
                 sample_rate=8000):
        self.sample_rate = sample_rate
        self.speaker_utts = defaultdict(list)
        self.min_dur = int(sample_rate * 2)

        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            for speaker_dir in sorted(os.listdir(data_dir)):
                speaker_path = os.path.join(data_dir, speaker_dir)
                if not os.path.isdir(speaker_path):
                    continue
                for chapter_dir in sorted(os.listdir(speaker_path)):
                    chapter_path = os.path.join(speaker_path, chapter_dir)
                    if not os.path.isdir(chapter_path):
                        continue
                    for f in sorted(os.listdir(chapter_path)):
                        if f.endswith('.flac'):
                            fpath = os.path.join(chapter_path, f)
                            try:
                                info = sf.info(fpath)
                                orig_sr = info.samplerate
                                num_frames = info.frames
                                dur = int(num_frames * self.sample_rate / orig_sr)
                                if dur >= self.min_dur:
                                    self.speaker_utts[speaker_dir].append((fpath, dur))
                            except Exception:
                                continue

        self.speakers = [s for s in self.speaker_utts if len(self.speaker_utts[s]) >= 2]
        if not self.speakers:
            raise RuntimeError("No valid speakers found in data_dirs: {}".format(data_dirs))
        self._total_utts = sum(len(v) for v in self.speaker_utts.values())

    def _get_resampler(self, orig_sr):
        return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)

    def _load_audio(self, path):
        data, sr = sf.read(path)
        data = data.astype(np.float32)
        if sr != self.sample_rate:
            resampler = self._get_resampler(sr)
            data = torch.from_numpy(data).unsqueeze(0)
            data = resampler(data).squeeze(0).numpy()
        return data

    def __len__(self):
        return min(self._total_utts, 5000)

    def __getitem__(self, index):
        spk_a = random.choice(self.speakers)
        spk_b = random.choice([s for s in self.speakers if s != spk_a])

        utts_a = self.speaker_utts[spk_a]
        utts_b = self.speaker_utts[spk_b]

        aux_path, aux_dur = random.choice(utts_a)

        ref_candidates = [(p, d) for p, d in utts_a if p != aux_path]
        if not ref_candidates:
            ref_candidates = utts_a
        ref_path, ref_dur = random.choice(ref_candidates)

        interf_path, interf_dur = random.choice(utts_b)

        dur = min(ref_dur, interf_dur)

        aux = self._load_audio(aux_path)
        ref = self._load_audio(ref_path)[:dur].astype(np.float32)
        interf = self._load_audio(interf_path)[:dur].astype(np.float32)

        min_len = min(len(ref), len(interf))
        ref = ref[:min_len]
        interf = interf[:min_len]
        dur = min_len

        snr = random.uniform(0, 5)
        ref_power = np.mean(ref ** 2) + 1e-8
        interf_power = np.mean(interf ** 2) + 1e-8
        scale = np.sqrt(ref_power / (interf_power * (10 ** (snr / 10)) + 1e-8))
        mix = ref + scale * interf

        mix_std = np.std(mix)
        aux_std = np.std(aux)
        mix = mix / mix_std
        aux = aux / aux_std

        return {"mix": mix.astype(np.float32),
                "ref": ref.astype(np.float32),
                "aux": aux.astype(np.float32),
                "aux_len": len(aux),
                "mix_scale": np.float32(mix_std)}


class ChunkSplitter:
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = eg["aux_len"]
        chunk["valid_len"] = int(self.chunk_size)
        chunk["mix_scale"] = eg["mix_scale"]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        if N < self.least:
            return []
        chunks = []
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["valid_len"] = int(N)
            chunk["mix_scale"] = eg["mix_scale"]
            chunks.append(chunk)
        else:
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader:
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True,
                 pin_memory=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                       train=train,
                                       least=chunk_size // 2)
        self.eg_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers,
                                                       shuffle=train,
                                                       pin_memory=pin_memory,
                                                       collate_fn=self._collate)

    def _collate(self, batch):
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _pad_aux(self, chunk_list):
        lens_list = []
        for chunk_item in chunk_list:
            lens_list.append(chunk_item['aux_len'])
        max_len = np.max(lens_list)
        for idx in range(len(chunk_list)):
            P = max_len - len(chunk_list[idx]["aux"])
            chunk_list[idx]["aux"] = np.pad(chunk_list[idx]["aux"], (0, P), "constant")
        return chunk_list

    def _merge(self, chunk_list):
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)

        blist = []
        i = 0
        while i + self.batch_size <= len(chunk_list):
            batch = chunk_list[i:i + self.batch_size]
            aux_lens = [item['aux_len'] for item in batch]
            min_unique = max(2, len(batch) // 2)
            if len(set(aux_lens)) < min_unique:
                for attempt in range(self.batch_size):
                    if i + self.batch_size >= len(chunk_list):
                        break
                    item = chunk_list.pop(i + self.batch_size - 1)
                    chunk_list.append(item)
                    batch = chunk_list[i:i + self.batch_size]
                    aux_lens = [item['aux_len'] for item in batch]
                    if len(set(aux_lens)) >= min_unique:
                        break
                else:
                    pass
            batch = default_collate(self._pad_aux(batch))
            blist.append(batch)
            i += self.batch_size

        rn = len(chunk_list) - i
        return blist, chunk_list[i:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
