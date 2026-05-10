import os
import math
import torch
import torch.nn as nn
import numpy as np
import traceback

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def _load_embedder(hp, embedder_path):
    if embedder_path is None or embedder_path == '' or embedder_path == 'speechbrain':
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"},
        )

        class SpeechBrainWrapper:
            def __init__(self, clf):
                self.clf = clf

            def __call__(self, dvec_inputs):
                wavs_t = [torch.from_numpy(w.astype(np.float32)).cuda() for w in dvec_inputs]
                wavs_padded = torch.nn.utils.rnn.pad_sequence(wavs_t, batch_first=True)
                emb = self.clf.encode_batch(wavs_padded)  # [B, N, emb_dim]
                if emb.dim() == 3:
                    emb = emb.mean(dim=1)  # [B, emb_dim]
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                return emb

        wrapper = SpeechBrainWrapper(classifier)
        return wrapper, None

    embedder_pt = torch.load(embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)

    class CustomWrapper:
        def __init__(self, model, audio_obj):
            self.model = model
            self.audio = audio_obj

        def __call__(self, dvec_inputs):
            dvec_list = list()
            for inp in dvec_inputs:
                if isinstance(inp, np.ndarray):
                    mel = self.audio.get_mel(inp)
                    mel = torch.from_numpy(mel).float().cuda()
                else:
                    mel = inp.cuda()
                dvec = self.model(mel)
                dvec_list.append(dvec)
            return torch.stack(dvec_list, dim=0)

    wrapper = CustomWrapper(embedder, audio)
    return wrapper, embedder


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    embedder_path = getattr(args, 'embedder_path', None)
    dvec_fn, embedder_ref = _load_embedder(hp, embedder_path)

    audio = Audio(hp)
    model = VoiceFilter(hp).cuda()
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        criterion = nn.MSELoss()
        while True:
            model.train()
            for dvec_inputs, target_mag, mixed_mag in trainloader:
                target_mag = target_mag.cuda()
                mixed_mag = mixed_mag.cuda()

                dvec = dvec_fn(dvec_inputs)
                dvec = dvec.detach()

                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask

                loss = criterion(output, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss, step)

                if step % 10 == 0:
                    logger.info("step %d | train_loss %.6f" % (step, loss))

                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)
                    test_loss, sdr = validate(audio, model, dvec_fn, testloader, writer, step)
                    logger.info("step %d | test_loss %.6f | SDR %.4f dB" % (step, test_loss, sdr))
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
