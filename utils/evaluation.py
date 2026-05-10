import torch
import numpy as np
import torch.nn as nn
from mir_eval.separation import bss_eval_sources


def validate(audio, model, dvec_fn, testloader, writer, step):
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    sdr = 0.0
    with torch.no_grad():
        for batch in testloader:
            dvec_input, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]

            target_mag = target_mag.unsqueeze(0).cuda()
            mixed_mag = mixed_mag.unsqueeze(0).cuda()

            if isinstance(dvec_input, np.ndarray):
                dvec = dvec_fn([dvec_input])
            else:
                dvec = dvec_fn([dvec_input.numpy()]) if hasattr(dvec_input, 'numpy') else dvec_fn([dvec_input])
            dvec = dvec.detach()

            est_mask = model(mixed_mag, dvec)
            est_mag = est_mask * mixed_mag
            test_loss = criterion(target_mag, est_mag).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            target_mag = target_mag[0].cpu().detach().numpy()
            est_mag = est_mag[0].cpu().detach().numpy()
            est_wav = audio.spec2wav(est_mag, mixed_phase)
            est_mask = est_mask[0].cpu().detach().numpy()

            sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, target_wav, est_wav,
                                  mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                  step)
            break

    model.train()
    return test_loss, sdr
