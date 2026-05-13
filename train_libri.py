import argparse
import pprint

import toml

from src.dataset_libri import make_dataloader
from src.trainer import SiSnrTrainer
from src.utils import dump_json, get_logger
from model import pTFGridNet

import faulthandler
faulthandler.enable()

logger = get_logger(__name__)


def run(configs):
    gpuids = tuple(configs['gpu']['gpu_ids'])

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

    for conf, fname in zip([configs['net'], configs], ["net_config.json", "config.json"]):
        dump_json(conf, configs['save']['save_filename'], fname)

    train_loader = make_dataloader(train=True,
                                    data_dirs=configs['path']['train']['data_dirs'],
                                    chunk_length=configs['signal']['chunk_length'],
                                    sample_rate=configs['signal']['sr'],
                                    batch_size=configs['dataloader']['batch_size'],
                                    pin_memory=configs['dataloader']['pin_memory'],
                                    num_workers=configs['dataloader']['num_workers'])
    dev_loader = make_dataloader(train=False,
                                  data_dirs=configs['path']['val']['data_dirs'],
                                  chunk_length=configs['signal']['chunk_length'],
                                  sample_rate=configs['signal']['sr'],
                                  batch_size=configs['dataloader']['batch_size'],
                                  pin_memory=configs['dataloader']['pin_memory'],
                                  num_workers=configs['dataloader']['num_workers'])

    trainer.run(train_loader, dev_loader, num_epochs=configs['optimizer']['epochs'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to start pTFGridNet.py training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default="configs/train_libri_config.toml",
                        help="Path to configs")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    configs = toml.load(args.config)
    print(configs)

    import traceback
    try:
        run(configs)
    except Exception:
        traceback.print_exc()
        raise
