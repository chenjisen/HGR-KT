import logging
import warnings
from pprint import pformat
from typing import Union

import coloredlogs
import numpy as np
import pytorch_lightning as pl

from config.HANArgs import HANArgs
from config.KTArgs import KTArgs
from config.SeewooArgs import SeewooArgs
from config.parser import MyParser
from data.data2 import KTDataModule2 as KTDataModule
from get_model import get_model


def train_model(args: Union[KTArgs, SeewooArgs, HANArgs]) -> dict[str, float]:
    kt_dm = KTDataModule(args)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.weight_dir,
            filename=f'fold={kt_dm.fold}-' + '{epoch}-{step}-{val_auc:.4f}',
            monitor='val_auc',
            mode='max'
        ),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.early_stop_patience,
        )
    ]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    lit_model = get_model(args, kt_dm)

    if args.train:
        logging.debug("training...")
        trainer.fit(lit_model, kt_dm, ckpt_path=args.ckpt_path)
    else:
        trainer.checkpoint_callback.best_model_path = args.ckpt_path
    result = trainer.test(lit_model, datamodule=kt_dm, ckpt_path='best')
    logging.debug(trainer.checkpoint_callback.best_model_path)
    return result[0]


def train(**kwargs) -> dict[str, Union[float, str]]:
    train_init()

    parser = MyParser(KTArgs)
    if 'HAN' in parser.parse_model_name(kwargs):
        parser.add_argument_from_class(HANArgs)
    args: KTArgs = parser.parse_args(kwargs)

    logging.getLogger().addHandler(logging.FileHandler(args.log_path, 'w'))
    logging.critical(args.dataset_name + '-' + args.model)
    logging.debug(f'args: {args}')

    acc_list = []
    auc_list = []

    for split in range(args.n_splits):
        seed = args.next_seed
        logging.warning(f'{args.dataset_name}-{args.model}, {split=}, {seed=}')
        if args.determine:
            pl.seed_everything(seed, workers=True)
        result = train_model(args)
        acc_list.append(result['test_acc'])
        auc_list.append(result['test_auc'])

    acc_mean, auc_mean, = np.mean(acc_list).item(), np.mean(auc_list).item()
    acc_std, auc_std = np.std(acc_list).item(), np.std(auc_list).item()

    desc = f'acc: {acc_mean:.04f} ± {acc_std:.04f}, auc: {auc_mean:.04f} ± {auc_std:.04f}'
    logging.critical(args.dataset_name + '-' + args.model + ': ' + desc + '\n')
    return {'acc': acc_mean, 'auc': auc_mean, 'desc': desc}


def train_init():
    with open('filtered-warnings') as f:
        for s in f:
            warnings.filterwarnings('ignore', s[:-1])
    coloredlogs.install(level='DEBUG', fmt="%(asctime)s %(message)s")


def print_results(models, dataset_list, results):
    logging.warning(pformat(results))
    for model in models:
        dataset_list_2 = []
        result_list = []
        for dataset in dataset_list:
            if (model, dataset) in results:
                result = results[model, dataset]
                dataset_list_2.append(dataset)
                result_list.extend((f"{result['acc']:.4f}", f"{result['auc']:.4f}"))
        logging.critical('\t  '.join(dataset_list_2))
        logging.critical(' '.join(result_list))


def get_args(args_dict, key):
    for k, v in args_dict.items():
        if k in key or key in k:
            return v
    return {}


if __name__ == '__main__':
    train()
