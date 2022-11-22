from typing import Union

import pytorch_lightning as pl

import LitModel
import network
from config.HANArgs import HANArgs
from config.KTArgs import KTArgs
from data.data2 import KTDataModule2 as KTDataModule


def get_model(args: Union[KTArgs, HANArgs], kt_dm: KTDataModule) -> pl.LightningModule:
    model_name = args.model
    if 'HANKT' in model_name:
        args.hg = kt_dm.heterograph
    model = getattr(network, model_name)(args)

    if 'HANKT1' in model_name:
        lit_model = LitModel.LitHetGKT1(model, args.lr, args.han_lr)
    elif model_name == 'HANKT2':
        lit_model = LitModel.LitHetGKT2(model, args.lr, args.han_lr)
    elif model_name in ('MyDHKT1', 'LtDHKT'):
        lit_model = LitModel.LitDHKT1(model, args.lr, args.alpha)
    elif model_name == 'MyDHKT2':
        lit_model = LitModel.LitDHKT2(model, args.lr, args.alpha)
    elif model_name == 'MyDKTWithEC1':
        lit_model = LitModel.LitKT1(model, args.lr)
    else:
        lit_model = LitModel.LitKT2(model, args.lr)

    return lit_model
