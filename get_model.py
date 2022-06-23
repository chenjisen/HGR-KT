from typing import Union

import pytorch_lightning as pl

import LitModel
import network
from config.HANArgs import HANArgs
from config.KTArgs import KTArgs
from config.SeewooArgs import SeewooArgs
from data.data2 import KTDataModule2 as KTDataModule


def get_model(args: Union[KTArgs, SeewooArgs, HANArgs], kt_dm: KTDataModule) -> pl.LightningModule:
    if 'Seewoo' in args.model:
        lit_model = get_seewoo_model(args)
    else:
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


def get_seewoo_model(args: Union[KTArgs, HANArgs]) -> pl.LightningModule:
    from network import Seewoo
    model_name = args.model.removeprefix('Seewoo')
    Seewoo.config.ARGS = args
    if model_name == 'DKT':
        model = Seewoo.DKT(
            args.input_dim, args.hidden_dim, args.num_layers, args.exercise_count, args.dropout)
    elif model_name == 'DKVMN':
        model = Seewoo.DKVMN(
            args.key_dim, args.hidden_dim, args.summary_dim, args.exercise_count, args.concept_num)
    elif model_name == 'NPA':
        model = Seewoo.NPA(
            args.input_dim, args.hidden_dim, args.attention_dim, args.fc_dim,
            args.num_layers, args.exercise_count, args.dropout)
    elif model_name == 'SAKT':
        model = Seewoo.SAKT(
            args.hidden_dim, args.exercise_count, args.num_layers, args.num_head, args.dropout)
    else:
        raise NotImplementedError
    model.d_model = args.hidden_dim
    model.name = model_name
    lit_model = LitModel.LitSeewooKT1(model, args.lr, args.warm_up_step_count)
    return lit_model
