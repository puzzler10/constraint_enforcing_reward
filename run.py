#!/usr/bin/env python
# coding: utf-8


## Imports and environment variables 
import torch, wandb, os, pandas as pd 
from src.utils import set_seed, set_session_options, setup_logging, setup_parser, update_config_with_parsed_arguments, resume_wandb_run, display_all, print_important_cfg_vars
from src.config import Config
from src.models import prepare_models, get_optimizer
from src.data import ProcessedDataset
from src.trainer import Trainer
from src.insights import (postprocess_df, create_and_log_wandb_postrun_plots, get_training_dfs)
from fastcore.basics import in_jupyter

import logging 
logger = logging.getLogger("run")

import warnings
warnings.filterwarnings("ignore", message="Passing `max_length` to BeamSearchScorer is deprecated")  # we ignore the warning because it works anyway for diverse beam search 


if __name__ == "__main__":
    cfg = Config()  # default values
    if not in_jupyter():  # override with any -- options when running with command line
        parser = setup_parser()
        newargs = vars(parser.parse_args())
        cfg = update_config_with_parsed_arguments(cfg, newargs)
    if cfg.use_small_ds:  cfg = cfg.small_ds()
    set_seed(cfg.seed)
    set_session_options()
    setup_logging(cfg, disable_other_loggers=True)
    vm_tokenizer,vm_model,pp_tokenizer,pp_model,ref_pp_model,sts_model,nli_tokenizer,nli_model,cola_tokenizer,cola_model,cfg = prepare_models(cfg)
    optimizer = get_optimizer(cfg, pp_model)
    ds = ProcessedDataset(cfg, vm_tokenizer, vm_model, pp_tokenizer, sts_model, load_processed_from_file=False)

    cfg.wandb['mode'] = 'online'
    trainer = Trainer(cfg, vm_tokenizer,vm_model,pp_tokenizer,pp_model,ref_pp_model,sts_model,nli_tokenizer,nli_model,cola_tokenizer,cola_model, optimizer,
            ds)
    print_important_cfg_vars(cfg)
    trainer.train()

    trainer.run.finish()




