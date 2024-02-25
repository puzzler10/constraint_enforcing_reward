__all__ = ['logger', 'set_seed', 'set_session_options', 'setup_logging', 'setup_parser',
           'update_config_with_parsed_arguments', 'timecode', 'print_device_info', 'dump_tensors', 'Monitor',
           'show_gpu', 'round_t', 'merge_dicts', 'display_all', 'print_important_cfg_vars', 'unpack_nested_lists_in_df',
           'append_df_to_csv', 'robust_rmtree', 'test_pp_model', 'start_wandb_run', 'resume_wandb_run', 'table2df']


import torch, numpy as np, pandas as pd, time, GPUtil, wandb, os, shutil, subprocess, argparse
from timeit import default_timer as timer
from pprint import pprint
from threading import Thread
import logging
logger = logging.getLogger("src.utils")


def set_seed(seed):
    """Sets all seeds for the session"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



def set_session_options():
    """Sets some useful options for the sesson"""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # set to false if not working
    pd.set_option("display.max_colwidth", 400)
    pd.options.mode.chained_assignment = None


def setup_logging(cfg, disable_other_loggers=True):
    """taken from this recipe from the logging cookbook:
    https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=cfg.path_logfile,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s') # set a format which is simpler for console use
    console.setFormatter(formatter)  # tell the handler to use this format
    logging.getLogger('').addHandler(console)  # add the handler to the root logger

    if disable_other_loggers:
        allowed_modules = ["src", "wandb"]  #  "sentence_transformers", "transformers", "datasets"
        logger.debug(f"Disabling all loggers except those from the following libraries: {allowed_modules}")
        for log_name, log_obj in logging.Logger.manager.loggerDict.items():
            if not any(mod in log_name for mod in allowed_modules):
                log_obj.disabled = True



def setup_parser():
    """Set up command line options"""
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--dataset_name", choices=['simple', 'rotten_tomatoes', 'financial'])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--acc_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_train_epochs", type=int)
    parser.add_argument("--batch_size_train", type=int)
    parser.add_argument("--batch_size_eval", type=int)
    parser.add_argument("--eval_freq", type=int)
    parser.add_argument("--early_stopping_min_epochs", type=int)
    parser.add_argument("--early_stopping_metric")

    ## Paraphrase generation parameters (general)
    parser.add_argument("--min_pp_length", type=int)
    parser.add_argument("--max_pp_length", type=int)
    # Train
    parser.add_argument("--decode_method_train", choices=['sample', 'greedy'])
    parser.add_argument("--gen_train_temperature", type=float)
    parser.add_argument("--gen_train_top_p", type=float)
    # Eval
    parser.add_argument("--n_eval_seq", type=int)
    parser.add_argument("--decode_method_eval", choices=['sample', 'beam_search', 'diverse_beam_search',  "diverse_beam_search_low_diversity", "diverse_beam_search_high_diversity"])
    parser.add_argument("--gen_eval_temperature", type=float)
    parser.add_argument("--gen_eval_top_p", type=float)
    parser.add_argument("--gen_eval_num_beam_groups", type=int)
    parser.add_argument("--gen_eval_diversity_penalty", type=float)

    # Paraphrase reward terms
    parser.add_argument("--reward_fn")
    parser.add_argument("--reward_clip_max", type=float)
    parser.add_argument("--reward_vm_multiplier", type=float)
    parser.add_argument("--sts_threshold", type=float)
    parser.add_argument("--acceptability_threshold", type=float)
    parser.add_argument("--contradiction_threshold", type=float)
    parser.add_argument("--pp_letter_diff_threshold", type=int)

    # Reward penalty terms
    parser.add_argument("--reward_penalty_type", choices=["kl_div", "ref_logp"])
    parser.add_argument("--kl_coef", type=float)
    parser.add_argument("--ref_logp_coef", type=float)

    #parser.add_argument('args', nargs=argparse.REMAINDER)  # activate to put keywords in kwargs.
    return parser


def update_config_with_parsed_arguments(cfg, newargs):
    """newargs: dict of values from the parser"""
    for k,v in newargs.items():
        if v is not None:
            if   k.startswith("gen_train"): cfg.gen_params_train[k.split('gen_train_')[1]] = v
            elif k.startswith("gen_eval") : cfg.gen_params_eval[ k.split('gen_eval_' )[1]] = v
            else:     setattr(cfg, k, v)
    if newargs["decode_method_train"] == "greedy":
        cfg.gen_params_train["do_sample"] = False
        cfg.gen_params_train['temperature'] = None
        cfg.gen_params_train['top_p'] = None
        if newargs["gen_train_temperature"] is not None or newargs["gen_train_top_p"] is not None:
            warnings.warn("Generation parameters for training specified but will be ignored because train decode method set to greedy.")
    if newargs["decode_method_eval"] == "sample":
        if newargs["gen_eval_num_beam_groups"] is not None or newargs["gen_eval_diversity_penalty"] is not None:
            warnings.warn("Some generation parameters for eval ignored because eval decode method set to sample.")
        cfg.gen_params_eval["do_sample"] = True
        cfg.gen_params_eval["num_beams"] = 1
        cfg.gen_params_eval['diversity_penalty'] = None
        cfg.gen_params_eval['num_beam_groups'] = None
    if newargs["decode_method_eval"] == "beam_search":
        if newargs["gen_eval_num_beam_groups"] is not None or newargs["gen_eval_diversity_penalty"] is not None or \
           newargs["gen_eval_top_p"] is not None or newargs["gen_eval_temperature"] is not None:
            warnings.warn("Some generation parameters for eval ignored because eval decode method set to beam_search.")
        cfg.gen_params_eval["do_sample"] = False
        cfg.gen_params_eval["num_beams"] = cfg.n_eval_seq
        cfg.gen_params_eval['diversity_penalty'] = None
        cfg.gen_params_eval['top_p'] = None
        cfg.gen_params_eval['temperature'] = None
        cfg.gen_params_eval['num_beam_groups'] = None
    if newargs["decode_method_eval"] == "diverse_beam_search":
        if newargs["gen_eval_top_p"] is not None or newargs["gen_eval_temperature"] is not None:
            warnings.warn("Some generation parameters for eval ignored because eval decode method set to diverse beam search.")
        cfg.gen_params_eval["do_sample"] = False
        cfg.gen_params_eval["num_beams"] = cfg.n_eval_seq
        cfg.gen_params_eval['top_p'] = None
        cfg.gen_params_eval['temperature'] = None
    if newargs["decode_method_eval"] == "diverse_beam_search_low_diversity":
        num_beam_groups =  max(int(cfg.n_eval_seq / 8), 1)
        logger.info(f"Setting num_beam_groups to {num_beam_groups}")
        cfg.gen_params_eval["num_beam_groups"] = num_beam_groups
        cfg.gen_params_eval["diversity_penalty"] = 1.
        cfg.gen_params_eval["do_sample"] = False
        cfg.gen_params_eval["num_beams"] = cfg.n_eval_seq
        cfg.gen_params_eval['top_p'] = None
        cfg.gen_params_eval['temperature'] = None
    if newargs["decode_method_eval"] == "diverse_beam_search_high_diversity":
        logger.info(f"Setting num_beam_groups to {cfg.n_eval_seq}")
        cfg.gen_params_eval["num_beam_groups"] = cfg.n_eval_seq
        cfg.gen_params_eval["diversity_penalty"] = 1.
        cfg.gen_params_eval["do_sample"] = False
        cfg.gen_params_eval["num_beams"] = cfg.n_eval_seq
        cfg.gen_params_eval['top_p'] = None
        cfg.gen_params_eval['temperature'] = None

    if   cfg.dataset_name == "simple":           cfg.adjust_config_for_simple_dataset()
    elif cfg.dataset_name == "rotten_tomatoes":  cfg.adjust_config_for_rotten_tomatoes_dataset()
    elif cfg.dataset_name == "financial":        cfg.adjust_config_for_financial_dataset()
    cfg._validate_n_epochs()
    cfg.gen_params_eval = cfg._get_gen_params_eval()
    return cfg


class timecode:
    """This class is used for timing code"""
    def __enter__(self):
        self.t0 = timer()
        return self

    def __exit__(self, type, value, traceback):
        self.t = timer() - self.t0


def print_device_info():
    """
    Prints some statistics around versions and the GPU's available for
    the host machine
    """
    import torch
    import sys
    print("######## Diagnostics and version information ######## ")
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    #! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name())
    print ('Current cuda device ', torch.cuda.current_device())
    print("#################################################################")



def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector.
    Useful when running into an out of memory error on the GPU. """
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                            " GPU" if obj.is_cuda else "",
                                            " pinned" if obj.is_pinned else "",
                                            pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                    type(obj.data).__name__,
                                                    " GPU" if obj.is_cuda else "",
                                                    " pinned" if obj.data.is_pinned else "",
                                                    " grad" if obj.requires_grad else "",
                                                    " volatile" if obj.volatile else "",
                                                    pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)



class Monitor(Thread):
    """Use this to check that you are using the GPU during your pytorch functions and to track memory usage
    of the GPU's as well."""
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True



def show_gpu(msg):
    """
    ref: https://github.com/huggingface/transformers/issues/1742#issue-518262673
    put in logger.info()
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'],
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    return f"{msg} {100*pct:2.1f}% ({used} out of {total})"


def round_t(t, dp=2):
    """Return rounded tensors for easy viewing. t is a tensor, dp=decimal places"""
    if t.device.type == "cuda": t=t.cpu()
    return t.detach().numpy().round(dp)


def merge_dicts(d1, d2):
    """Merge the two dicts and return the result. Check first that there is no key overlap."""
    assert set(d1.keys()).isdisjoint(d2.keys())
    return {**d1, **d2}


def display_all(df):
    with pd.option_context("display.max_rows", 3000):
        with pd.option_context("display.max_columns", 1000):
            with pd.option_context("max_colwidth", 480):
                display(df)


def print_important_cfg_vars(cfg):
    d = vars(cfg)
    ignore_keys = ['dl_batch_sizes', 'orig_cname', 'label_cname', 'n_shards', 'shard_contiguous', 'save_model_while_training', 'save_model_freq', 'devicenum',
                    'splits', 'metrics','vm_num_labels','vocab_size', 'contra_label','dl_n_batches','dl_leftover_batch_size','acc_leftover_batches',
                  'bucket_by_length', 'embedding_padding_multiple', 'n_train_steps', 'pin_memory', 'pad_token_embeddings', 'remove_long_orig_examples',
                  'remove_misclassified_examples',  'reward_clip_min', 'reward_base', 'shuffle_train', 'wandb', 'zero_grad_with_none', 'unfreeze_last_n_layers',
                  'datetime_run', 'device',  'n_wkrs', 'orig_padding_multiple', 'run_id', 'run_name']
    ignore_keys = ignore_keys + [o for o in d.keys() if "path_" in o]
    d1 = {k:v for k,v in d.items() if k not in ignore_keys}
    pprint(d1, sort_dicts=False)


def unpack_nested_lists_in_df(df, scalar_cols=[]):
    """Take a df where we have lists stored in the cells and convert it to many rows.
    Put all columns without lists stored in the cells into `scalar_cols`."""
    return df.set_index(scalar_cols).apply(pd.Series.explode).reset_index()


def append_df_to_csv(df, path):
    """Checks columns and other stuff before appending"""
    import os
    if not os.path.isfile(path):   df.to_csv(path, mode='a', index=False)  # create with header if not exists
    elif len(df.columns) != len(pd.read_csv(path, nrows=1).columns):
        raise Exception("Columns do not match. Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(path, nrows=1).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(path, nrows=1).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match.")
    else:
        df.to_csv(path, mode='a', index=False, header=False)


def robust_rmtree(path, logger=None, max_retries=6):
    """Robustly tries to delete paths.
    Retries several times (with increasing delays) if an OSError
    occurs.  If the final attempt fails, the Exception is propagated
    to the caller.
    """
    dt = 1
    for i in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if logger:
                logger.info('Unable to remove path: %s' % path)
                logger.info('Retrying after %d seconds' % dt)
            time.sleep(dt)
            dt *= 2

    # Final attempt, pass any Exceptions up to caller.
    shutil.rmtree(path)


def test_pp_model(text, pp_tokenizer, pp_model):
    """Get some paraphrases for a bit of text"""
    print("ORIGINAL\n",text)
    num_beams = 10
    num_return_sequences = 10
    batch = pp_tokenizer([text], return_tensors='pt', max_length=60, truncation=True)
    translated = pp_model.generate(**batch, num_beams=num_beams, num_return_sequences=num_return_sequences, length_penalty=1)
    tgt_text = pp_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def start_wandb_run(cfg, log_code=True):
    """Start wandb run, set up paths, update cfg, create dir for model artifacts if needed,"""
    run = wandb.init(project=cfg.wandb['project'], entity=cfg.wandb['entity'],
                          config=vars(cfg), mode=cfg.wandb['mode'],
                          notes=cfg.wandb['run_notes'], save_code=log_code)
    if log_code: run.log_code(".")
    cfg.run_name,cfg.run_id = run.name, run.id
    cfg.path_run = f"{cfg.path_checkpoints}{run.name}/"
    if not os.path.exists(cfg.path_run): os.makedirs(cfg.path_run, exist_ok=True)
    return run, cfg


def resume_wandb_run(cfg):
    run = wandb.init(project="project_name", entity="entity_name", config=vars(cfg),
                     resume='must', id=cfg.run_id , mode=cfg.wandb['mode'])
    return run


def table2df(table):
    """Convert wandb table to pandas dataframe"""
    return pd.DataFrame(data=table.data, columns=table.columns)