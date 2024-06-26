__all__ = ['logger', 'prep_dsd_simple', 'prep_dsd_rotten_tomatoes', 'prep_dsd_financial', 'prep_dsd_raw_snli',
           'get_train_valid_test_split', 'ProcessedDataset']


import torch, random, pandas as pd, os, warnings, shutil, uuid
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset, load_from_disk, DatasetDict, ClassLabel
from IPython.display import display, HTML
from .models import get_vm_probs
import logging
logger = logging.getLogger("src.data")



def prep_dsd_simple(cfg):
    """Load the simple dataset and package it up in a DatasetDict (dsd)
    with splits for train, valid, test."""
    dsd = DatasetDict()
    for s in ['train', 'valid', 'test']:
        dsd[s] = load_dataset('csv',
            data_files=f"{cfg.path_data}simple_dataset_{s}.csv", keep_in_memory=False)['train']
    return dsd

def prep_dsd_rotten_tomatoes(cfg):
    """Load the rotten tomatoes dataset and package it up in a DatasetDict (dsd)
    with splits for train, valid, test."""
    dsd = load_dataset("rotten_tomatoes")
    dsd['valid'] = dsd.pop('validation') 
    return dsd

def prep_dsd_financial(cfg):
    """Load the financial dataset and package it up in a DatasetDict (dsd)
    with splits for train, valid, test."""
    dsd = load_dataset("financial_phrasebank", "sentences_50agree")
   # dsd = dsd.filter(lambda x: x['label'] != 1)  # drop netural examples
    dsd = get_train_valid_test_split(dsd)
    dsd = dsd.rename_column('sentence', 'text')
    return dsd

def prep_dsd_trec(cfg): 
    dsd = load_dataset("trec")
    dsd = get_train_valid_test_split(dsd)
    dsd = dsd.rename_column('label-coarse', 'label') 
    dsd = dsd.remove_columns('label-fine')
    return dsd


def get_train_valid_test_split(dsd, train_size=0.8):
        dsd1 = dsd['train'].train_test_split(train_size=train_size, shuffle=True, seed=0)
        dsd2 = dsd1['test'].train_test_split(train_size=0.5, shuffle=True, seed=0)
        return DatasetDict({
            'train': dsd1['train'],
            'valid': dsd2['train'],
            'test': dsd2['test']
        })


class ProcessedDataset:
    """Class that wraps a raw dataset (e.g. from huggingface datasets) and performs preprocessing on it."""
    def __init__(self, cfg, vm_tokenizer, vm_model, pp_tokenizer, sts_model,
                 load_processed_from_file=True):
        """load_processed_from_file: set to true to load completed version from file, false will process the data. """
        self._cfg,self._vm_tokenizer,self._vm_model,self._pp_tokenizer,self._sts_model = cfg,vm_tokenizer,vm_model,pp_tokenizer,sts_model
        shard_suffix = f"_{self._cfg.n_shards}_shards" if self._cfg.use_small_ds else ""
        self.cache_path_raw = f"{self._cfg.path_data_cache}{self._cfg.dataset_name}_raw{shard_suffix}"
        self.cache_path_tkn = f"{self._cfg.path_data_cache}{self._cfg.dataset_name}_tkn{shard_suffix}"

        logger.info(f"Will load dataset {self._cfg.dataset_name} with use_small_ds set to {self._cfg.use_small_ds}")

        if load_processed_from_file:
            if os.path.exists(self.cache_path_raw) and os.path.exists(self.cache_path_tkn):
                logger.info("Cache file found for processed dataset, so loading that dataset.")
                self.dsd_raw = load_from_disk(self.cache_path_raw)
                self.dsd_tkn = load_from_disk(self.cache_path_tkn)
                self._prep_dataloaders()
            else:
                warnings.warn("Cache file not found, so will now process the raw dataset.")
                self._preprocess_dataset()
        else:
            self._preprocess_dataset()
        self._update_cfg()

        logger.debug(f"Dataset lengths: {self._cfg.ds_length}")
        logger.debug(f"Total training epochs:{self._cfg.n_train_steps}")
        logger.debug(f"Last batch size in each epoch is: {self._cfg.dl_leftover_batch_size}")
        logger.debug(f"Dataloader batch sizes are: {self._cfg.dl_batch_sizes}")

    def _preprocess_dataset(self):
        """Add columns, tokenize, transform, prepare dataloaders, and do other preprocessing tasks."""
        if   self._cfg.dataset_name == "simple":          dsd = prep_dsd_simple(self._cfg)
        elif self._cfg.dataset_name == "rotten_tomatoes": dsd = prep_dsd_rotten_tomatoes(self._cfg)
        elif self._cfg.dataset_name == "financial":       dsd = prep_dsd_financial(self._cfg)
        elif self._cfg.dataset_name == "trec":            dsd = prep_dsd_trec(self._cfg)
        else: raise Exception("cfg.dataset_name not valid")
        dsd = dsd.map(self._add_idx, batched=True, with_indices=True)  # add idx column
        if self._cfg.use_small_ds: dsd = self._shard_dsd(dsd)  # do after adding idx so it's consistent across runs
        # add VM score & filter out misclassified examples.
        dsd = dsd.map(self._add_vm_orig_score, batched=True)
        if self._cfg.remove_misclassified_examples:  dsd = dsd.filter(lambda x: x['orig_vm_predclass'] == x['label'])
        dsd = dsd.map(self._prep_input_for_pp_model,  batched=True)  # preprocess raw text so pp model can read
        dsd = dsd.map(self._add_sts_orig_embeddings,  batched=True)  # add STS score
        dsd = dsd.map(self._tokenize_fn,              batched=True)  # tokenize
        # add n_tokens & filter out examples that have more tokens than a threshold
        dsd = dsd.map(self._add_n_tokens,             batched=True)  # add n_tokens
        if self._cfg.remove_long_orig_examples:      dsd = dsd.filter(lambda x: x['n_tokens'] <= self._cfg.orig_max_length)
        dsd = dsd.map(self._add_n_letters,            batched=True)  # add n_letters
        if self._cfg.bucket_by_length: dsd = dsd.sort("n_tokens", reverse=True)  # sort by n_tokens (high to low), useful for cuda memory caching and reducing number of padding tokens
        ## Split dsd into dsd_raw and dsd_tkn
        assert dsd.column_names['train'] == dsd.column_names['valid'] == dsd.column_names['test']
        self.cnames_dsd_raw = ['idx', 'text','text_with_prefix', 'label']
        self.cnames_dsd_tkn = [o for o in dsd.column_names['train'] if o not in ['text', 'text_with_prefix']]
        self.dsd_raw = dsd.remove_columns([o for o in  dsd['train'].column_names if o not in self.cnames_dsd_raw])
        self.dsd_tkn = dsd.remove_columns(["text", 'text_with_prefix'])
        for s in ['train', 'valid', 'test']: assert len(self.dsd_raw[s]) == len(self.dsd_tkn[s])  # check ds has same number of elements in raw and tkn
        self._cache_processed_ds()
        self._prep_dataloaders()

    def _prep_dataloaders(self):
        self.dld_raw = self._get_dataloaders_dict(self.dsd_raw, collate_fn=self._collate_fn_raw)  # dict of data loaders that serve raw text
        self.dld_tkn = self._get_dataloaders_dict(self.dsd_tkn, collate_fn=self._collate_fn_tkn)  # dict of data loaders that serve tokenized text

    def _prep_input_for_pp_model(self, batch):
        """The t5 paraphrase model needs a prefix and postfix, PEGASUS doesn't."""
        if self._cfg.using_t5(): batch['text_with_prefix'] = ["paraphrase: " + sen + " </s>" for sen in batch['text']]
        else:                    batch['text_with_prefix'] = batch['text']
        return batch

    def _add_idx(self, batch, idx):
        """Add row numbers"""
        batch['idx'] = idx
        return batch

    def _add_n_tokens(self, batch):
        """Add the number of tokens of the orig text """
        batch['n_tokens'] = [len(o) for o in batch['input_ids']]
        return batch

    def _add_n_letters(self, batch):
        batch['n_letters'] = [len(o) for o in batch['text']]
        return batch

    def _add_sts_orig_embeddings(self, batch):
        """Add the sts embeddings of the orig text"""
        batch['orig_sts_embeddings'] = self._sts_model.encode(batch['text'], batch_size=64, convert_to_tensor=False)
        return batch

    def _add_vm_orig_score(self, batch):
        """Add the vm score of the orig text"""
        labels = torch.tensor(batch[self._cfg.label_cname], device=self._cfg.device)
        orig_probs,orig_predclass = get_vm_probs(batch['text'], self._cfg, self._vm_tokenizer,
                                                 self._vm_model, return_predclass=True)
        batch['orig_truelabel_probs'] = torch.gather(orig_probs,1, labels[:,None]).squeeze().cpu().tolist()
        batch['orig_vm_predclass'] = orig_predclass.cpu().tolist()
        return batch

    def _tokenize_fn(self, batch):
        """Tokenize a batch of orig text using the paraphrase tokenizer."""
        if self._cfg.remove_long_orig_examples:  return self._pp_tokenizer(batch['text_with_prefix'])   # we drop the long examples later
        else:                               return self._pp_tokenizer(batch['text_with_prefix'], truncation=True, max_length=self._cfg.orig_max_length)

    def _collate_fn_tkn(self, x):
        """Collate function used by the DataLoader that serves tokenized data.
        x is a list (with length batch_size) of dicts. Keys should be the same across dicts.
        I guess an error is raised if not. """
        # check all keys are the same in the list. the assert is quick (~1e-5 seconds)
        for o in x: assert set(o) == set(x[0])
        d = dict()
        for k in x[0].keys():  d[k] = [o[k] for o in x]
        return self._pp_tokenizer.pad(d, pad_to_multiple_of=self._cfg.orig_padding_multiple, return_tensors="pt")

    def _collate_fn_raw(self, x):
        """Collate function used by the DataLoader that serves raw data. x is a list of data."""
        d = dict()
        for o in x: assert set(o) == set(x[0])  # check all keys are the same in list
        for k in x[0].keys(): d[k] = [o[k] for o in x]
        return d

    def _get_sampler(self, ds):
        """Returns a RandomSampler. Used so we can keep the same shuffle order across multiple data loaders.
        Used when self._cfg.shuffle_train = True"""
        g = torch.Generator()
        g.manual_seed(seed)
        return RandomSampler(ds, generator=g)

    def _shard_dsd(self, dsd):
        """Replaces dsd with a smaller shard of itself."""
        for k,v in dsd.items():
            dsd[k] = v.shard(self._cfg.n_shards, 0, contiguous=self._cfg.shard_contiguous)
        return dsd

    def _get_dataloaders_dict(self, dsd, collate_fn):
        """Prepare a dict of dataloaders for train, valid and test"""
        if self._cfg.bucket_by_length and self._cfg.shuffle_train:  raise Exception("Can only do one of bucket by length or shuffle")
        d = dict()
        for split, ds in dsd.items():
            if self._cfg.shuffle_train:
                if split == "train":
                    sampler = self.get_sampler(ds)
                    d[split] =  DataLoader(ds, batch_size=self._cfg.batch_size_train,
                                           sampler=sampler, collate_fn=collate_fn,
                                           num_workers=self._cfg.n_wkrs, pin_memory=self._cfg.pin_memory)
                else:
                    d[split] =  DataLoader(ds, batch_size=self._cfg.batch_size_eval,
                                           shuffle=False, collate_fn=collate_fn,
                                           num_workers=self._cfg.n_wkrs, pin_memory=self._cfg.pin_memory)
            if self._cfg.bucket_by_length:
                batch_size = self._cfg.batch_size_train if split == "train" else self._cfg.batch_size_eval
                d[split] =  DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                       num_workers=self._cfg.n_wkrs, pin_memory=self._cfg.pin_memory)

        # Add eval dataloader for train: same as train but bigger batch size and explicitly no shuffling.
        d['train_eval'] = DataLoader(dsd['train'], batch_size=self._cfg.batch_size_eval, shuffle=False,
                                    collate_fn=collate_fn,
                                     num_workers=self._cfg.n_wkrs, pin_memory=self._cfg.pin_memory)
        return d

    def _update_cfg(self):
        self._cfg.ds_length,self._cfg.dl_n_batches,self._cfg.dl_leftover_batch_size,self._cfg.dl_batch_sizes = dict(),dict(),dict(),dict()
        def get_dl_batch_sizes(batch_size, dl_n_batches):
            if self._cfg.dl_leftover_batch_size[k] == 0:
                return [batch_size for i in range(dl_n_batches)]
            else:
                l = [batch_size for i in range(dl_n_batches - 1)]
                l.append(self._cfg.dl_leftover_batch_size[k])
                return l

        for k,v in self.dsd_raw.items(): self._cfg.ds_length[k] = len(v)   # Dataset lengths
        for k,v in self.dld_raw.items():
            self._cfg.dl_n_batches[k] = len(v)   # Dataloader lengths
            # Dataloader last batch size and list of batch sizes
            ds_k = "train" if k == "train_eval" else k
            if k == "train":
                self._cfg.dl_leftover_batch_size[k] = self._cfg.ds_length[ds_k] % self._cfg.batch_size_train
                self._cfg.dl_batch_sizes[k]     = get_dl_batch_sizes(self._cfg.batch_size_train, self._cfg.dl_n_batches[k])
            else:
                self._cfg.dl_leftover_batch_size[k] = self._cfg.ds_length[ds_k] % self._cfg.batch_size_eval
                self._cfg.dl_batch_sizes[k]     = get_dl_batch_sizes(self._cfg.batch_size_eval, self._cfg.dl_n_batches[k])

        # Total number of training steps
        self._cfg.n_train_steps = self._cfg.n_train_epochs * self._cfg.dl_n_batches['train']

    def _cache_processed_ds(self):
        def _reset_dir(path):
            if os.path.exists(path) and os.path.isdir(path):
                # So deleting the old files sometimes throws errors because of race conditions, I think
                # so as a workaround we will just move files to old directories and then periodicallly clean them.
                #                robust_rmtree(path, logger=None, max_retries=6)
                path_old_files = f"{self._cfg.path_data_cache}old_files/"
                os.makedirs(path_old_files, exist_ok=True)
                shutil.move(path, f"{path_old_files}{uuid.uuid4().hex}")
            os.makedirs(path, exist_ok=True)
        _reset_dir(self.cache_path_raw)
        _reset_dir(self.cache_path_tkn)
        self.dsd_raw.save_to_disk(dataset_dict_path = self.cache_path_raw)
        self.dsd_tkn.save_to_disk(dataset_dict_path = self.cache_path_tkn)

    def show_random_elements(self, ds, num_examples=10):
        """Print some elements in a nice format so you can take a look at them.
        Split is one of 'train', 'test', 'valid'.
        Use for a dataset `ds` from the `dataset` package.  """
        assert num_examples <= len(ds), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(ds)-1)
            while pick in picks:
                pick = random.randint(0, len(ds)-1)
            picks.append(pick)
        df = pd.DataFrame(ds[picks])
        for column, typ in ds.features.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
        display(HTML(df.to_html()))