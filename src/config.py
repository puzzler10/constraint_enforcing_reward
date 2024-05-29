
__all__ = ['Config']

import torch
import datetime

class Config:
    def __init__(self):
        """Set up default parameters"""
        ### Models and datasets
        # PP options
        # 1. tuner007/pegasus_paraphrase (2.12 GB)
        # 2. prithivida/parrot_paraphraser_on_T5 (850 MB)
        # 3. ramsrigouthamg/t5-large-paraphraser-diverse-high-quality (2.75 GB)
        self.pp_name = "prithivida/parrot_paraphraser_on_T5"
        self.dataset_name = "trec"
        # STS options
        # 1. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        # 2. sentence-transformers/paraphrase-MiniLM-L12-v2
        self.sts_name = "sentence-transformers/paraphrase-MiniLM-L12-v2"
        # NLI options
        # 1. microsoft/deberta-base-mnli (~512 MB)
        # 2. howey/electra-small-mnli
        self.nli_name = "howey/electra-small-mnli"
        self.cola_name = "textattack/albert-base-v2-CoLA"
        self._select_vm_model()


        ### Important parameters
        self.seed = 420
        self.use_small_ds = False
        self.lr = 1e-4

        self.batch_size_train = 32
        self.batch_size_eval = 32
        self.acc_steps = 2
        self.eval_freq = 1

        self.early_stopping_min_epochs = 8
        self.n_train_epochs = 50

        self.reward_fn = "reward_fn_contradiction_and_letter_diff"
        self.reward_clip_max = 10
        self.reward_clip_min = 0
        self.reward_base = 0
        self.reward_vm_multiplier = 12
        self.sts_threshold = 0.8
        self.acceptability_threshold = 0.5  # min "acceptable" prob required.
        self.contradiction_threshold = 0.2
        self.pp_letter_diff_threshold = 30

        self.reward_penalty_type = "kl_div"  # "kl_div" or "ref_logp"
        self.kl_coef = 0.25        if self.reward_penalty_type == "kl_div"   else None
        self.ref_logp_coef = 0.05  if self.reward_penalty_type == "ref_logp" else None

        self.min_pp_length = 3
        self.max_pp_length = 48
        self.n_eval_seq = 8
        self.decode_method_train = "sample"  # "sample" or "greedy"
        self.decode_method_eval = "beam_search"
        self.gen_params_train = {
            "do_sample": True        if self.decode_method_train == "sample" else False,
            "temperature": 1         if self.decode_method_train == "sample" else None,
            "top_p": 0.95            if self.decode_method_train == "sample" else None,
        }
        self.gen_params_eval = self._get_gen_params_eval()


        # Early stopping (determined during eval on valid set)
        self.early_stopping = True
        self.early_stopping_metric = "any_adv_example_proportion"   # don't add -valid to the end of this.

        # Other parameters (usually left untouched)
        self.orig_max_length = 32  # longest for pegasus is 60, longest for Parrot is 32
        self.pin_memory = True
        self.zero_grad_with_none = False
        self.orig_padding_multiple = 8   # pad input to multiple of this
        self.bucket_by_length = True
        self.shuffle_train = False
        self.remove_misclassified_examples = True
        self.remove_long_orig_examples = True
        self.unfreeze_last_n_layers = "all"  #counting from the back. set to "all" to do no layer freezing, else set to an int

        ### Used for testing
        self.n_shards = None
        self.shard_contiguous = None


        ### W&B parameters
        self.wandb = dict(
            mode = "disabled",  # set to "disabled" to turn off wandb, "online" to enable it. If online, enter project and entity name.
            project = "project_name",
            entity = "entity_name",
            log_grads = False,
            log_grads_freq = 1,  # no effect if wandb_log_grads is False
            log_token_entropy = True,
            log_token_probabilities = True,
            run_notes = f""
        )

        ### Devices and GPU settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.devicenum = torch.cuda.current_device() if self.device.type == 'cuda' else -1
        self.n_wkrs = 4 * torch.cuda.device_count()

        ## Globals
        self.datetime_run = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.path_data = "./data/"
        self.path_checkpoints = "../model_checkpoints/travis_attack/"
        self.path_run = None  # keep as None; this is automatically filled out by trainer (code in utils)
        self.path_data_cache = "/data/tproth/.cache/huggingface/datasets/"  # fill this out yourself
        self.path_logs = f"./logs/"
        self.path_logfile = self.path_logs + f"run_{self.datetime_run}.txt"
        self.path_ref_pp_baselines = "./baselines/ref_pp_baselines/"
        self.path_results = "./results/"


        # Adjust config depending on dataset.
        if self.dataset_name   == "simple":           self.adjust_config_for_simple_dataset()
        elif self.dataset_name == "rotten_tomatoes":  self.adjust_config_for_rotten_tomatoes_dataset()
        elif self.dataset_name == "financial":        self.adjust_config_for_financial_dataset()
        elif self.dataset_name == "trec":             self.adjust_config_for_trec_dataset()

        # Checks
        self._validate_n_epochs()

    def _get_gen_params_eval(self):
        common_params = dict(num_return_sequences=self.n_eval_seq)
        gen_params_eval = dict(
            beam_search         = dict(**common_params, do_sample=False, num_beams=self.n_eval_seq,
                                       top_p=None, temperature=None,
                                       diversity_penalty=None, num_beam_groups=1),
            diverse_beam_search = dict(**common_params, do_sample=False, num_beams=self.n_eval_seq,
                                       top_p=None, temperature=None,
                                       diversity_penalty=1., num_beam_groups=self.n_eval_seq),
            sample              = dict(**common_params, do_sample=True,  num_beams=1,
                                       top_p=0.95, temperature=1,
                                       diversity_penalty=None, num_beam_groups=1)
        )
        return gen_params_eval[self.decode_method_eval]

    def _select_vm_model(self):
        pass
        #if   self.dataset_name in ["rotten_tomatoes", "simple"]:  self.vm_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
        #elif self.dataset_name == "financial":                    self.vm_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        # elif self.dataset_name == "trec":                         self.vm_name = "aychang/distilbert-base-cased-trec-coarse"

    def adjust_config_for_simple_dataset(self):
        """Adjust config for the simple dataset."""
        self.dataset_name = "simple"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self.batch_size_train = 4
        self.batch_size_eval = 4
        self.acc_steps = 2
        self.n_train_epochs = 4
        self.eval_freq = 1
        self._select_vm_model()
        return self

    def adjust_config_for_rotten_tomatoes_dataset(self):
        """Adjust config for the rotten_tomatoes dataset."""
        self.dataset_name = "rotten_tomatoes"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self._select_vm_model()
        return self

    def adjust_config_for_financial_dataset(self):
        """Adjust config for the financial dataset."""
        self.dataset_name = "financial"
        self.orig_cname = "sentence"
        self.label_cname = 'label'
        self._select_vm_model()
        return self

    def adjust_config_for_trec_dataset(self): 
        """Adjust config for the TREC dataset."""
        self.dataset_name = "trec"
        self.orig_cname = "text"
        self.label_cname = 'label'
        self._select_vm_model()
        return self

    def small_ds(self):
        """Adjust the config to use a small dataset (for testing purposes).
        Not possible when using the simple dataset. """
        if self.dataset_name == "simple":
            raise Exception("Don't shard when using the simple dataset (no need)")
        self.use_small_ds = True  # for testing purposes
        self.n_shards = 3
        self.shard_contiguous = False
        return self

    def _validate_n_epochs(self):
        if self.n_train_epochs % self.eval_freq != 0:
            raise Exception("Set n_train_epochs to a multiple of eval_freq so there are no leftover epochs.")

    def using_t5(self):
        return self.pp_name in ["prithivida/parrot_paraphraser_on_T5", "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"]
