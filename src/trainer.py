__all__ = ['logger', 'Trainer']


import torch, wandb, gc, numpy as np, pandas as pd, os, copy, warnings, string
from wandb.data_types import Histogram
from tqdm.auto import tqdm
from .utils import (timecode, merge_dicts, unpack_nested_lists_in_df,
                                  append_df_to_csv, start_wandb_run)
from .tests import check_no_nans_or_infs
from .models import save_pp_model, resume_pp_model, get_vm_probs, get_nli_probs, get_cola_probs
from datasets import Dataset
from torch.distributions import Categorical
from sentence_transformers.util import pytorch_cos_sim
from fastcore.basics import store_attr
import logging
logger = logging.getLogger("src.trainer")


class Trainer:
    def __init__(self, cfg, vm_tokenizer, vm_model, pp_tokenizer, pp_model, ref_pp_model, sts_model,
                 nli_tokenizer, nli_model, cola_tokenizer, cola_model, optimizer, ds):
        store_attr()
        self._cfg = self.cfg; del self.cfg;
        self.epoch,self.acc_num,self.global_step,self.param_norm = 0,0,0,0
        self._setup_data_stores()
        self._setup_gradient_accumulation_variables()
        self.early_stopping_flag = False
        self.linking_contrast_phrases = [o.strip() for o in open("./linking_contrast_phrases.txt").readlines()]


    def train(self):
        self._setup_wandb_run()
        #%lprun -f _training_function -f  get_pp_logp -f training_step -f  reward_fn -f  loss_fn -f eval_dl  notebook_launcher(_training_function, args=(pp_model, vm_model, dld_tkn, dld_raw, optimizer), num_processes=1)
        self._training_function()

    def _is_last_epoch(self): return (self.early_stopping_flag and self._cfg.early_stopping) or (self.epoch == self._cfg.n_train_epochs)

    def _setup_wandb_run(self):
        """Init wandb run, set up paths, create dir for model artifacts if needed, """
        self.run, self._cfg = start_wandb_run(self._cfg, log_code=True)
        if self._cfg.wandb['log_grads']: wandb.watch(self.pp_model, log='gradients', log_freq=self._cfg.wandb['log_grads_freq'])

    def _setup_data_stores(self):
     #   self.eval_epoch_df_d = dict(train=[], valid=[], test=[]) # each eval epoch dataframe appended to here
        self.orig_baselines = dict()    # keys are idx, values are mean reward per (orig, pp_l) during training eval
        self.initial_metric_d = dict(train=dict(), valid=dict(), test=dict())  # used for wandb metrics at the end
        # Early stopping
        self.best_eval_valid_metric,self.best_eval_valid_epoch = -99999,-99999   # tracks best val + epoch so we can load the model later
        self.eval_valid_metrics = list()  # holds all eval values (used for calculating median)
        self.best_model_path = ""

    def _setup_gradient_accumulation_variables(self):
        """acc_global_l is a list of all batch sizes encountered during training.
            """
        self.acc_global_l = self._cfg.dl_batch_sizes['train'] * self._cfg.n_train_epochs
        assert len(self.acc_global_l) ==  self._cfg.n_train_steps
        # Check if there will be leftover batches
        self._cfg.acc_leftover_batches =  self._cfg.n_train_steps % self._cfg.acc_steps
        if self._cfg.acc_leftover_batches != 0:
            msg = f"Config set to do gradient accumulation every {self._cfg.acc_steps} batches, and there are \
            {self._cfg.n_train_steps} total training steps, so there will be {self._cfg.acc_leftover_batches} batches at \
            the end that will not be trained on."
            warnings.warn(msg)
        self._reset_acc_lists()

    def _reset_acc_lists(self):
        """call this at start and every time you call opt step"""
        # acc_current_l is a list of the batch sizes in the current accumulation batch.
        last_step = (self._cfg.n_train_steps - 1) - self._cfg.acc_leftover_batches
        if self.global_step == 0:   # at start of training
            self.acc_current_l = self.acc_global_l[self.global_step:self._cfg.acc_steps]
            assert len(self.acc_current_l) == self._cfg.acc_steps
        else:
            self.acc_current_l = self.acc_global_l[(self.global_step+1):(self.global_step+self._cfg.acc_steps+1)]
            if self.global_step == last_step:  assert len(self.acc_current_l) == self._cfg.acc_leftover_batches
            else:                              assert len(self.acc_current_l) == self._cfg.acc_steps
        self.acc_current_n_examples = sum(self.acc_current_l)

    def _training_function(self):
        progress_bar = tqdm(range(self._cfg.n_train_steps))
        self.pp_model.zero_grad(set_to_none=self._cfg.zero_grad_with_none)

        # initial eval of untrained model (at epoch 0) for results comparison and to calculate initial baselines for REINFORCE
        logger.info("Launching initial eval run: train"); self._eval_function(split='train')
        logger.info("Launching initial eval run: valid"); self._eval_function(split='valid')
        logger.info("Launching initial eval run: test" ); self._eval_function(split='test')

        # Training loop
        for self.epoch in range(1, self._cfg.n_train_epochs+1):
            logger.info(f"Now on epoch {self.epoch} of {self._cfg.n_train_epochs}")
            if not self.pp_model.training: self.pp_model.train()
            with timecode() as time_train_one_epoch:
                self.train_batch_results = []  # each train batch appended to here, list of dicts
                for self.batch_num, (data, raw) in enumerate(zip(self.ds.dld_tkn['train'], self.ds.dld_raw['train'])):
                    self.batch_d,self.batch_time_d,self.batch_wandb_d = dict(),dict(),dict()
                    self._training_step(data, raw)
                    if self._batch_for_opt_step(): self._reset_acc_lists()
                    self.acc_num = (self.acc_num + 1) % self._cfg.acc_steps
                    self.global_step += 1
                    progress_bar.update(1)
            wandb.log({'time/train_one_epoch_time': time_train_one_epoch.t,
                       'time/train_one_epoch_thoroughput': len(self.ds.dsd_tkn['train']) / time_train_one_epoch.t,
                       'epoch': self.epoch}, commit=True)
            if self._cfg.wandb['log_grads'] and self.epoch % self._cfg.wandb_log_grads_freq == 0:
                plt = self._plot_grad_flow(self.pp_model.named_parameters())
                wandb.log({"gradient flow": wandb.Image(plt)})  # doesn't work as a non-image (i.e. plotly)
                del plt

            df = pd.DataFrame(self.train_batch_results)
            df = df.apply(pd.Series.explode).reset_index(drop=True)
            df = self._set_df_colorder(df, is_eval=False)
            append_df_to_csv(df, path = f"{self._cfg.path_run}training_step.csv")

            # Evaluation loop
            if self.epoch % self._cfg.eval_freq == 0:
                with timecode() as time_eval_train:    self._eval_function(split='train')
                with timecode() as time_eval_valid:    self._eval_function(split='valid')
                with timecode() as time_eval_gc_collect:                   gc.collect()
                with timecode() as time_eval_empty_cache:                  torch.cuda.empty_cache()
                wandb.log({'time/eval_train_time': time_eval_train.t, 'time/eval_valid_time': time_eval_valid.t,
                           'time/eval_train_thoroughput': len(self.ds.dsd_tkn['train']) / time_eval_train.t,
                           'time/eval_valid_thoroughput': len(self.ds.dsd_tkn['valid']) / time_eval_valid.t,
                           'time/eval_gc_collect': time_eval_gc_collect.t,
                           'time/eval_empty_cache': time_eval_empty_cache.t,
                           'epoch': self.epoch}, commit=True)
            if self._is_last_epoch():
                logger.info(f"Evaluating test set with best model at path : {self.best_model_path}")
                self.pp_model, self.optimizer = resume_pp_model(self.pp_model, self.optimizer, self.best_model_path)
                self._eval_function(split='test')
                self._update_wandb_summary()
                break

    def _training_step(self, data, raw):
        """Forward pass, loss function, backwards pass, parameter update (with gradient accumulation optional),
        recording results, wandb logging.
        """
        if not self.pp_model.training: self.pp_model.train()
        if not self.vm_model.training: self.vm_model.train()
        for k in ['input_ids', 'attention_mask', 'label', "orig_truelabel_probs", "orig_sts_embeddings"]: data[k] = data[k].to(self._cfg.device)
        with timecode() as self.batch_time_d['time_generate_pp']:
            pp_output = self.pp_model.generate_with_grad(
                input_ids=data['input_ids'], attention_mask=data['attention_mask'],
                num_return_sequences=1, num_beams=1, **self._cfg.gen_params_train,
                min_length=self._cfg.min_pp_length, max_length=self._cfg.max_pp_length,
                return_dict_in_generate=True, output_scores=True)  # , remove_invalid_values=False
               # pad_token_id = self.pp_tokenizer.pad_token_id, eos_token_id = self.pp_tokenizer.eos_token_id
            pp_l = self.pp_tokenizer.batch_decode(pp_output.sequences, skip_special_tokens=True)
        # Update_batch_size_and_length_variables
        self.orig_batch_size,self.orig_length =   data['input_ids'].shape[0],   data['input_ids'].shape[1]
        self.pp_batch_size,  self.pp_length   = pp_output.sequences.shape[0], pp_output.sequences.shape[1]
        # Loss function, backwards pass, optimizer step + measure gradient update norm
        with timecode() as self.batch_time_d['time_loss_fn']:            loss_batch = self._loss_fn(data, raw, pp_output, pp_l)
        with timecode() as self.batch_time_d['time_backwards']:          loss_batch.backward()
        with timecode() as self.batch_time_d['time_calc_gradient_norm']: self.grad_norm = self._get_gradient_update_norm()
        with timecode() as self.batch_time_d['time_opt_step_and_calc_param_norm']:
            if self._batch_for_opt_step():  self._opt_step_and_calc_param_norm()
            else:
                self.param_norm = 0
                with timecode() as self.batch_time_d['time_opt_step']: pass  # need some value here
        # Gather values, append to containers, log to wandb
        self._prepare_train_batch_d(raw, data, pp_l)
        self.train_batch_results.append(self.batch_d)
        self._wandb_log_training_step()

    def _opt_step_and_calc_param_norm(self):
        ## record initial parameters
        params_all = [o.detach() for o in self.pp_model.parameters() if o[1].requires_grad]
        params_initial = [ p.clone() for p in params_all]
        # step + zero grad
        with timecode() as self.batch_time_d['time_opt_step']:
            self.optimizer.step()
            self.pp_model.zero_grad(set_to_none=self._cfg.zero_grad_with_none)
        # record norm
        self.param_norm = 0
        for (p_init, p_new) in zip(params_initial, params_all):
            self.param_norm += (p_new - p_init).data.norm(2).item() ** 2
        self.param_norm = self.grad_norm ** 0.5

    def _get_gradient_update_norm(self):
        total_norm = 0
        for p in [o for o in self.pp_model.parameters() if o[1].requires_grad]:
            if p.grad is not None:  # the embed_position layers on encoder/decoder dont keep grad ()
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _batch_for_opt_step(self): return self.acc_num == (self._cfg.acc_steps - 1)

    def _prepare_train_batch_d(self, raw, data, pp_l):
        self.batch_d = merge_dicts(self.batch_d, { 'idx': raw['idx'],
            'epoch': self.epoch, 'batch_num': self.batch_num, 'global_step': self.global_step,
            'acc_num': self.acc_num, "acc_batch_n_examples": self.acc_current_n_examples,
            "orig": raw['text'], "label": data['label'].cpu().tolist(),
            "orig_truelabel_probs": data['orig_truelabel_probs'].cpu().tolist(),
            'orig_length': self.orig_length, 'orig_batch_size': self.orig_batch_size,
            "pp": pp_l, 'pp_length': self.pp_length, 'pp_batch_size': self.pp_batch_size
        })
        for k, v in self.batch_time_d.items(): self.batch_time_d[k] = v.t  # extract times from timecode object
        self.batch_d = merge_dicts(self.batch_d, self.batch_time_d)

    def _wandb_log_training_step(self):
        self.batch_wandb_d = merge_dicts(self.batch_wandb_d, {
            'vm_scores_hist':       Histogram(self.batch_d['vm_scores']),
            'vm_scores_mean':       np.mean(  self.batch_d['vm_scores']),
            'sts_scores_hist':      Histogram(self.batch_d['sts_scores']),
            'sts_scores_mean':      np.mean(  self.batch_d['sts_scores']),
            'pp_logp_hist':         Histogram(self.batch_d['pp_logp']),
            'pp_logp_mean':         np.mean(  self.batch_d['pp_logp']),
            'ref_logp_hist':        Histogram(self.batch_d['ref_logp']),
            'ref_logp_mean':        np.mean(  self.batch_d['ref_logp']),
            'diff_logp_hist' :         Histogram(self.batch_d['diff_logp']),
            'reward_pp_hist':         Histogram(self.batch_d['reward_pp']),
            'reward_pp_mean':         np.mean(  self.batch_d['reward_pp']),
            'reward_pp_minus_baseline_hist':          Histogram(  self.batch_d['reward_pp_minus_baseline']),
            'reward_pp_minus_baseline_mean':          np.mean(    self.batch_d['reward_pp_minus_baseline']),
            'reward_pp_minus_baseline_with_penalty_hist':          Histogram(self.batch_d['reward_pp_minus_baseline_with_penalty']),
            'reward_pp_minus_baseline_with_penalty_mean':          np.mean(  self.batch_d['reward_pp_minus_baseline_with_penalty']),
            'loss_hist'   :         Histogram(self.batch_d['loss']),
            'pp_letter_diff_hist':     Histogram(self.batch_d['pp_letter_diff']),
            'pp_letter_diff_mean':     np.mean(  self.batch_d['pp_letter_diff']),
            'pp_letter_percent_hist':  Histogram(self.batch_d['pp_letter_percent']),
            'pp_letter_percent_mean':  np.mean(  self.batch_d['pp_letter_percent']),
            'contradiction_scores_hist':     Histogram(self.batch_d['contradiction_scores']),
            'contradiction_scores_mean':     np.mean(  self.batch_d['contradiction_scores']),
            'acceptability_scores_hist':     Histogram(self.batch_d['acceptability_scores']),
            'acceptability_scores_mean':     np.mean(  self.batch_d['acceptability_scores']),
            'lcp_conditions_mean':     np.mean(  self.batch_d['lcp_conditions']),
            'acc_batch_sizes':      Histogram(self.acc_current_l),
            "gradient_norm":        self.grad_norm,
            "parameter_norm":       self.param_norm
        })
        self.batch_wandb_d = merge_dicts(self.batch_wandb_d, self.batch_d)
        not_for_wandb_keys = ['orig', 'label','orig_truelabel_probs', 'pp', 'loss', 'pp_logp','ref_logp','diff_logp',
                              'reward_pp_minus_baseline_with_penalty', 'reward_pp_minus_baseline',
                              'reward_pp', 'sts_scores', 'vm_scores', 'pp_letter_diff', 'pp_letter_percent',
                              'contradiction_scores', 'acceptability_scores', 'lcp_conditions',
                              'pp_predclass_probs', 'label_flip', 'pp_predclass', 'pp_truelabel_probs']
        for k in not_for_wandb_keys:  self.batch_wandb_d.pop(k, None)
        wandb.log(self.batch_wandb_d, commit=True)

    def _loss_fn(self, data, raw, pp_output, pp_l):
        with timecode() as self.batch_time_d['time_pp_logp']:        pp_logp = self._get_pp_logp(pp_output)
        with timecode() as self.batch_time_d['time_ref_logprobs']:  ref_logp = self._get_ref_logp(orig_ids=data['input_ids'], pp_ids=pp_output.sequences)
        diff_logp =  pp_logp - ref_logp  # One score per batch item here. Sometimes negative.
        kl_div = torch.clip(torch.mean(diff_logp), min=0) # Get KL div by taking mean of logprob differences. Then we clip it so it has min 0 (just in case).
        with timecode() as self.batch_time_d['time_loss_fn_loss_calc']:
            if   self._cfg.reward_penalty_type == "kl_div":   reward_penalty = - (self._cfg.kl_coef       * kl_div)   # both kl_div and kl_coef >=0 so this term is negative
            elif self._cfg.reward_penalty_type == "ref_logp": reward_penalty =    self._cfg.ref_logp_coef * torch.mean(ref_logp)      # ref_logp is negative, ref_logp_coef >= 0, so this term is negative
        with timecode() as self.batch_time_d['time_reward_fn']: reward_pp = self._reward_fn(data, raw, pp_l)
        baselines = torch.tensor([self.orig_baselines[idx] for idx in raw['idx']], device=self._cfg.device)  # this term >= 0

        # Calculations
        reward_pp_minus_baseline = torch.clip(reward_pp - baselines, min=0)
        reward_pp_minus_baseline_with_penalty = torch.clip(reward_pp_minus_baseline + reward_penalty, min=0)
        loss       = -reward_pp_minus_baseline_with_penalty * pp_logp
        loss_sum   = torch.sum(loss)  # we scale it later
        loss_batch = loss_sum / self.acc_current_n_examples  # for gradient accumulation

        self.batch_d['pp_logp']     =                      pp_logp.detach().cpu().tolist()
        self.batch_d['ref_logp']    =                     ref_logp.detach().cpu().tolist()
        self.batch_d['diff_logp'] =                      diff_logp.detach().cpu().tolist()
        self.batch_d['kl_div']      =                       kl_div.detach().cpu().tolist()
        self.batch_d['reward_pp_minus_baseline'] = reward_pp_minus_baseline.detach().cpu().tolist()
        self.batch_d['reward_penalty'] =            reward_penalty.detach().cpu().tolist()
        self.batch_d['reward_pp_minus_baseline_with_penalty'] =  reward_pp_minus_baseline_with_penalty.detach().cpu().tolist()
        self.batch_d['loss']       =                          loss.detach().cpu().tolist()
        self.batch_d['loss_sum']   =                      loss_sum.detach().cpu().tolist()
        self.batch_d['loss_batch']   =                  loss_batch.detach().cpu().tolist()
        return loss_batch

    def _get_vm_scores(self, pp_l, labels, orig_truelabel_probs):
        """Victim model probability differences between orig and pp"""
        pp_probs = get_vm_probs(pp_l, self._cfg, self.vm_tokenizer, self.vm_model, return_predclass=False)
        pp_predclass = torch.argmax(pp_probs, axis=1)
        pp_truelabel_probs   = torch.gather(pp_probs, 1, labels[:,None]).squeeze()
        pp_predclass_probs   = torch.gather(pp_probs, 1, pp_predclass[ :,None]).squeeze()
        label_flip = ((pp_predclass != labels) * 1)
        vm_scores = (orig_truelabel_probs - pp_truelabel_probs)
        return dict(pp_truelabel_probs=pp_truelabel_probs, pp_predclass=pp_predclass, pp_predclass_probs=pp_predclass_probs, vm_scores=vm_scores, label_flip=label_flip)

    def _get_sts_scores(self, pp_l, orig_sts_embeddings, eval_mode=False):
        """Calculate STS scores when there is one orig and a list of pp_l"""
        if not orig_sts_embeddings.is_cuda: orig_sts_embeddings = orig_sts_embeddings.to(self._cfg.device)
        pp_sts_embeddings = self.sts_model.encode(pp_l, convert_to_tensor=True, device=self._cfg.device, show_progress_bar=False)
        # pytorch_cos_sim returns a cosine similarity matrix if we have one paraphrse, with which we want the diagonal
        # else it returns a list.
        if eval_mode:            sts_scores = pytorch_cos_sim(orig_sts_embeddings, pp_sts_embeddings).cpu().tolist()    # eval case
        else:                    sts_scores = pytorch_cos_sim(orig_sts_embeddings, pp_sts_embeddings).diagonal()        # training case
        return sts_scores

    def _get_pp_letter_diff(self, pp_l, orig_n_letters):
        pp_n_letters = np.array([len(o) for o in pp_l])
        orig_letters = np.array(orig_n_letters)
        pp_letter_diff    = orig_n_letters - pp_n_letters
        pp_letter_percent = pp_n_letters / orig_n_letters
        return dict(pp_letter_diff=pp_letter_diff, pp_letter_percent=pp_letter_percent)

    def _get_contradiction_scores(self, orig_l, pp_l):
        contradiction_scores = get_nli_probs(orig_l, pp_l, self._cfg, self.nli_tokenizer, self.nli_model)[:, self._cfg.contra_label]
        return contradiction_scores

    def _get_acceptability_scores(self, pp_l):
        acceptability_scores = get_cola_probs(pp_l, self._cfg, self.cola_tokenizer, self.cola_model)[:, self._cfg.cola_positive_label]   # acceptable class
        return acceptability_scores

    def _get_linking_contrast_phrase_conditions(self, orig_l, pp_l):
        """True: ok, False: fail. Logic: it's ok to include a linking contrast phrase if there is
        one in the original to start with, but not if there isn't """
        def clean_sen_l(sen_l): return [sen.strip(string.punctuation).strip().lower() for sen in sen_l]
        def has_linking_contrast_phrase(sen):
            return any([sen.startswith(phrase + " ") or sen.endswith(" " + phrase) for phrase in self.linking_contrast_phrases])
        orig_l_cleaned,pp_l_cleaned = clean_sen_l(orig_l),clean_sen_l(pp_l)
        phrase_present_orig_l = [has_linking_contrast_phrase(sen=orig) for orig in orig_l_cleaned]
        phrase_present_pp_l   = [has_linking_contrast_phrase(sen=pp)   for pp   in pp_l_cleaned]
        return [True if phrase_present_orig else not phrase_present_pp
            for phrase_present_orig, phrase_present_pp in zip(phrase_present_orig_l, phrase_present_pp_l)]


    def _is_valid_pp(self, sts_score, pp_letter_diff, contradiction_score, acceptability_score, lcp_condition):
        if sts_score           < self._cfg.sts_threshold:                              return False
        if acceptability_score < self._cfg.acceptability_threshold:                    return False
        if contradiction_score > self._cfg.contradiction_threshold:                    return False
        if pp_letter_diff >   self._cfg.pp_letter_diff_threshold:                      return False
        if pp_letter_diff < - self._cfg.pp_letter_diff_threshold:                      return False
        if not lcp_condition:                                                          return False
        return True

    def _get_reward(self, vm_scores, sts_scores, pp_letter_diff, contradiction_scores, acceptability_scores, lcp_conditions):
        def reward_fn_contradiction_and_letter_diff(vm_score, sts_score, pp_letter_diff, contradiction_score, acceptability_score, lcp_condition):
            if not self._is_valid_pp(sts_score, pp_letter_diff, contradiction_score, acceptability_score, lcp_condition): return 0.
            reward = self._cfg.reward_base + vm_score * self._cfg.reward_vm_multiplier
            return min(max(self._cfg.reward_clip_min, reward), self._cfg.reward_clip_max)

        def calc_reward(vm_scores, sts_scores, pp_letter_diff, contradiction_scores, acceptability_scores, lcp_conditions):
            if self._cfg.reward_fn == "reward_fn_contradiction_and_letter_diff": reward_fn = reward_fn_contradiction_and_letter_diff
            return torch.tensor([
                reward_fn(vm, sts, ldiff, contra, acpt, lcp) for vm,sts,ldiff,contra,acpt,lcp
                    in zip(vm_scores, sts_scores, pp_letter_diff, contradiction_scores, acceptability_scores, lcp_conditions)
            ], device=self._cfg.device)
        rewards = calc_reward(vm_scores, sts_scores, pp_letter_diff, contradiction_scores, acceptability_scores, lcp_conditions)
        return rewards

    def _reward_fn(self, data, raw, pp_l):
        """Used for training, need 1-1 """
        with timecode() as self.batch_time_d['time_vm_scores']:
            vm_d = self._get_vm_scores(pp_l, data['label'], data['orig_truelabel_probs'])
            vm_scores = vm_d['vm_scores']
        with timecode() as self.batch_time_d['time_sts_scores']: sts_scores = self._get_sts_scores(pp_l, data['orig_sts_embeddings'], eval_mode=False)
        with timecode() as self.batch_time_d['time_pp_letter_diff']:
            pp_diff_d = self._get_pp_letter_diff(pp_l, data['n_letters'].cpu().tolist())
            pp_letter_diff = pp_diff_d['pp_letter_diff']
        with timecode() as self.batch_time_d['time_contradiction_scores']:  contradiction_scores = self._get_contradiction_scores(raw['text'], pp_l)
        with timecode() as self.batch_time_d['time_acceptability_scores']:  acceptability_scores = self._get_acceptability_scores(pp_l)
        with timecode() as self.batch_time_d['time_lcp_conditions']      :  lcp_conditions = self._get_linking_contrast_phrase_conditions(raw['text'], pp_l)
        rewards = self._get_reward(vm_scores, sts_scores, pp_letter_diff, contradiction_scores, acceptability_scores, lcp_conditions)

        self.batch_d['pp_truelabel_probs']  = vm_d['pp_truelabel_probs'].detach().cpu().tolist()
        self.batch_d['pp_predclass']        = vm_d['pp_predclass'].detach().cpu().tolist()
        self.batch_d['pp_predclass_probs']  = vm_d['pp_predclass_probs'].detach().cpu().tolist()
        self.batch_d['label_flip']          = vm_d['label_flip'].detach().cpu().tolist()
        self.batch_d['label_flip_fraction'] = np.mean(self.batch_d['label_flip'])
        self.batch_d['reward_pp']              = rewards.detach().cpu().tolist()
        self.batch_d['vm_scores']            = vm_scores.detach().cpu().tolist()
        self.batch_d['sts_scores']           = sts_scores.detach().cpu().tolist()
        self.batch_d['pp_letter_diff']      = pp_letter_diff.tolist()
        self.batch_d['pp_letter_percent']   = pp_diff_d['pp_letter_percent'].tolist()
        self.batch_d['contradiction_scores'] = contradiction_scores.cpu().tolist()
        self.batch_d['acceptability_scores'] = acceptability_scores.cpu().tolist()
        self.batch_d['lcp_conditions']      = lcp_conditions
        return rewards

    def _get_pp_logp(self, pp_output):
        """log(p(pp|orig)) basically.
        works for greedy search, will need tweaking for other types probably"""
        ### We want to align tokens with token probabilities. The first token is given at the start
        # and has no probability attached to it, so we remove it.
        seq_without_first_tkn = pp_output.sequences[:, 1:]
        assert seq_without_first_tkn.shape == torch.Size([self.orig_batch_size, self.pp_length - 1])
        ### Convert from tuple of scores to one big tensor of scores
        scores_stacked = torch.stack(pp_output.scores, 1)
        ### TESTS
        # We check shape and that there is no +inf or nan in scores.
        # Scores can have -inf in them - see explanation in `exploring_generation`.
        assert scores_stacked.shape == torch.Size([self.orig_batch_size, (self.pp_length - 1), self._cfg.vocab_size])
        assert torch.all(~torch.isnan(scores_stacked))
        assert torch.all(~torch.isposinf(scores_stacked))
        ### Take log softmax of scores and then extract those that correspond
        # to the generated sequences
        scores_log_softmax = scores_stacked.log_softmax(2)
        seq_token_log_probs = torch.gather(scores_log_softmax,2,seq_without_first_tkn[:,:,None]).squeeze(-1)

        ### TESTS
        # -inf is possible in scores_log_softmax and seq_token_log_probs before the attention mask is added.
        assert torch.all(~torch.isnan(   scores_log_softmax))
        assert torch.all(~torch.isposinf(scores_log_softmax))
        def _check_scores_log_softmax_sums(scores_log_softmax):
            sums = scores_log_softmax.exp().sum(2)
            # check that the axes is right
            # we want to sum over token probabilities at each generation step, so we
            # should end up with a shape [self.orig_batch_size, self.pp_length]
            assert sums.shape[0] == self.orig_batch_size
            assert sums.shape[1] == self.pp_length - 1
            # check that they sum to 1 along the self.pp_length axis (or close enough at least)
            assert torch.allclose(sums, torch.ones(sums.size(), device=self._cfg.device), atol = 5e-2)
        _check_scores_log_softmax_sums(scores_log_softmax)
        assert seq_token_log_probs.shape == seq_without_first_tkn.shape  # probs should be 1-1 with the filtered tkns: check shape to confirm

        ### Generate attention mask to identify padding tokens. Then apply it to the
        # sequence probabilities so that we don't consider probability of padding tokens
        # when getting sequence probabilities.
        # Also replace the -inf values in seq_token_log_probs with a large negative number because if we
        # leave them in we end up with nan's introduced after multiplying with attention_mask,
        # since  -inf * 0 = nan
        attention_mask = self.pp_model._prepare_attention_mask_for_generation(
            seq_without_first_tkn, self.pp_tokenizer.pad_token_id, self.pp_tokenizer.eos_token_id
        )
        seq_token_log_probs = torch.nan_to_num(seq_token_log_probs, nan=None, posinf=None, neginf=-50)
        seq_token_log_probs = seq_token_log_probs * attention_mask
        ### TESTS
        assert seq_token_log_probs.shape == attention_mask.shape == seq_token_log_probs.shape
        # check attention mask only has 0 for padding tokens and not eos tokens or anything else
        assert all(seq_without_first_tkn[attention_mask == 0] == self.pp_tokenizer.pad_token_id)
        check_no_nans_or_infs(seq_token_log_probs)
        ### Get sequence probabilities by summing up token log probabilities
        seq_log_prob = seq_token_log_probs.sum(-1)
        assert seq_log_prob.shape == torch.Size([self.pp_batch_size])
        check_no_nans_or_infs(seq_log_prob)
        # normalise for length
        logprobs_normalised = seq_log_prob / attention_mask.sum(1)  # normalise for length of generated sequence
        if self.pp_model.training:  # don't bother logging or calculate entropy, token_probs in eval mode
            if self._cfg.wandb['log_token_entropy']:
                with timecode() as self.batch_time_d['time_log_entropy']:
                    self.batch_wandb_d['ent_hist'] = self._get_entropy_hist(scores_stacked, attention_mask)
            if self._cfg.wandb['log_token_probabilities']:
                with timecode() as self.batch_time_d['time_log_token_probabilities']:
                    self.batch_wandb_d = merge_dicts(self.batch_wandb_d,
                        self._get_token_probability_metrics(scores_log_softmax, attention_mask, k=3))
        return logprobs_normalised

    def _get_ref_logp(self, orig_ids, pp_ids):
        decoder_start_token_ids = torch.tensor([self.ref_pp_model.config.decoder_start_token_id], device=self._cfg.device).repeat(self.orig_batch_size, 1)
        pp_ids = torch.cat([decoder_start_token_ids, pp_ids], 1)
        logprobs = []
        for i in range(pp_ids.shape[1] - 1):
            decoder_input_ids = pp_ids[:, 0:(i+1)]
            outputs = self.ref_pp_model(input_ids=orig_ids, decoder_input_ids=decoder_input_ids)
            token_logprobs = outputs.logits[:,i,:].log_softmax(1)
            pp_next_token_ids = pp_ids[:,i+1].unsqueeze(-1)
            pp_next_token_logprobs = torch.gather(token_logprobs, 1, pp_next_token_ids).detach().squeeze(-1)
            logprobs.append(pp_next_token_logprobs)
        logprobs = torch.stack(logprobs, 1)
        attention_mask = self.ref_pp_model._prepare_attention_mask_for_generation(pp_ids[:,1:],
                self.pp_tokenizer.pad_token_id, self.pp_tokenizer.eos_token_id)
        logprobs = logprobs * attention_mask
        logprobs_sum = logprobs.sum(1)
        logprobs_normalised = logprobs_sum / attention_mask.sum(1)  # normalise for length of generated sequence
        return logprobs_normalised

    def _get_entropy_hist(self, scores_stacked, attention_mask):
        ent = Categorical(logits = scores_stacked).entropy().detach()
        assert ent.shape == attention_mask.shape == torch.Size([self.pp_batch_size, self.pp_length - 1])
        ent = ent * attention_mask  # stop values after eos token from contributing to ent score
        # first remove structure (otherwise we have ragged arrays), then remove corresponding attention mask values
        # we can't just filter by ent[ent != 0] because we might have zero tokens during the sequence
        att_flat= attention_mask.flatten()
        indices = torch.nonzero(att_flat)
        ent_flat = ent.flatten()[indices].flatten()
        assert ent_flat.shape[0] == (torch.sum(att_flat)*1).item()
        # check everything we filter out is zero
        torch.isclose(ent.flatten()[torch.nonzero(~(att_flat > 0))].sum(), torch.tensor(0.), 1e-3)
        return Histogram(ent_flat.detach().cpu().tolist())

    def _get_token_probability_metrics(self, scores_log_softmax, attention_mask, k=3):
        token_prob_d = dict()
        tkn_kmaxprob, _ = torch.topk(scores_log_softmax, largest=True, k=k, dim=2)
        tkn_kmaxprob = tkn_kmaxprob.detach()
        tkn_kmaxprob = torch.nan_to_num(tkn_kmaxprob, nan=None, posinf=None, neginf=-20)
        assert tkn_kmaxprob.shape == torch.Size([self.pp_batch_size, self.pp_length - 1, k])

        # % of first prob over 0.9, 0.75, 0.5, 0.3, 0.1
        top_probs = tkn_kmaxprob[:,:,0].exp()
        top_probs = (top_probs * attention_mask).flatten()
        top_probs = top_probs[top_probs != 0]
        prob_threshold_l = [0.99, 0.975, 0.95, 0.90, 0.75, 0.5, 0.3, 0.1]
        for p in prob_threshold_l:
            token_prob_d[f"top_token_prob_over_{str(p)}"] = (torch.sum(top_probs > p) / top_probs.shape[0]).item()

        # avg + median + lower + upper quartile of first, second, third choice probs
        tkn_kmaxprob_mask = tkn_kmaxprob * attention_mask[:,:,None]  # broadcasting over kth dim
        for i in range(k):
            probs = tkn_kmaxprob_mask[:,:, i].flatten()
            probs = probs[probs != 0]
            token_prob_d[f"rank_{i+1}_histogram"] = Histogram(probs.detach().cpu().tolist())
            token_prob_d[f"rank_{i+1}_token_prob_mean"] = probs.mean().item()
            token_prob_d[f"rank_{i+1}_token_prob_median"] = probs.median().item()
            token_prob_d[f"rank_{i+1}_token_prob_0.25_quantile"] = probs.quantile(0.25).item()
            token_prob_d[f"rank_{i+1}_token_prob_0.75_quantile"] = probs.quantile(0.75).item()

        # tokens over probs above 0.1, 0.01, 0.001, 0.0001, 1/vocab_size prob
        allprobs = (scores_log_softmax.detach().exp() * attention_mask[:,:,None]).flatten()
        allprobs = allprobs[allprobs != 0]
        for p in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
            token_prob_d[f"%_of_tokens_above_prob_{p}"] =  (torch.sum(allprobs > p) / allprobs.shape[0]).item()
        token_prob_d[f"%_of_tokens_above_prob_1/vocab_size"] = \
            (torch.sum(allprobs > (1/self._cfg.vocab_size)) / allprobs.shape[0]).item()
        return token_prob_d

    def _eval_function(self, split):
        ## Setup
        if self.pp_model.training:     self.pp_model.eval()
        if self.vm_model.training:     self.vm_model.eval()
        dl_key = "train_eval" if split == "train" else split
        dl_raw = self.ds.dld_raw[dl_key]
        dl_tkn = self.ds.dld_tkn[dl_key]
        eval_batch_results = list()  # each eval batch appended to here, list of dicts
        ## Loop through batches in eval set
        for eval_batch_num, (data, raw) in enumerate(zip(dl_tkn, dl_raw)):
            pp_output = self.pp_model.generate(input_ids=data['input_ids'].to(self._cfg.device), attention_mask=data['attention_mask'].to(self._cfg.device),
                                          **self._cfg.gen_params_eval,   min_length=self._cfg.min_pp_length, max_length=self._cfg.max_pp_length,
                                        remove_invalid_values=False,
                                          pad_token_id = self.pp_tokenizer.pad_token_id,eos_token_id = self.pp_tokenizer.eos_token_id)
            pp_l = self.pp_tokenizer.batch_decode(pp_output, skip_special_tokens=True)
            pp_l_nested = [pp_l[i:i+self._cfg.n_eval_seq] for i in range(0, len(pp_l), self._cfg.n_eval_seq)]  # put paraphrases in nested lists
            assert all([len(l) == self._cfg.n_eval_seq for l in pp_l_nested])  # make sure we generate the same number of paraphrases for each
            eval_batch_results.append({'idx': raw['idx'], 'orig': raw['text'], 'pp':pp_l_nested, 'orig_n_letters': data['n_letters'].tolist(),
                                  'label': raw['label'], 'orig_truelabel_probs': data['orig_truelabel_probs'].tolist(), 'orig_sts_embeddings': data['orig_sts_embeddings'] })

        ## Convert eval batches to dataframes and create paraphrase identifier `pp_idx`
        df = pd.DataFrame(eval_batch_results)
        df = df.apply(pd.Series.explode).reset_index(drop=True)  # This dataframe has one row per original example
        def get_pp_idx(row): return ["orig_" + str(row['idx']) + "-epoch_" + str(self.epoch) +  "-pp_" +  str(pp_i) for pp_i in range(1, len(row['pp'])+1)]
        df['pp_idx'] = df.apply(get_pp_idx, axis=1)

        ## Create seperate dataframe for sts scores and expand original dataframe
        df_sts = df[['pp_idx', 'pp', 'orig_sts_embeddings']]
        df1 = df.drop(columns='orig_sts_embeddings')
        scalar_cols = [o for o in df1.columns if o not in ['pp', 'pp_idx']]
        df_expanded = unpack_nested_lists_in_df(df1, scalar_cols=scalar_cols) # This dataframe has one row per paraphrase

        ## Add reward component scores
        ds_expanded = Dataset.from_pandas(df_expanded)
        def add_vm_scores_eval(batch):
            output = self._get_vm_scores(pp_l=batch['pp'], labels=torch.tensor(batch['label'], device = self._cfg.device),
                                            orig_truelabel_probs=torch.tensor(batch['orig_truelabel_probs'], device=self._cfg.device))
            for k, v in output.items(): batch[k] = v.cpu().tolist()
            return batch
        def add_pp_letter_diff(batch):
            output = self._get_pp_letter_diff(pp_l=batch['pp'], orig_n_letters=batch['orig_n_letters'])
            for k, v in output.items(): batch[k] = v.tolist()
            return batch
        def add_contradiction_score(batch):
            batch['contradiction_scores'] = self._get_contradiction_scores(orig_l=batch['orig'], pp_l=batch['pp']).cpu().tolist()
            return batch
        def add_acceptability_score(batch):
            batch['acceptability_scores'] = self._get_acceptability_scores(pp_l=batch['pp']).cpu().tolist()
            return batch
        def add_lcp_condition(batch):
            batch['lcp_conditions'] = self._get_linking_contrast_phrase_conditions(orig_l=batch['orig'], pp_l=batch['pp'])
            return batch

        ds_expanded = ds_expanded.map(add_vm_scores_eval,        batched=True)
        ds_expanded = ds_expanded.map(add_pp_letter_diff,        batched=True)
        ds_expanded = ds_expanded.map(add_contradiction_score,   batched=True)
        ds_expanded = ds_expanded.map(add_acceptability_score,   batched=True)
        ds_expanded = ds_expanded.map(add_lcp_condition,         batched=True)
        def add_sts_scores_eval(row):  return self._get_sts_scores(row['pp'], row['orig_sts_embeddings'], eval_mode=True)[0]
        df_sts['sts_scores'] = df_sts.apply(add_sts_scores_eval, axis=1)

        ## Merge together results
        df_sts = df_sts.drop(columns = ['pp','orig_sts_embeddings'])
        df_sts_expanded = df_sts.apply(pd.Series.explode).reset_index(drop=True)
        ds_expanded = Dataset.from_pandas(ds_expanded.to_pandas().merge(df_sts_expanded, how='left', on='pp_idx').reset_index(drop=True))

        ## Calculate rewards and identify adversarial examples
        def add_reward(batch):
            batch['reward_pp'] = self._get_reward(vm_scores=batch['vm_scores'], sts_scores=batch['sts_scores'],
                    pp_letter_diff=batch['pp_letter_diff'], contradiction_scores=batch['contradiction_scores'],
                    acceptability_scores=batch['acceptability_scores'], lcp_conditions=batch['lcp_conditions']).cpu().tolist()
            return batch
        ds_expanded = ds_expanded.map(add_reward,   batched=True)
        def add_is_valid_pp(example):
            example['is_valid_pp'] = self._is_valid_pp(sts_score=example['sts_scores'], pp_letter_diff=example['pp_letter_diff'],
                contradiction_score=example['contradiction_scores'], acceptability_score=example['acceptability_scores'],
                lcp_condition=example['lcp_conditions'])*1
            return example
        ds_expanded = ds_expanded.map(add_is_valid_pp,   batched=False)
        def add_is_adv_example(batch):
            batch['is_adv_example'] = (np.array(batch['is_valid_pp']) * np.array(batch['label_flip'])).tolist()
            return batch
        ds_expanded = ds_expanded.map(add_is_adv_example,   batched=True)

        ### Calculate summary statistics
        df_expanded = ds_expanded.to_pandas()
        # Remove duplicate paraphrases
        df_expanded = df_expanded.drop_duplicates(subset=df_expanded.columns.difference(['pp_idx', 'sts_scores'])) # sts scores sometimes has rounding errors
        eval_metric_cols = ['label_flip', 'is_valid_pp', 'is_adv_example', 'reward_pp', 'vm_scores', 'sts_scores',  'pp_letter_diff',
                            'contradiction_scores', 'acceptability_scores', 'lcp_conditions']
        agg_metrics = ['mean','std']  # using mean in favour of median
        ## avg across each orig
        df_grp_stats = df_expanded[['idx'] + eval_metric_cols].groupby('idx').agg(agg_metrics).fillna(0) # if there is one example, std is NaN, so we just replace with 0
        df_grp_stats.columns = df_grp_stats.columns = ["-".join(a) for a in df_grp_stats.columns.to_flat_index()]
        df_grp_stats = df_grp_stats.merge(df_expanded.groupby('idx').size().rename('n_pp').to_frame(), how='left', left_index=True, right_index=True)
        # For training set save mean reward as the REINFORCE baseline
        if split == "train": self.orig_baselines = df_grp_stats['reward_pp-mean'].to_dict()  # idx is extracted as the index automatically

        # average orig-level stats to get whole dataset stats
        df_overall_stats = df_grp_stats.groupby(lambda _: True).agg('mean').reset_index(drop=True)
        overall_metrics_d = df_overall_stats.iloc[0].to_dict()
        overall_metrics_d['any_adv_example_proportion'] = np.mean((df_grp_stats['is_adv_example-mean'] > 0 ) * 1)
        overall_metrics_d_split = {f"{k}-{split}" : v  for k,v in overall_metrics_d.items()}

        # Save baseline metric values if it is the first epoch (for comparison later)
        if self.epoch == 0: self.initial_metric_d[split] = copy.deepcopy(overall_metrics_d)
        # add keys
        df_expanded['epoch'] = self.epoch
        overall_metrics_d['epoch'] = self.epoch
        overall_metrics_d_split['epoch'] = self.epoch

        # Track eval metrics, update the "best" model if needed, check early stopping
        if split == "valid":
            this_metric = overall_metrics_d[self._cfg.early_stopping_metric]
            self.eval_valid_metrics.append(this_metric)
            logger.info(f"Epoch: {self.epoch}. Min epochs before early stopping activated: {self._cfg.early_stopping_min_epochs}")
            logger.info(f"Eval metric: {this_metric:.3f} | Running median: {np.median(self.eval_valid_metrics):.3f}")

            if this_metric > self.best_eval_valid_metric:
                self.best_eval_valid_metric = this_metric
                self.best_eval_valid_epoch  = self.epoch
                # Save model and delete previous best model
                path = f"{self._cfg.path_run}model_{self.epoch}.pt"
                save_pp_model(self.pp_model, self.optimizer, path)
                if os.path.exists(self.best_model_path): os.remove(self.best_model_path)
                self.best_model_path = path

            if self.epoch > self._cfg.early_stopping_min_epochs:  # don't test for early stopping too early because otherwise it stops too soon
                if this_metric <= np.median(self.eval_valid_metrics):
                    logger.info(f"Early stopping activated.")
                    self.early_stopping_flag = True

        def save_eval_stats_to_csv(split, overall_metrics_d):
            overall_metrics_d['split'] = split
            ref_model_keys = [
                'datetime_run', 'run_name', 'dataset_name', 'epoch',  'seed',
                'decode_method_train', 'decode_method_eval', 'gen_params_train', 'gen_params_eval',
                'reward_fn','reward_clip_max', 'reward_clip_min', 'reward_base', 'reward_vm_multiplier',
                'sts_threshold', 'contradiction_threshold', 'pp_letter_diff_threshold',
                'pp_name', 'sts_name','nli_name', 'vm_name', 'orig_max_length', 'use_small_ds',
                'batch_size_train', 'batch_size_eval', 'lr', 'acc_steps'
            ]
            d = vars(self._cfg)
            ref_model_d = dict((k, d[k]) for k in ref_model_keys if k in d)
            results = merge_dicts(ref_model_d, overall_metrics_d)
            results_df = pd.json_normalize(results, sep='.')  # flatten nested dict
            results_df.insert(3, 'split', results_df.pop('split'))
            results_df.insert(4, 'epoch', results_df.pop('epoch'))
            results_df.insert(1, 'run_name', results_df.pop('run_name'))
            append_df_to_csv(results_df, f"{self._cfg.path_results}run_results_final.csv")

        ## For train and eval we calc + log histograms to wandb. We dont need that for test.
        if split in ['train', 'valid']:
            ## Log histograms to wandb
            wandb_eval_d = dict()
            mean_only = ['label_flip', 'is_valid_pp', 'is_adv_example', 'lcp_conditions']
            mean_and_std = ['reward_pp', 'vm_scores', 'sts_scores', 'pp_letter_diff', 'contradiction_scores', 'acceptability_scores']
            for k in mean_only + mean_and_std:
                name = k + "-mean"
                wandb_eval_d[name + "-"+ split + "-hist"] = Histogram(df_grp_stats[name].tolist())
            for k in mean_and_std:
                name = k + "-std"
                wandb_eval_d[name + "-" + split + "-hist"] = Histogram(df_grp_stats[name].tolist())
            wandb_eval_d['n_pp-' + split + '-hist'] = Histogram(df_grp_stats['n_pp'].tolist())
            wandb_eval_d = merge_dicts(overall_metrics_d_split, wandb_eval_d)
            wandb.log(wandb_eval_d, commit=True)
        elif split == "test":
            wandb.log(overall_metrics_d_split, commit=True)

        ## Save eval stats to CSV
        save_eval_stats_to_csv(split, overall_metrics_d)

        ## Save paraphrase-level dataframe to csv
        df_expanded = self._set_df_colorder(df_expanded, is_eval=True)
        fname = f"{self._cfg.path_run}{split}.csv"
        append_df_to_csv(df_expanded, path = fname)

    def _set_df_colorder(self, df, is_eval=False):
        if is_eval:
            colorder_eval = [
               'idx', 'epoch', 'orig', 'pp', 'pp_idx', 'is_adv_example', 'is_valid_pp', 'label_flip', 'reward_pp',
                'vm_scores','sts_scores', 'contradiction_scores', 'acceptability_scores', 'pp_letter_diff',
                'label', 'orig_truelabel_probs', 'pp_truelabel_probs', 'pp_predclass', 'pp_predclass_probs',
                'orig_n_letters','pp_letter_percent'
            ]
            df = df[colorder_eval]
        else:
            colorder_train=[
                'idx','epoch', 'orig',  'pp','orig_truelabel_probs','pp_truelabel_probs',
                'pp_predclass_probs','label','pp_predclass','label_flip', 'vm_scores','sts_scores',
                'pp_letter_diff', 'pp_letter_percent',  'contradiction_scores', 'acceptability_scores', 'pp_logp','ref_logp', 'diff_logp',
                'kl_div', 'reward_pp', 'reward_pp_minus_baseline', 'reward_penalty', 'reward_pp_minus_baseline_with_penalty',
                'loss','batch_num',
                'global_step','acc_num','loss_sum', 'loss_batch', 'label_flip_fraction',
                'orig_length','orig_batch_size','pp_length','pp_batch_size'
            ]
            colorder_train= colorder_train + [o for o in df.columns if 'time_' in o]
            assert len(set(colorder_train).difference(set(df.columns))) == 0
            df = df[colorder_train]
        return df

    def _update_wandb_summary(self):
        for split in ['train', 'valid', 'test']:
            self.run.summary['baseline_'+ split] = self.initial_metric_d[split]
            self.run.summary['change_'  + split] = dict()
            for k,initial_value in self.initial_metric_d[split].items():
                self.run.summary['change_'+ split][k] = self.run.summary[k + "-" + split] - initial_value

    def _plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.'''
        plt.figure()
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        return plt