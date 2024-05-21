__all__ = ['setup_baselines_parser', 'StsScoreConstraint', 'ContradictionScoreConstraint',
           'AcceptabilityScoreConstraint', 'PpLetterDiffConstraint', 'LCPConstraint', 'AttackRecipes']


import functools, string, nltk, transformers
from functools import partial
import argparse


from sentence_transformers.util import pytorch_cos_sim
from textattack.search_methods import GreedySearch
from textattack.constraints import Constraint
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
from textattack import Attack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import AttackRecipe
from textattack.search_methods import BeamSearch, ImprovedGeneticAlgorithm
from textattack.constraints import Constraint
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import CompositeTransformation, WordSwapEmbedding, WordSwapMaskedLM
from textattack.goal_functions import UntargetedClassification

from textattack.search_methods import GreedyWordSwapWIR

from textattack.transformations.word_insertions import WordInsertionMaskedLM
from textattack.transformations.word_merges import  WordMergeMaskedLM

from fastcore.basics import store_attr


from .models import prepare_models, get_nli_probs, get_cola_probs
from .data import  ProcessedDataset

from .config import Config



def setup_baselines_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name")
    parser.add_argument("--split")
    parser.add_argument("--attack_name")
    parser.add_argument("--vm_name", required=True)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--beam_sz", type=int)
    parser.add_argument("--max_candidates", type=int)
    parser.add_argument("--sts_threshold", type=float)
    parser.add_argument("--contradiction_threshold", type=float)
    #parser.add_argument('args', nargs=argparse.REMAINDER)  # activate to put keywords in kwargs.
    return parser


class StsScoreConstraint(Constraint):
    def __init__(self, sts_model, sts_threshold):
        super().__init__(True)  # need the true here to compare against original (as opposed to previous x') I think
        self.sts_threshold = sts_threshold
        self.sts_model     = sts_model

    @functools.lru_cache(maxsize=2**14)
    def get_embedding(self, text):  return self.sts_model.encode(text)

    def _check_constraint(self, transformed_text, current_text):
        orig_embedding = self.get_embedding(current_text.text)
        pp_embedding   = self.get_embedding(transformed_text.text)
        sts_score = pytorch_cos_sim(orig_embedding, pp_embedding).item()
        if sts_score > self.sts_threshold:   return True
        else:                                return False


class ContradictionScoreConstraint(Constraint):
    def __init__(self, cfg, nli_tokenizer, nli_model, contradiction_threshold):
        super().__init__(True)
        self.cfg = cfg
        self.nli_tokenizer = nli_tokenizer
        self.nli_model     = nli_model
        self.contradiction_threshold = contradiction_threshold

    def _check_constraint(self, transformed_text, current_text):
        orig =     current_text.text
        pp   = transformed_text.text
        contradiction_score = get_nli_probs(orig, pp, self.cfg, self.nli_tokenizer, self.nli_model).cpu()[0][self.cfg.contra_label].item()
        if contradiction_score < self.contradiction_threshold:   return True
        else:                                                    return False


class AcceptabilityScoreConstraint(Constraint):
    def __init__(self, cfg, cola_tokenizer, cola_model, acceptability_threshold):
        super().__init__(True)
        self.cfg = cfg
        self.cola_tokenizer = cola_tokenizer
        self.cola_model     = cola_model
        self.acceptability_threshold = acceptability_threshold

    def _check_constraint(self, transformed_text, current_text):
        pp = transformed_text.text
        acceptability_score = get_cola_probs(pp, self.cfg, self.cola_tokenizer, self.cola_model)[0, self.cfg.cola_positive_label].cpu().item()
        if acceptability_score > self.acceptability_threshold:  return True
        else:                                                   return False


class PpLetterDiffConstraint(Constraint):
    def __init__(self, pp_letter_diff_threshold):
        super().__init__(True)
        self.pp_letter_diff_threshold = pp_letter_diff_threshold

    def _check_constraint(self, transformed_text, current_text):
        orig =     current_text.text
        pp   = transformed_text.text
        return abs(len(orig) - len(pp)) < self.pp_letter_diff_threshold


class LCPConstraint(Constraint):
    def __init__(self, linking_contrast_phrases):
        super().__init__(True)
        self.linking_contrast_phrases = linking_contrast_phrases

    def _get_linking_contrast_phrase_conditions(self, orig_l, pp_l):
        """True: ok, False: fail. Logic: it's ok to include a linking contrast phrase if there is
        one in the original to start with, but not if there isn't.
        Copied from trainer class"""
        def clean_sen_l(sen_l): return [sen.strip(string.punctuation).strip().lower() for sen in sen_l]
        def has_linking_contrast_phrase(sen):
            return any([sen.startswith(phrase + " ") or sen.endswith(" " + phrase) for phrase in self.linking_contrast_phrases])
        orig_l_cleaned,pp_l_cleaned = clean_sen_l(orig_l),clean_sen_l(pp_l)
        phrase_present_orig_l = [has_linking_contrast_phrase(sen=orig) for orig in orig_l_cleaned]
        phrase_present_pp_l   = [has_linking_contrast_phrase(sen=pp)   for pp   in pp_l_cleaned]
        return [True if phrase_present_orig else not phrase_present_pp
            for phrase_present_orig, phrase_present_pp in zip(phrase_present_orig_l, phrase_present_pp_l)]

    def _check_constraint(self, transformed_text, current_text):
        orig =     current_text.text
        pp   = transformed_text.text
        condition_met = self._get_linking_contrast_phrase_conditions([orig], [pp])[0]
        return condition_met


class AttackRecipes:
    def __init__(self, param_d):
        store_attr()
        if   param_d['ds_name'] == "financial":              cfg = Config().adjust_config_for_financial_dataset()
        elif param_d['ds_name'] == "rotten_tomatoes":        cfg = Config().adjust_config_for_rotten_tomatoes_dataset()
        elif param_d['ds_name'] == "trec":                   cfg = Config().adjust_config_for_trec_dataset()
        elif param_d['ds_name'] == "simple":                 cfg = Config().adjust_config_for_simple_dataset()
        cfg.vm_name = param_d['vm_name']
        vm_tokenizer,vm_model,pp_tokenizer,_,_,sts_model,nli_tokenizer,nli_model,cola_tokenizer,cola_model,cfg = prepare_models(cfg)
        vm_tokenizer, vm_model, pp_tokenizer, _, _, sts_model, nli_tokenizer, nli_model, cola_tokenizer, cola_model, cfg
        self.ds = ProcessedDataset(cfg, vm_tokenizer, vm_model, pp_tokenizer, sts_model, load_processed_from_file=False)
        self.model_wrapper = HuggingFaceModelWrapper(vm_model, vm_tokenizer)
        self.goal_function =  UntargetedClassification(self.model_wrapper)
        self.constraints = [
             RepeatModification(),
             StopwordModification(nltk.corpus.stopwords.words("english")),
             StsScoreConstraint(sts_model, param_d['sts_threshold']),
             ContradictionScoreConstraint(cfg, nli_tokenizer,  nli_model,  param_d['contradiction_threshold']),
             AcceptabilityScoreConstraint(cfg, cola_tokenizer, cola_model, param_d['acceptability_threshold']),
             PpLetterDiffConstraint(param_d['pp_letter_diff_threshold']),
             LCPConstraint(linking_contrast_phrases=[o.strip() for o in open("./linking_contrast_phrases.txt").readlines()])
        ]
        self.masked_lm           = transformers.AutoModelForCausalLM.from_pretrained("distilroberta-base")
        self.masked_lm_tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
        self.WordSwapLM = partial(WordSwapMaskedLM, method="bae", masked_language_model=self.masked_lm,tokenizer=self.masked_lm_tokenizer)

    class CFEmbeddingWordReplaceBeamSearchAttack(AttackRecipe):
        @staticmethod
        def build(common_class, beam_sz, max_candidates):
            transformation = WordSwapEmbedding(max_candidates=max_candidates)
            search_method = BeamSearch(beam_width=beam_sz)
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    class LMWordReplaceBeamSearchAttack(AttackRecipe):
        @staticmethod
        def build(common_class, beam_sz, max_candidates):
            transformation = common_class.WordSwapLM(max_candidates=max_candidates)
            search_method = BeamSearch(beam_width=beam_sz)
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    class LMWordAddDeleteReplaceBeamSearchAttack(AttackRecipe):
        @staticmethod
        def build(common_class, beam_sz, max_candidates):
            transformation = CompositeTransformation(
                [common_class.WordSwapLM(max_candidates=max_candidates),
                 WordInsertionMaskedLM(        masked_language_model=common_class.masked_lm,tokenizer=common_class.masked_lm_tokenizer,max_candidates=max_candidates),
                 WordMergeMaskedLM(            masked_language_model=common_class.masked_lm,tokenizer=common_class.masked_lm_tokenizer,max_candidates=max_candidates)]
            )
            search_method = BeamSearch(beam_width=beam_sz)
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    class LMWordReplaceGeneticAlgorithmAttack(AttackRecipe):
        @staticmethod
        def build(common_class, max_candidates, pop_size, max_iters, max_replace_times_per_index):
            transformation = common_class.WordSwapLM(max_candidates=max_candidates)
            search_method = ImprovedGeneticAlgorithm(pop_size=pop_size, max_iters=max_iters,
                                                     max_replace_times_per_index=max_replace_times_per_index, post_crossover_check=False)
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    class CLARE_mod(AttackRecipe): 
        @staticmethod
        def build(common_class):
            shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained("distilroberta-base")
            shared_tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
            transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-4,
                ),
                WordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                ),
                WordMergeMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-3,
                )
            ])
            search_method = GreedySearch()
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack
 
    class TextFooler_mod(AttackRecipe): 
        @staticmethod
        def build(common_class):
            transformation = WordSwapEmbedding(max_candidates=50)
            search_method = GreedyWordSwapWIR(wir_method="delete")
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    class IGA_mod(AttackRecipe): 
        @staticmethod
        def build(common_class):
            transformation = WordSwapEmbedding(max_candidates=50)
            search_method = ImprovedGeneticAlgorithm(
                pop_size=60,
                max_iters=20,
                max_replace_times_per_index=5,
                post_crossover_check=False
            )
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    class BAE_mod(AttackRecipe): 
        @staticmethod
        def build(common_class):
            transformation = WordSwapMaskedLM(
                method="bae", max_candidates=50, min_confidence=0.0
            )
            search_method = GreedyWordSwapWIR(wir_method="delete")
            attack = Attack(common_class.goal_function, common_class.constraints, transformation, search_method)
            return attack

    def get_attack_list(self):
        common_class = self
        attack_list = [
            {
                "attack_num": 1,
                "attack_code": "LM-WR-BS-b2m5",
                "attack_recipe": self.LMWordReplaceBeamSearchAttack.build(common_class, beam_sz=2, max_candidates=5)
            }, {
                "attack_num": 2,
                "attack_code": "LM-WR-BS-b5m25",
                "attack_recipe": self.LMWordReplaceBeamSearchAttack.build(common_class, beam_sz=5, max_candidates=25)
            }, {
                "attack_num": 3,
                "attack_code": "LM-WR-BS-b10m50",
                "attack_recipe": self.LMWordReplaceBeamSearchAttack.build(common_class, beam_sz=10, max_candidates=50)
            }, {
                "attack_num": 4,
                "attack_code": "LM-WADR-BS-b5m25",
                "attack_recipe": self.LMWordAddDeleteReplaceBeamSearchAttack.build(common_class, beam_sz=5, max_candidates=25)
            }, {
                "attack_num": 5,
                "attack_code": "CF-WR-BS-b5m25",
                "attack_recipe": self.CFEmbeddingWordReplaceBeamSearchAttack.build(common_class, beam_sz=5, max_candidates=25)
            }, {
                "attack_num": 6,
                "attack_code": "LM-WR-GA-m25p60mi20mr5",
                "attack_recipe": self.LMWordReplaceGeneticAlgorithmAttack.build(common_class, max_candidates=25, pop_size=60, max_iters=20, max_replace_times_per_index=5)
            }, 
            {
                "attack_num": 7,
                "attack_code": "TextFooler",
                "attack_recipe": self.TextFooler_mod.build(common_class)
            }, 
            {
                "attack_num": 8,
                "attack_code": "CLARE",
                "attack_recipe": self.CLARE_mod.build(common_class)
            }, 
            {
                "attack_num": 9,
                "attack_code": "IGA",
                "attack_recipe": self.IGA_mod.build(common_class)
            }, 
            {
                "attack_num": 10,
                "attack_code": "BAE-R",
                "attack_recipe": self.BAE_mod.build(common_class)
            }

        ]
        return attack_list