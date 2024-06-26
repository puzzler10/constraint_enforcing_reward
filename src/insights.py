__all__ = ['logger', 'nlp', 'get_training_dfs', 'postprocess_df', 'create_and_log_wandb_postrun_plots']


import  wandb, spacy, textstat, psutil, pandas as pd, editdistance, plotly.express as px
import functools, operator, string
from datasets import Dataset
from difflib import ndiff
from lexicalrichness import LexicalRichness
from collections import defaultdict
from itertools import groupby

import logging
logger = logging.getLogger("src.insights")


nlp = spacy.load("en_core_web_sm")
#run = resume_wandb_run(cfg)


def get_training_dfs(path_run, postprocessed=False):
    """Return a dict of dataframes with all training and eval data"""
    df_d = dict()
    for key in ['training_step', 'train', 'valid', 'test']:
        try:
            if postprocessed:
                fname = f"{path_run}{key}_postprocessed.pkl"
                df_d[key] = pd.read_pickle(fname)
            else:
                fname = f"{path_run}{key}.csv"
                df_d[key] = pd.read_csv(fname)
        except FileNotFoundError:
            pass
    logger.info(f'Dataframes have shapes {[f"{k}: {df.shape}" for (k, df) in df_d.items()]}')
    return df_d

def postprocess_df(df, filter_idx=None, num_proc=min(8, psutil.cpu_count())):
    """set df to one of training_step, train, valid, test
    filter_idx - for testing (remove later) """
    df = df.sort_values(by=['idx', "epoch"], axis=0)
    if filter_idx is not None:   df = df.query("idx <= @filter_idx")  # for testing purposes
    for col in ['sts_scores','vm_scores','reward_pp', 'pp_truelabel_probs']:  df.loc[:, col] = df.loc[:, col].round(5)
    df = _add_number_of_unique_pps_per_idx(df)
    df = _add_number_of_pp_changes_per_idx(df)
    df = _add_epoch_of_first_label_flip(   df)
    df = _add_text_metrics(df, num_proc=num_proc)
    return df

def _add_number_of_unique_pps_per_idx(df):
    df_grp = df.groupby("idx").agg({"pp":"nunique"})
    df_grp= df_grp.rename(columns = {"pp":"idx_n_unique_pp"})
    df = df.merge(df_grp, left_on='idx', right_index=True, how='left')
    return df

def _add_number_of_pp_changes_per_idx(df):
    df['pp_changed'] = df.sort_values(["idx","epoch"]).groupby('idx')['pp'].shift().ne(df['pp']).astype(int)
    df_grp = df.groupby('idx').agg({'pp_changed': 'sum'})
    df_grp= df_grp.rename(columns = {"pp_changed":"idx_n_pp_changes"})
    df_grp['idx_n_pp_changes'] -= 1  # The first paraphrase isn't a change
    df = df.drop('pp_changed', axis=1) # don't need this anymore
    df = df.merge(df_grp, left_on='idx', right_index=True, how='left')
    return df

def _add_epoch_of_first_label_flip(df):
    rownum_of_first_flip = df.groupby('idx')[['epoch','label_flip']].idxmax()['label_flip'] ## works since idxmax returns first max
    df_grp = df[['idx','epoch']].loc[rownum_of_first_flip]
    df_grp= df_grp.rename(columns = {"epoch":"epoch_of_first_label_flip"})
    df = df.merge(df_grp, left_on='idx', right_on='idx', how='left')
    return df

def _add_text_metrics(df, num_proc=min(8, psutil.cpu_count())):
    df = _add_text_metrics_for_column(df, "orig", suffix="orig", num_proc=num_proc)
    df = _add_text_metrics_for_column(df, "pp",   suffix="pp",   num_proc=num_proc)
    logger.info("Calculating metric differences between orig and pp")
    for k in _get_text_metrics("some arbritary text here").keys():  df[f"{k}_diff"] = df[f"{k}_orig"] - df[f"{k}_pp"]
    df = _add_text_pair_metrics(df, num_proc=num_proc)
    return df


def _add_text_metrics_for_column(df, cname, suffix, num_proc):
    logger.info(f"Adding text metrics for column {cname}")
    ds_cname = Dataset.from_pandas(df[cname].drop_duplicates().to_frame())
    ds_cname = _get_text_metrics_for_ds(ds_cname, cname=cname, suffix=suffix, num_proc=num_proc)
    df = pd.merge(df, pd.DataFrame(ds_cname), how='left', on=[cname])
    return df

def _get_text_metrics_for_ds(ds, cname, suffix, num_proc):
    x = ds.map(_get_text_metrics, input_columns = [cname], batched=False, num_proc=num_proc)
    colnames_mapping = dict()
    for k in x.column_names: colnames_mapping[k] = k + f"_{suffix}" if k != cname else k    # rename columns
    return x.rename_columns(colnames_mapping)

def _get_text_metrics(text):
    d = defaultdict(lambda: 0)
    d['n_words'] = LexicalRichness(text).words
    d['n_sentences'] = textstat.sentence_count(text)
    def get_chartype_count(text, strset): return len(list(filter(functools.partial(operator.contains, strset), text)))
    d['n_punctuation'] = get_chartype_count(text, strset=string.punctuation)
    d['n_digits']      = get_chartype_count(text, strset=string.digits)
    d['n_letters']     = get_chartype_count(text, strset=string.ascii_letters)
    return d


def _add_text_pair_metrics(df, num_proc):
    logger.info("Calculating text pair statistics for (orig, pp) unique pairs")
    ds_pairs = Dataset.from_pandas(df[['orig','pp']].drop_duplicates())
    ds_pairs = _get_text_pair_metrics_for_ds(ds_pairs, num_proc=num_proc)
    df = pd.merge(df, pd.DataFrame(ds_pairs), how='left', on=['orig', 'pp'])
    return df

def _get_text_pair_metrics_for_ds(ds, num_proc):
    return ds.map(_get_text_pair_metrics, input_columns = ["orig", "pp"], batched=False, num_proc=num_proc)

def _get_text_pair_metrics(orig, pp):
    d = _get_removals_insertions_unchanged_phrases(orig, pp)
    d['edit_distance_token_level'] = _get_token_level_edit_distance(orig, pp)
    return d

def _get_removals_insertions_unchanged_phrases(orig, pp):
    orig_t,pp_t  = [tkn.text for tkn in nlp(orig)],[tkn.text for tkn in nlp(pp)]
    diff = [x for x in ndiff(orig_t, pp_t)]
    ins_idx,ins_tkns,ins_tkn_grps,ins_phrases = _get_subsequences(diff, "insertions")
    rem_idx,rem_tkns,rem_tkn_grps,rem_phrases = _get_subsequences(diff, "removals")
    unc_idx,unc_tkns,unc_tkn_grps,unc_phrases = _get_subsequences(diff, "unchanged")
    return {'removals_idx': rem_idx,
            'removals': rem_phrases,
            'insertions_idx': ins_idx,
            'insertions': ins_phrases,
            'unchanged_idx': unc_idx,
            'unchanged': unc_phrases,
            'n_segments_inserted': len(ins_tkn_grps),
            'n_segments_removed': len(rem_tkn_grps),
            'n_tokens_inserted': len(ins_tkns),
            'n_tokens_removed': len(rem_tkns),
            'is_truncation': _is_truncation(rem_idx, unc_idx),
            'any_phrase_capitalised': _any_phrase_capitalised(rem_phrases, ins_phrases),
            'any_phrase_decapitalised': _any_phrase_capitalised(ins_phrases, rem_phrases)}


def _join_punctuation(seq, characters=set(string.punctuation)):
    "Generator to join tokens respecting punctuation, but doesn't work that well."
    seq = iter(seq)
    current = next(seq)
    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt
    yield current

def _get_subsequences(diff, sign):
    op = {"insertions": "+", "removals": "-", "unchanged": " "}[sign]
    idx,tokens = [],[]
    for i, o in enumerate(diff):
        if o[0] == op:  idx.append(i); tokens.append(o[2:])
    ## Group tokens that go together
    token_groups = []
    # bit of a mystery this bit but seems to work. just need 1-1 mapping between data and tokens
    for k, g in groupby(zip(enumerate(idx), tokens), lambda ix: ix[0][0] - ix[0][1]):
        token_groups.append(list(map(operator.itemgetter(1), g)))
    phrases = [' '.join(_join_punctuation(l)) for l in token_groups]
    return idx, tokens, token_groups, phrases

def _is_truncation(rem_idx, unc_idx):
    """determines if a given phrase is trunctated or not. unc_idx = unchanged_idx, rem_idx = removals_idx """
    if len(rem_idx) == 0 or len(unc_idx) == 0: return False
    if max(unc_idx) < max(rem_idx):  return True
    else:                            return False

def _any_phrase_capitalised(lower_case_phrases, upper_case_phrases):
    """tests if any of the phrases in lower_case_phrases, when capitalised, are present in upper_case_phrases"""
    for lc_p in lower_case_phrases:
        for uc_p in upper_case_phrases:
            if lc_p.capitalize() == uc_p: return True
    return False

def _get_token_level_edit_distance(s1, s2):
    l1,l2 = [o.text for o in nlp(s1)],[o.text for o in nlp(s2)]
    return editdistance.eval(l1,l2)


def create_and_log_wandb_postrun_plots(df_d):
    df_concat = _prepare_df_concat(df_d)
    wandb_plot_d = _prepare_wandb_postrun_plots(df_concat)
    wandb.log(wandb_plot_d)

def _prepare_df_concat(df_d):
    for k,df in df_d.items():
        if  k == "training_step": df_d[k]['data_split'] = k
        else:                     df_d[k]['data_split'] = f"eval_{k}"
    df_concat = pd.concat(df_d.values()).reset_index(drop=True)
    df_concat.loc[df_concat.epoch_of_first_label_flip == 0, 'epoch_of_first_label_flip'] = None  # stop wrong spike at 0
    return df_concat

def _prepare_wandb_postrun_plots(df_concat):
    fig_l = []
    hist_config_dicts = [
        {
            'cname': 'epoch_of_first_label_flip',
            'xlabel': "Epoch of first label flip",
            'desc': "Cumulative prob epoch of first label flip for each original example",
            'cumulative': True,
        },
        {
            'cname': 'idx_n_unique_pp',
            'xlabel': "Unique paraphrases per original example",
            "desc": "Number of generated unique paraphrases per original example during training",
            'cumulative': False,
        },
        {
            'cname': 'idx_n_pp_changes',
            'xlabel': "Paraphrase changes per original example",
            "desc": "Number of paraphrase changes per original example during training",
            'cumulative': False,
        }]
    for d in hist_config_dicts:  fig_l.append({f"pp_metrics/{d['cname']}": _plot_idx_hist(df_concat, d['cname'],d['xlabel'],d['cumulative'])})
    line_cnames = [o for o in df_concat.columns if "_diff" in o] + \
        ["is_truncation", 'any_phrase_capitalised', 'any_phrase_decapitalised', 'n_segments_inserted',
         'n_segments_removed', 'n_tokens_inserted', 'n_tokens_removed','edit_distance_token_level']
    for cname in line_cnames: fig_l.append({f"pp_metrics/{cname}": _plot_epoch_line_charts(df_concat, cname)})
    return {k:v for d in fig_l for k,v in d.items()}

def _plot_idx_hist(df_concat, cname, xlabel, cumulative=False):
    df1 = df_concat[['data_split','idx', cname]].drop_duplicates()
    fig = px.histogram(df1, x=cname, color='data_split', marginal="box",
                       labels={cname: xlabel},cumulative=cumulative, barmode='group',
                      histnorm='probability', color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(showlegend=False)
    fig.update_layout(font_size=8)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(autosize=True)
    return fig

def _plot_epoch_line_charts(df_concat, cname):
    df1 = df_concat[['data_split','epoch', cname]]
    df_grp = df1.groupby(['data_split', 'epoch']).agg('mean').reset_index()
    fig = px.line(df_grp, x="epoch", y=cname, color='data_split', labels={cname: cname + "_avg"},
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(showlegend=False)
    fig.update_layout(font_size=8)
    fig.update_layout(autosize=True)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig