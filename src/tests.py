__all__ = ['check_no_nans_or_infs', 'check_parameters_update', 'print_info_on_generated_text']


import torch


def check_no_nans_or_infs(x):
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isneginf(x))
    assert torch.all(~torch.isposinf(x))


def check_parameters_update(dl):
    """
    This checks which parameters are being updated.
    We run one forward pass+backward pass (updating the parameters once)
    and look at which ones change.
    """
    # Check which parameters should be updated
    params_with_grad = [o for o in pp_model.named_parameters() if o[1].requires_grad]
    print("---- Parameters with 'requires_grad' and their sizes ------")
    for (name, p) in params_with_grad:  print(name, p.size())

    ## Take a step and see which weights update
    params_all = [o for o in pp_model.named_parameters()]  # this is updated by a training step
    params_all_initial = [(name, p.clone()) for (name, p) in params_all]  # Initial values

    # take a step
    loss, reward, pp_logp = training_step(data)

    print("\n---- Matrix norm of parameter update for one step ------\n")
    for (_,old_p), (name, new_p) in zip(params_all_initial, params_all):
        print (name, torch.norm(new_p - old_p).item())


def print_info_on_generated_text():
    """
        Prints a bunch of statistics around the generated text. Useful for debugging purposes.
        So far only works for greedy search.
        OUTDATED OUTDATED
    """
    logger.info("\n######################################################################\n")
    logger.info(f"Original text: {text}")
    tgt_text = pp_tokenizer.batch_decode(translated.sequences, skip_special_tokens=True)
    tgt_text_with_tokens = pp_tokenizer.batch_decode(translated.sequences, skip_special_tokens=False)
    logger.info(f"Generated text: {tgt_text}")
    logger.info(f"Generated text with special tokens: {tgt_text_with_tokens}")
    logger.info(f"Shape of translated.sequences:{translated.sequences.shape}")
    logger.info(f"translated.sequences:{translated.sequences}")
    logger.info(f"Scores is a tuple of length {len(translated.scores)} \
    and each score is a tensor of shape {translated.scores[0].shape}")
    scores_stacked = torch.stack(translated.scores, 1)
    logger.info(f"Stacking the scores into a tensor of shape {scores_stacked.shape}")
    scores_softmax = torch.softmax(scores_stacked, 2)
    logger.info(f"Now taking softmax. This shouldn't change the shape, but just to check,\
    its shape is {scores_softmax.shape}")
    probsums = scores_softmax.sum(axis=2)
    logger.info(f"These are probabilities now and so they should all sum to 1 (or close to it) in the axis \
    corresponding to each time step. We can check the sums here: {probsums}, but it's a long tensor \
    of shape {probsums.shape} and hard to see, so summing over all these values and removing 1 \
    from each gives {torch.sum(probsums - 1)} \
    which should be close to 0.")
    seq_without_first_tkn = translated.sequences[:, 1:]
    logger.info("Now calculating sequence probabilities")
    seq_token_probs = torch.gather(scores_softmax,2,seq_without_first_tkn[:,:,None]).squeeze(-1)
    seq_prob = seq_token_probs.prod(-1).item()
    logger.info(f"Sequence probability: {seq_prob}")

    # Get the 2nd and 3rd most likely tokens at each st
    topk_ids = torch.topk(scores_softmax,3,dim=2).indices[:,:,1:]
    topk_tokens_probs = torch.gather(scores_softmax,2,topk_ids).squeeze(-1)
    toks2 = pp_tokenizer.convert_ids_to_tokens(topk_ids[:,:,0].squeeze())
    toks3 = pp_tokenizer.convert_ids_to_tokens(topk_ids[:,:,1].squeeze())
    tok_probs2 = topk_tokens_probs[:,:,0].squeeze()
    tok_probs3 = topk_tokens_probs[:,:,1].squeeze()

    logger.info(f"Probabilities of getting the top 3 tokens at each step:")
    tokens = pp_tokenizer.convert_ids_to_tokens(seq_without_first_tkn.squeeze())
    for (p, t, p2,t2,p3,t3)  in zip(seq_token_probs.squeeze(), tokens, tok_probs2, toks2, tok_probs3, toks3):
        logger.info(f"{t}: {round(p.item(),3)}  {t2}: {round(p2.item(),3)}  {t3}: {round(p3.item(),3)}")
