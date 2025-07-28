
def create_nlg_eval(**kwargs):
    return lambda *args, **kwargs2: nlg_eval(*args, **(kwargs2 | kwargs))


def nlg_eval(model, tokenizer, dataloader, logger=None, **kwargs):
    from sacrebleu.metrics import BLEU, CHRF, TER

    total = 0
    acc = 0
    model.eval()

    metrics = [
        BLEU(), CHRF(), TER()
    ]

    all_pred = []
    all_gold = []
    for test_batch in dataloader:
        test_batch = {k: v.to(model.device) for k, v in test_batch.items()}
        test_batch_inputs = dict(test_batch)
        del test_batch_inputs["labels"]
        r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1] + 2,
                                                  early_stopping="never", **kwargs), skip_special_tokens=True)
        gold = tokenizer.batch_decode(100 * (test_batch["labels"] == -100) + test_batch["labels"],
                                      skip_special_tokens=True)  # replace -100 by 0

        if logger is not None:
            logger.log_output([{"prediction": pi, "gold": gi, "input": ii}
                               for pi, gi, ii in
                               zip(r, gold, tokenizer.batch_decode(test_batch["input_ids"], skip_special_tokens=True))])

        all_pred.extend(r)
        all_gold.extend(gold)
        for x, y in zip(r, gold):
            total += 1
            acc += (x == y)

    ret = {"acc": acc / total}
    for metric in metrics:
        result = metric.corpus_score(all_pred, [all_gold]) # a single reference only
        ret[result.name] = result.score

    return ret
