
import re

chunk_format = re.compile("\([^()]+\)")

from collections import Counter


def extract_chunks(s):
    return Counter(chunk_format.findall(s.lower()))


def create_chunk_eval(**kwargs):
    return lambda *args, **kwargs2: chunk_eval(*args, **(kwargs2 | kwargs))


def chunk_eval(model, tokenizer, dataloader, logger=None, **kwargs):
    model.eval()
    total = 0
    correct = 0

    correct_chunks = 0
    num_pred_chunks = 0
    num_gold_chunks = 0

    for test_batch in dataloader:
        test_batch = {k: v.to(model.device) for k, v in test_batch.items()}
        test_batch_inputs = dict(test_batch)
        del test_batch_inputs["labels"]
        r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1] + 5,
                                                  early_stopping="never", **kwargs), skip_special_tokens=True)
        gold = tokenizer.batch_decode((100 + tokenizer.eos_token_id) *(test_batch["labels"] == -100) + test_batch["labels"], skip_special_tokens=True) # replace -100 by eos token id, which will be skipped.

        if logger is not None:
            logger.log_output([{"prediction": pi, "gold": gi, "input": ii}
                               for pi, gi, ii in
                               zip(r, gold, tokenizer.batch_decode(test_batch["input_ids"], skip_special_tokens=True))])

        for p, g in zip(r, gold):
            total += 1
            # print(p, "|||", g)

            correct += (p == g)

            pred_chunks = extract_chunks(p)
            gold_chunks = extract_chunks(g)
            len_intersection = sum((pred_chunks & gold_chunks).values())

            correct_chunks += len_intersection
            num_gold_chunks += sum(gold_chunks.values())
            num_pred_chunks += sum(pred_chunks.values())

    precision = correct_chunks / num_pred_chunks if num_pred_chunks > 0 else 0
    recall = correct_chunks / num_gold_chunks
    f = 2*precision*recall / (precision + recall) if precision + recall > 0 else 0

    return {"acc": correct / total, "P": precision, "R": recall, "F": f}


if __name__ == "__main__":

    s = ("( NP The action ) ( VP came ) ( SBAR as ) ( NP Congress ) ( VP sent ) ( PP to ) ( NP President Bush ) ( NP a "
         "fiscal 1990 bill ) ( VP providing ) ( NP an estimated $ 156.7 billion ) ( PP for ) ( NP the Departments ) ( "
         "PP of ) ( NP Labor ) , ( NP Education ) , ( NP Health ) , ( NP Health ) and ( NP Human Services ) .")

    print(extract_chunks(s))
