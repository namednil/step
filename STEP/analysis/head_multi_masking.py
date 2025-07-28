"""
Like Figure 4 in the EMNLP paper but masking heads entirely, not just the attention to the prefix.


"""

import pandas as pd
import torch
import transformers

import pickle
import tqdm
from STEP.sip_grammar import UDPretrainingModel
from STEP.data_loading import load_ud_grammar_pickle

import numpy as np


class MaskedLM(torch.nn.Module):
    def __init__(self, model, num_layer=None, num_heads=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        num_layer = num_layer or self.model.config.num_layer
        num_heads = num_heads or self.model.config.num_heads
        self.head_mask = torch.ones(num_layer, num_heads).to(model.device)
        self.decoder_head_mask = torch.ones(num_layer, num_heads).to(model.device)
        self.cross_attn_head_mask = torch.ones(num_layer, num_heads).to(model.device)

    def forward(self, **kwargs):
        return self.model(**(kwargs | {"head_mask": self.head_mask, "decoder_head_mask": self.decoder_head_mask,
                                       "cross_attn_head_mask": self.cross_attn_head_mask}))

    @property
    def device(self):
        return self.model.device

    def generate(self, **kwargs):
        return self.model.generate(**(
                kwargs | {"head_mask": self.head_mask, "decoder_head_mask": self.decoder_head_mask,
                          "cross_attn_head_mask": self.cross_attn_head_mask}))


def create_head_masked_model(model, masked_heads: list[tuple[int, int]], masked_cross_heads: list[tuple[int, int]] = (),
                             **kwargs):
    masked_model = MaskedLM(model, **kwargs)
    for layer, head in masked_heads:
        masked_model.head_mask[layer, head] = 0.0
    for layer, head in masked_cross_heads:
        masked_model.cross_attn_head_mask[layer, head] = 0.0
    return masked_model


def get_model_greedy_acc(model, data_loader):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            r = model(**batch)
            total += batch["labels"].shape[0]
            greedy_predict = torch.argmax(r.logits, dim=-1)  # shape (batch, seq_len)
            tokens_correct = torch.logical_or(greedy_predict == batch["labels"], batch["labels"] == -100)
            all_tokens_correct = torch.all(tokens_correct, dim=-1)  # shape (batch,)
            correct += all_tokens_correct.sum().detach().cpu().numpy()
    return correct / total


class IncrementalVocab:
    def __init__(self, l=None):
        self.l2i = dict()
        self.i2l = []
        if l is not None:
            for i, x in enumerate(l):
                self.l2i[x] = i
                self.i2l.append(x)

    def __len__(self):
        return len(self.l2i)

    def __getitem__(self, item) -> int:
        if item in self.l2i:
            return self.l2i[item]
        self.l2i[item] = len(self.l2i)
        self.i2l.append(item)
        return len(self.l2i) - 1

    def get_obj(self, i: int):
        return self.i2l[i]


def only_label(label):
    """
    Create a filter function that only lets transformations through that use a specific label.
    :param label:
    :return:
    """

    def objects_only(dp):
        used_ud_labels = {rule.lhs for rule in dp["grammar"]}
        ok = len(used_ud_labels & {l2i[label] + 1}) > 0
        return ok

    return objects_only


def get_data_loader(label, tokenizer, **kwargs):
    return load_ud_grammar_pickle(
        "grammars/step_single_transformation.pkl.xz",
        tokenizer, 32, True, filter_f=only_label(label), **kwargs)


def estimate_masking_acc_from_heads(sip_pretrained_model, data_loader, num_heads, from_heads, n=10):
    from_heads = list(from_heads)
    total_accs = 0
    if num_heads >= len(from_heads):
        return get_model_greedy_acc(create_head_masked_model(sip_pretrained_model, from_heads,
                                                             masked_cross_heads=[], num_layer=12, num_heads=12),
                                    data_loader)
    for i in range(n):
        chosen_head_indices = np.random.choice(np.arange(len(from_heads)), (num_heads,), replace=False)
        masking = [from_heads[index] for index in chosen_head_indices]
        # print(masking)
        masked_model = create_head_masked_model(sip_pretrained_model, masking,
                                                masked_cross_heads=[], num_layer=12, num_heads=12)
        acc = get_model_greedy_acc(masked_model, data_loader)
        total_accs += acc
    return total_accs / n


def estimate_masking_acc(sip_pretrained_model, data_loader, num_heads, n=10):
    return estimate_masking_acc_from_heads(sip_pretrained_model, data_loader, num_heads,
                                           [(layer, head) for layer in range(12) for head in range(12)], n)


interpretable_heads = {'cop': [(0, 3), (4, 11), (7, 11), (8, 11), (9, 5), (9, 6), (10, 5), (11, 11)],
                       'expl': [(0, 7), (7, 11), (8, 2), (8, 11), (9, 6), (9, 7), (11, 11)],
                       'amod': [(4, 6), (6, 6), (7, 11), (8, 0), (8, 11), (9, 5), (11, 11)],
                       'compound': [(4, 6), (6, 6), (7, 6), (7, 11), (8, 11), (9, 5), (9, 7), (9, 11), (11, 11)],
                       'det': [(4, 6), (7, 11), (8, 11), (9, 5), (9, 6), (10, 5)],
                       'nmod:poss': [(4, 6), (4, 11), (7, 11), (8, 11), (9, 5), (9, 6), (11, 11)],
                       'advmod': [(4, 11), (6, 6), (7, 11), (8, 11), (9, 5), (9, 6), (9, 11), (11, 11)],
                       'aux': [(4, 11), (7, 11), (8, 11), (9, 5), (9, 6), (10, 5), (11, 11)],
                       'mark': [(4, 11), (8, 11), (9, 5), (9, 6), (11, 11)],
                       'fixed': [(5, 5), (8, 2), (8, 6), (9, 4), (9, 6), (10, 1), (10, 4), (10, 6), (10, 11), (11, 11)],
                       'compound:prt': [(6, 2), (6, 6), (7, 11), (8, 2), (8, 6), (9, 4), (9, 6), (10, 4), (10, 6),
                                        (10, 11), (11, 11)],
                       'acl': [(6, 6), (7, 11), (8, 2), (9, 4), (10, 6), (10, 11), (11, 11)],
                       'nummod': [(6, 6), (7, 11), (8, 11), (9, 6), (11, 11)],
                       'flat': [(6, 11), (7, 11), (8, 2), (8, 11), (9, 4), (10, 6), (10, 11), (11, 11)],
                       'aux:pass': [(7, 11), (8, 11), (9, 5), (9, 6), (10, 5), (11, 11)],
                       'iobj': [(7, 11), (10, 4), (10, 11)],
                       'nsubj': [(7, 11), (8, 11), (9, 5), (9, 6), (9, 11), (11, 11)],
                       'obj': [(7, 11), (10, 4), (10, 6), (10, 11), (11, 11)],
                       'obl:tmod': [(7, 11), (9, 4), (10, 4), (10, 6), (11, 11)], 'case': [(8, 11), (9, 5)],
                       'cc': [(8, 11), (9, 5), (9, 6), (11, 11)],
                       'obl:npmod': [(8, 11), (9, 6), (9, 11), (10, 6), (11, 11)],
                       'punct': [(8, 11), (9, 6), (10, 6), (10, 11), (11, 5)], 'csubj': [(9, 11), (10, 6), (11, 11)],
                       'nsubj:pass': [(9, 11), (10, 6), (11, 11)], 'obl': [(9, 11), (10, 6)], 'acl:relcl': [(10, 6)],
                       'advcl': [(10, 6), (11, 11)], 'appos': [(10, 6), (10, 11), (11, 11)], 'ccomp': [(10, 6)],
                       'conj': [(10, 6)], 'nmod': [(10, 6), (10, 11)], 'vocative': [(10, 6)],
                       'xcomp': [(10, 6), (10, 11)]}

all_interpretable_heads = set()
for heads in interpretable_heads.values():
    all_interpretable_heads.update(heads)

if __name__ == "__main__":
    with open("grammars/pt_data_step_l2i.pkl",
              "rb") as f:
        l2i = pickle.load(f)

    index_to_ud_label = ["PAD"] + l2i.i2l
    sip_pretrained_model = UDPretrainingModel.from_pretrained(
        "models/step_model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

    reductions = {"relation": [], "masking_interpretable": [], "expected_masking": [], "no_masking": [],
                  "other_interpretable": []}

    for relation in interpretable_heads:
        try:
            data_loader = get_data_loader(relation, tokenizer)
        except ValueError:
            continue
        print(relation)
        print("without masking")
        no_masking = get_model_greedy_acc(sip_pretrained_model, data_loader)
        print(no_masking)
        print(f"masking interpretable heads ({interpretable_heads[relation]})")
        masked_model = create_head_masked_model(sip_pretrained_model, interpretable_heads[relation], num_layer=12,
                                                num_heads=12)
        interpret_acc = get_model_greedy_acc(masked_model, data_loader)
        print(interpret_acc)
        print(f"Est. expected impact of masking {len(interpretable_heads[relation])} heads:")
        est_impact = estimate_masking_acc(sip_pretrained_model, data_loader, len(interpretable_heads[relation]), n=50)
        print(est_impact)
        print("===")

        print("Est. expected impact of masking other interpretable heads")
        other_interpretable_heads = all_interpretable_heads - set(interpretable_heads[relation])
        other_interpretable_heads_acc = estimate_masking_acc_from_heads(sip_pretrained_model, data_loader,
                                                                        len(interpretable_heads[relation]),
                                                                        from_heads=other_interpretable_heads, n=50)

        reductions["relation"].append(relation)
        reductions["masking_interpretable"].append(interpret_acc)
        reductions["no_masking"].append(no_masking)
        reductions["expected_masking"].append(est_impact)
        reductions["other_interpretable"].append(other_interpretable_heads_acc)

    reductions = pd.DataFrame(reductions)
    reductions.to_csv("head_multi_masking.csv")
