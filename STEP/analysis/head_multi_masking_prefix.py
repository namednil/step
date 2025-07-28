"""
Figure 4 of the EMNLP paper.

Mask specific attention heads and only the attention to the prefix.
"""

import pandas as pd
import torch
import transformers

import pickle
import tqdm

from STEP.analysis.head_multi_masking import interpretable_heads, get_data_loader, \
    get_model_greedy_acc, create_head_masked_model, estimate_masking_acc, all_interpretable_heads
from STEP.sip_grammar import UDPretrainingModel
from STEP.data_loading import load_ud_grammar_pickle

import numpy as np

def swor_gumbel_uniform(n, k):
    """
    Uniformly select k elements out of n objects without replacement.
    Takes linear time, uses the Gumbel max trick and introselect.
    Note: the order in which elements appear is not specified but not random

    see: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
    and https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    """
    assert k <= n
    if k == 0:
        return []
    G = np.random.gumbel(0, 1, size=(n))   # Gumbel noise

    return np.argpartition(G, -k)[-k:]   # select k largest indices in Gumbel noise

class PrefixMaskedLM(torch.nn.Module):

    def __init__(self, model, prefix_len: int, masked_heads: list[tuple[int, int]], num_layer=None, num_heads=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.num_layer = num_layer or self.model.config.num_layer
        self.num_heads = num_heads or self.model.config.num_heads
        self.head_mask = torch.ones(num_layer, num_heads).to(model.device)
        self.decoder_head_mask = torch.ones(num_layer, num_heads).to(model.device)
        self.prefix_len = prefix_len
        self.masked_heads = masked_heads

    def _gen_head_mask(self, kwargs):
        seq_len = kwargs["input_ids"].shape[1] + self.prefix_len # we take the input_ids which doesn't yet account for the prefix
        head_mask = torch.ones((self.num_layer, 1, self.num_heads, seq_len, seq_len))
        for (layer, head) in self.masked_heads:
            head_mask[layer, :, head, :, :self.prefix_len] = 0.0
        return head_mask.to(self.model.device)

    def forward(self, **kwargs):
        # T5 also accepts a head mask with this layout: (layers, batch_size, n_heads, seq_length, key_length)
        return self.model(**(kwargs | {"head_mask": self._gen_head_mask(kwargs), "decoder_head_mask": self.decoder_head_mask}))

    @property
    def device(self):
        return self.model.device

    def generate(self, **kwargs):
        return self.model.generate(**(
                    kwargs | {"head_mask": self._gen_head_mask(kwargs), "decoder_head_mask": self.decoder_head_mask}))


class RandomPositionMaskedLM(torch.nn.Module):

    def __init__(self, model, num_masked_tokens: int, num_masked_heads: int, num_layer=None,
                 num_heads=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.num_layer = num_layer or self.model.config.num_layer
        self.num_heads = num_heads or self.model.config.num_heads
        self.head_mask = torch.ones(num_layer, num_heads).to(model.device)
        self.decoder_head_mask = torch.ones(num_layer, num_heads).to(model.device)
        self.num_masked_tokens = num_masked_tokens
        self.num_masked_heads = num_masked_heads

        self.all_heads = []
        for layer in range(num_layer):
            for head in range(num_heads):
                self.all_heads.append((layer, head))
        assert self.num_masked_heads <= len(self.all_heads)

    def _gen_head_mask(self, kwargs):
        seq_len = kwargs["input_ids"].shape[
                      1] + self.num_masked_tokens  # we take the input_ids which doesn't yet account for the prefix
        batch_size = kwargs["input_ids"].shape[0]
        head_mask = torch.ones((self.num_layer, batch_size, self.num_heads, seq_len, seq_len)).numpy()

        num_tokens = self.num_masked_tokens + (kwargs["input_ids"] != 0).sum(dim=1) - 1 #shape (batch_size,) # -1 because the last position is the EOS token - we don't mask that one because if usually has a specific no-op function.
        num_tokens = num_tokens.cpu().numpy()
        for batch_id in range(batch_size):
            heads_to_mask = [self.all_heads[index] for index in swor_gumbel_uniform(len(self.all_heads), self.num_masked_heads)]
            random_positions = swor_gumbel_uniform(num_tokens[batch_id], self.num_masked_tokens)
            for layer, head in heads_to_mask:
                for p in random_positions:
                    head_mask[layer, batch_id, head, :, p] = 0.0

        return torch.from_numpy(head_mask).to(self.model.device)

    def forward(self, **kwargs):
        # T5 also accepts a head mask with this layout: (layers, batch_size, n_heads, seq_length, key_length)
        return self.model(
            **(kwargs | {"head_mask": self._gen_head_mask(kwargs), "decoder_head_mask": self.decoder_head_mask}))

    @property
    def device(self):
        return self.model.device

    def generate(self, **kwargs):
        return self.model.generate(**(
                kwargs | {"head_mask": self._gen_head_mask(kwargs), "decoder_head_mask": self.decoder_head_mask}))


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


def estimate_prefix_masking_acc_from_heads(sip_pretrained_model, prefix_len, data_loader, num_heads, from_heads, n=10):
    from_heads = list(from_heads)
    total_accs = 0
    if num_heads >= len(from_heads):
        return get_model_greedy_acc(create_head_masked_model(sip_pretrained_model, from_heads,
                                                masked_cross_heads=[], num_layer=12, num_heads=12), data_loader)
    for i in range(n):
        chosen_head_indices = np.random.choice(np.arange(len(from_heads)), (num_heads,), replace=False)
        masking = [from_heads[index] for index in chosen_head_indices]
        #print(masking)
        masked_model = PrefixMaskedLM(sip_pretrained_model, prefix_len, masking, num_layer=12,
                       num_heads=12)
        acc = get_model_greedy_acc(masked_model, data_loader)
        total_accs += acc
    return total_accs / n

def estimate_masking_acc_random_tokens(sip_pretrained_model, num_tokens, data_loader, num_heads, n=10):
    # from_heads = [(layer, head) for layer in range(12) for head in range(12)]
    sum_acc = 0.0
    for i in range(n):
        # chosen_head_indices = np.random.choice(np.arange(len(from_heads)), (num_heads,), replace=False)
        # masking = [from_heads[index] for index in chosen_head_indices]
        masked_model = RandomPositionMaskedLM(sip_pretrained_model, num_tokens, num_heads, num_layer=12, num_heads=12)
        acc = get_model_greedy_acc(masked_model, data_loader)
        sum_acc += acc
    return sum_acc / n


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
        masked_model = PrefixMaskedLM(sip_pretrained_model, 1, interpretable_heads[relation], num_layer=12,
                                                num_heads=12)
        interpret_acc = get_model_greedy_acc(masked_model, data_loader)
        print(interpret_acc)
        print(f"Est. expected impact of masking {len(interpretable_heads[relation])} heads and random token positions:")
        est_impact = estimate_masking_acc_random_tokens(sip_pretrained_model, 1, data_loader, len(interpretable_heads[relation]), n=20)
        print(est_impact)
        print("===")

        print("Est. expected impact of masking other interpretable heads")
        other_interpretable_heads = all_interpretable_heads - set(interpretable_heads[relation])
        other_interpretable_heads_acc = estimate_prefix_masking_acc_from_heads(sip_pretrained_model, 1, data_loader,
                                                                        len(interpretable_heads[relation]),
                                                                        from_heads=other_interpretable_heads, n=20)

        reductions["relation"].append(relation)
        reductions["masking_interpretable"].append(interpret_acc)
        reductions["no_masking"].append(no_masking)
        reductions["expected_masking"].append(est_impact)
        reductions["other_interpretable"].append(other_interpretable_heads_acc)

    reductions = pd.DataFrame(reductions)
    reductions.to_csv("head_multi_masking_prefix.csv")
