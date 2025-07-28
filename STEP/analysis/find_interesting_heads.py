"""
See Section 5.1 of EMNLP paper.

This script tries to identify lookup heads from a sample of transformations
similar to those used during pre-training.
"""

import gzip
from collections import Counter, defaultdict

import conllu
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, utils

from STEP.sip_grammar import UDPretrainingModel
from STEP.data_loading import load_ud_grammar_pickle
import pickle


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

EDGE_FUNCTIONS_LIST = [
"concat",
"concat-deprel",
"rev-deprel",
"rev",
"improved-triple",
"improved-triple-by",
"bracket-1",
"bracket-invert-1",
"bracket-2",
"bracket-invert-2",
"bracket-3",
"bracket-4",
"ignore-dep",
"bracket-full",
]


class TokenToWordLink:
    """
    Maps between tokenized words and words as used in dependency trees.
    """

    def __init__(self, tokenlist, tokenizer):
        self.tokenlist = tokenlist
        self.batch_enc = tokenizer([t["form"] for t in tokenlist], is_split_into_words=True, add_special_tokens=True)

    def get_dep_token(self, hf_id):
        r = self.batch_enc.word_ids()[hf_id]
        if r is None:
            return None
        return self.tokenlist[r]

    def enumerate(self):
        for hf_index, word_id in enumerate(self.batch_enc.word_ids()):
            if word_id is not None:
                yield self.batch_enc["input_ids"][hf_index], self.tokenlist[word_id]


def t5_strip(token: str):
    symobls_to_strip = {'▁'}
    return "".join(c for c in token if c not in symobls_to_strip)

def get_children_dict(tokenlist) -> dict[int, list[int]]:
    """
    r[head] returns a list of children with 0-indexing. head is 1-based indexing!
    :param tokenlist:
    :return:
    """
    d = dict()
    for token in tokenlist:
        if token["head"] not in d:
            d[token["head"]] = []
        d[token["head"]].append(token["id"]-1)
    return d


if __name__ == "__main__":
    import tqdm

    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers.T5ForConditionalGeneration

    id2conllu = dict()
    with gzip.open("grammars/pt_data_step_used_dev_with_ids.conllu.gz", "rt") as f:
        for tokenlist in conllu.parse_incr(f):
            task_id = int(tokenlist.metadata.get("task_id", 0))
            # print(tokenlist.metadata)
            id2conllu[task_id] = TokenToWordLink(tokenlist, tokenizer)

    with open(
            "grammars/pt_data_step_l2i.pkl",
            "rb") as f:
        l2i = pickle.load(f)

    index_to_ud_label = ["PAD"] + l2i.i2l

    utils.logging.set_verbosity_error()  # Remove line to see warnings

    model = UDPretrainingModel.from_pretrained(
            "models/step_model",
            map_location="cpu")

    data_loader = load_ud_grammar_pickle(
        "grammars/pt_data_step_dev.pkl.xz",
        tokenizer, 1, False)

    head_counter = defaultdict(Counter)

    """
    We are trying to find a pattern for each head, what they like to attend to.
    """

    df = {"layer": [], "head": [], "t5token": [], "deprel": [], "deprel_in_prefix": [], "most_attended_to": []}


    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        ud_labels = [index_to_ud_label[int(ud_label.numpy())] for ud_label in batch["ud_labels"][0]]
        outputs = model(**batch, output_attentions=True)

        t5_tokens = ud_labels + tokenizer.convert_ids_to_tokens(batch["input_ids"][0].cpu().numpy())

        tokenlink = id2conllu[int(batch["task_ids"][0].numpy())]
        children_dict = get_children_dict(tokenlink.tokenlist)
        for layer in range(12):
            for head in range(12):
                relevant_attention_scores = outputs.encoder_attentions[layer][0, head] # (seq_len, seq_len) normalized over last dimension
                most_attended_positions = torch.argmax(relevant_attention_scores, dim=-1).numpy()
                
                for attend_from, position in enumerate(most_attended_positions):
                    if attend_from < len(ud_labels) or attend_from == relevant_attention_scores.shape[1]-1:
                        # outside the range where the tokens are
                        continue

                    if relevant_attention_scores[attend_from, position] < 0.5:
                    #     # not very high attention mass.
                        continue

                    corresponding_word = tokenlink.get_dep_token(attend_from - len(ud_labels))
                    corresponding_t5_token = t5_tokens[attend_from]
                    if not t5_strip(corresponding_t5_token):
                        continue
                    # print(corresponding_t5_token, corresponding_word)

                    if position < len(ud_labels):
                        # most attention to prefix
                        most_attended_to_type = ud_labels[position]
                    elif position == relevant_attention_scores.shape[1]-1: #last position: EOS symbol
                        most_attended_to_type = "EOS"
                    else:
                        most_attended_to_type = "token"

                    df["layer"].append(layer)
                    df["head"].append(head)
                    df["t5token"].append(corresponding_t5_token)
                    df["deprel"].append(corresponding_word["deprel"])
                    df["deprel_in_prefix"].append(corresponding_word["deprel"] in ud_labels)
                    df["most_attended_to"].append(most_attended_to_type)
                    head_counter[(layer, head, corresponding_word["deprel"])].update([most_attended_to_type])



    df = pd.DataFrame(df)
    df.to_csv("head_analysis_dev2.csv")


    head_counter_normalized = {k: {token: freq / sum(v.values()) for token, freq in v.items()} for k,v in head_counter.items()}
    # head_counter_normalized = head_counter
    for k in head_counter_normalized:
        # if any(p > 0.4 and most_attended_to not in {"EOS", "token"} for most_attended_to, p in head_counter_normalized[k].items()):
        # if any(p < 0.3 and most_attended_to == "token" for most_attended_to, p in head_counter_normalized[k].items()):
        if k[-1] == head_counter[k].most_common(1)[0][0]:
            print(k)
            print(head_counter[k])
            # print(head_counter_normalized[k])
            print("====")


