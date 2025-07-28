import sys
from collections import defaultdict

import numpy as np

def create_cogs_eval_match():
    return cogs_eval_exact_match


def close_brackets(predicted_texts: str):
    tokens = predicted_texts.strip()
    lb = 0
    for token in tokens:
        if token == "(":
            lb += 1
        if token == ")":
            lb -= 1
    if lb >= 1:
        for i in range(lb):
            tokens += " )"
    return tokens


import re

regex = re.compile("([(),= ])")


def tokenize_slog(s):
    return [w for w in regex.split(s) if w.strip()]


import nltk


def parse_slog(str) -> nltk.Tree:
    tokens = tokenize_slog(str)
    index = 0

    def parse_slog_trees() -> list[nltk.Tree]:
        nonlocal index
        if tokens[index] != "(":
            return []
        # assert tokens[index] == "("
        index += 1
        children = []
        while tokens[index] != ")":
            if tokens[index] == ",":
                index += 1  # simply skip this
            children.append(parse_slog_tree())
        index += 1
        return children

    def parse_slog_tree() -> nltk.Tree:
        nonlocal index
        if tokens[index + 1] == "=":
            relation = tokens[index]
            name = tokens[index + 1]
            name = relation + " " + name
            index += 2
            single_child = parse_slog_tree()
            return nltk.Tree(name, [single_child])
        else:
            if tokens[index] == "*":
                name = "* " + tokens[index + 1]
                index += 2
            else:
                name = tokens[index]
                index += 1
            children = parse_slog_trees()
            return nltk.Tree(name, children)

    return parse_slog_tree()


def sort_tree(t) -> None:
    if isinstance(t, nltk.Tree):
        t.sort()
        for c in t:
            sort_tree(c)

class SetTree:
    """
    Tree with sets of children, i.e. we don't care about the order or if there are repeated subtrees.
    """
    def __init__(self, node, children):
        self.node = node
        self.children = set(children)

    def __hash__(self):
        return hash(self.node) + sum(hash(c) for c in self.children)

    def __eq__(self, other):
        f = self.node == other.node and len(self.children) == len(other.children)
        if not f:
            return False

        for myc, otherc in zip(sorted(self.children, key=lambda x: hash(x)), sorted(other.children, key=lambda x: hash(x))):
            f = f and myc == otherc
        return f

    def __str__(self):
        return f"({self.node} " + " ".join(str(c) for c in sorted(self.children, key=lambda x: hash(x))) + ")"

    @staticmethod
    def from_tree(t):
        return SetTree(t.label(), {SetTree.from_tree(c) for c in t})

def cogs_eval_exact_match(model, tokenizer, dataloader, logger=None):
    totals = dict()
    correct = dict()
    correct_tree = dict()
    correct_set_tree = dict()
    model.eval()
    for test_batch in dataloader:
        gen_types = test_batch.pop("gen_type")
        test_batch = {k: v.to(model.device) for k, v in test_batch.items()}
        test_batch_inputs = dict(test_batch)
        del test_batch_inputs["labels"]
        r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1] + 2,
                                                  early_stopping="never", num_beams=1, no_repeat_ngram_size=0),
                                   skip_special_tokens=True)
        gold = tokenizer.batch_decode((100 + tokenizer.eos_token_id) *(test_batch["labels"] == -100) + test_batch["labels"], skip_special_tokens=True) # replace -100 by eos token id, which will be skipped.

        inputs = tokenizer.batch_decode(test_batch["input_ids"], skip_special_tokens=True)
        logged_outputs = []
        for x, y, gen_type, input_text in zip(r, gold, gen_types, inputs):
            x = close_brackets(x)

            logged_outputs.append({"prediction": x, "gold": y, "gen_type": gen_type, "input": input_text})
            # print(x, "\t|\t", y)
            totals[gen_type] = totals.get(gen_type, 0) + 1
            correct[gen_type] = correct.get(gen_type, 0) + (x == y)

            try:
                try:
                    ty = parse_slog(y)
                    sort_tree(ty)
                except IndexError:
                    print("Warning, couldn't parse gold tree:", y, file=sys.stderr)
                    continue

                tx = parse_slog(x)
                sort_tree(tx)

                correct_tree[gen_type] = correct_tree.get(gen_type, 0) + (tx == ty)
                correct_set_tree[gen_type] = correct_set_tree.get(gen_type, 0) + (SetTree.from_tree(tx) == SetTree.from_tree(ty))
            except IndexError:
                # predicted tree was malformed
                pass
        logger.log_output(logged_outputs)

    median_acc = np.median([correct[gen_type] / totals[gen_type] * 100 for gen_type in totals])
    median_tree_acc = np.median([correct_tree[gen_type] / totals[gen_type] * 100 for gen_type in totals])
    median_set_tree_acc = np.median([correct_set_tree[gen_type] / totals[gen_type] * 100 for gen_type in totals])

    r = {f"acc_{gen_type}": correct[gen_type] / totals[gen_type] * 100 for gen_type in totals} | {
        "acc": sum(correct.values()) / sum(totals.values()) * 100}
    r = r | {f"acc_tree_{gen_type}": correct_tree[gen_type] / totals[gen_type] * 100 for gen_type in totals} | {
        "acc_tree": sum(correct_tree.values()) / sum(totals.values()) * 100}
    r = r | {f"acc_set_tree_{gen_type}": correct_set_tree[gen_type] / totals[gen_type] * 100 for gen_type in totals} | {
        "acc_set_tree": sum(correct_set_tree.values()) / sum(totals.values()) * 100}
    r["acc_median"] = median_acc
    r["acc_tree_median"] = median_tree_acc
    r["acc_set_tree_median"] = median_set_tree_acc

    return r


