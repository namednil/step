"""
See beginning of Section 5.2 in EMNLP paper.
"""


import dataclasses
import gzip
import json
import lzma
import pickle
import sys
from collections import defaultdict, Counter
from typing import Callable, Optional

import conllu

from enum import Enum

import tqdm

import numpy as np
import transformers
from conllu import parse_conllu_plus_fields, parse_sentences, parse_token_and_metadata, SentenceGenerator

from STEP.data_gen.grammar_gen import ProductionRule, weighted_choice
from STEP.data_gen.utils import swor_gumbel_uniform

class EdgeDirection(Enum):
    LEFT = 1
    RIGHT = 2
    ANY = 3


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


class NamedFunction:
    counter = 0

    def __init__(self, name, function, contains_dep_rel, weight=1.0):
        self.id = NamedFunction.counter
        NamedFunction.counter += 1
        self.name = name
        self.function = function
        self.contains_dep_rel = contains_dep_rel
        self.weight = weight

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


@dataclasses.dataclass
class EdgeRule:
    label: str
    function: NamedFunction
    output_label_token: Optional[str] = None

    def get_info(self):
        return (self.label,)  # self.head_pos)

    @staticmethod
    def extract_info(head_token, dep_token):
        return (dep_token["deprel"],)

    def matches(self, head_token, dep_token):
        if dep_token["deprel"] != self.label:
            return False
        # elif head_token["upos"] != self.head_pos:
        #     return False
        # return self.direction == EdgeDirection.ANY or \
        #     (self.direction == EdgeDirection.LEFT and dep_token["id"] < head_token["id"]) or \
        #     (self.direction == EdgeDirection.RIGHT and dep_token["id"] > head_token["id"])
        return True

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


@dataclasses.dataclass
class TokenRule:
    head_pos: str
    first_f: NamedFunction
    last_f: NamedFunction

    def get_info(self):
        return (self.head_pos,)


def bracket_full(head_result, dep_result, **kwargs):
    other_full_bracket_children_before = 0
    other_full_bracket_children_after = 0
    my_function_name = kwargs["all_edge_rules"][kwargs["index"]].function.name
    assert my_function_name == "bracket-full"
    for i, edge_rule in enumerate(kwargs["all_edge_rules"]):
        if edge_rule is not None and edge_rule.function.name == my_function_name:
            if i < kwargs["index"]:
                other_full_bracket_children_before += 1
            elif i > kwargs["index"]:
                other_full_bracket_children_after += 1

    if other_full_bracket_children_before == 0 and other_full_bracket_children_after == 0:
        return head_result + ["(", kwargs["deprel"]] + dep_result + [")"]
    elif other_full_bracket_children_before == 0:
        return head_result + ["(", kwargs["deprel"]] + dep_result
    elif other_full_bracket_children_after == 0:
        return head_result + [",", kwargs["deprel"]] + dep_result + [")"]
    else:
        # middle child
        return head_result + [",", kwargs["deprel"]] + dep_result


def has_children(node):
    return len(node.children) > 0


EDGE_FUNCTIONS_LIST = [
    NamedFunction("concat", lambda head_result, dep_result, **kwargs: head_result + dep_result if kwargs["dep"]["id"] >
                                                                                                  kwargs["head"][
                                                                                                      "id"] else dep_result + head_result,
                  False),
    NamedFunction("concat-deprel",
                  lambda head_result, dep_result, **kwargs: head_result + [kwargs["deprel"]] + dep_result if
                  kwargs["dep"]["id"] > kwargs["head"]["id"] else dep_result + [kwargs["deprel"]] + head_result, True),
    NamedFunction("rev-deprel",
                  lambda head_result, dep_result, **kwargs: dep_result + [kwargs["deprel"]] + head_result if
                  kwargs["dep"]["id"] > kwargs["head"]["id"] else head_result + [kwargs["deprel"]] + dep_result, True),
    NamedFunction("rev", lambda head_result, dep_result, **kwargs: dep_result + head_result if kwargs["dep"]["id"] >
                                                                                               kwargs["head"][
                                                                                                   "id"] else head_result + dep_result,
                  False),

    NamedFunction("improved-triple",
                  lambda head_result, dep_result, **kwargs: head_result + ["(", kwargs["head"]["lemma"],
                                                                           kwargs["deprel"],
                                                                           kwargs["dep"]["lemma"], ")"] + (
                                                                dep_result if has_children(kwargs["dep_tree"]) else []),
                  True),

    NamedFunction("improved-triple-by",
                  lambda head_result, dep_result, **kwargs: head_result + ["(", kwargs["dep"]["lemma"],
                                                                           kwargs["deprel"], "by",
                                                                           kwargs["head"]["lemma"], ")"] + (
                                                                dep_result if has_children(kwargs["dep_tree"]) else []),
                  True),

    NamedFunction("bracket-1",
                  lambda head_result, dep_result, **kwargs: head_result + ["(", kwargs["deprel"]] + dep_result + [")"],
                  True),

    NamedFunction("bracket-invert-1",
                  lambda head_result, dep_result, **kwargs: dep_result + ["(", kwargs["deprel"], "by"] + head_result + [
                      ")"],
                  True),
    # NamedFunction("bracket-full", bracket_full),
    NamedFunction("bracket-2",
                  lambda head_result, dep_result, **kwargs: ["("] + head_result + [kwargs["deprel"]] + dep_result + [
                      ")"], True),
    NamedFunction("bracket-invert-2",
                  lambda head_result, dep_result, **kwargs: ["("] + dep_result + [kwargs["deprel"],
                                                                                  "by"] + head_result + [
                                                                ")"], True),

    NamedFunction("bracket-3", lambda head_result, dep_result, **kwargs: head_result + ["("] + dep_result + [")"],
                  False),

    NamedFunction("bracket-4",
                  lambda head_result, dep_result, **kwargs: head_result + [kwargs["deprel"], "("] + dep_result + [")"],
                  True),

    NamedFunction("ignore-dep", lambda head_result, dep_result, **kwargs: head_result, False),

    NamedFunction("bracket-full", bracket_full,
                  True),
]

EDGE_FUNCTIONS = {f.name: f for f in EDGE_FUNCTIONS_LIST}

EDGE_FUNCTIONS_WEIGHTS = [x.weight for x in EDGE_FUNCTIONS_LIST]
EDGE_FUNCTIONS_WEIGHTS_NON_IGNORE = [x.weight if not "ignore" in x.name else 0.0 for x in EDGE_FUNCTIONS_LIST]


def inside_out_order(token_tree):
    head = token_tree.token
    yield head
    children = list(token_tree.children)
    left_children = []
    right_children = []
    for c in children:
        if c.token["id"] < head["id"]:
            left_children.append(c)
        else:
            right_children.append(c)

    left_children.sort(key=lambda x: x.token["id"], reverse=True)
    right_children.sort(key=lambda x: x.token["id"])

    for l in left_children:
        yield l
    for r in right_children:
        yield r


def get_all_tokens(token_tree) -> list[dict]:
    """
    Read all descendants in depth-first order.
    :param token_tree:
    :return:
    """
    result = [token_tree.token]
    for child in token_tree.children:
        result.extend(get_all_tokens(child))
    return result


def read_off_string(token_tree) -> list[str]:
    tokens = get_all_tokens(token_tree)
    tokens.sort(key=lambda x: x["id"])
    return [t["form"] for t in tokens]


DUMMY_TERMINAL_SYMBOL = "<extra_id_97>"
SEP_1 = "<extra_id_96>"
SEP_2 = "<extra_id_95>"
SEP_3 = "<extra_id_94>"


class DepGrammarEval:
    def __init__(self, edge_rules: list[EdgeRule], replacements: dict[str, list[str]],
                 token_rules: list[TokenRule] = ()):
        self.edge_rules = dict()
        for rule in edge_rules:
            info = rule.get_info()
            if info not in self.edge_rules:
                self.edge_rules[info] = []
            self.edge_rules[info].append(rule)
        self.edge_rule_list = edge_rules

        self.replacements = replacements

        # self.token_rules = dict()
        # for rule in token_rules:
        #     info = rule.get_info()
        #     if info not in self.token_rules:
        #         self.token_rules[info] = []
        #     self.token_rules[info].append(rule)

    def apply(self, token_tree) -> list[str]:
        children_it = iter(inside_out_order(token_tree))
        head = next(children_it)
        children_it = list(children_it)
        if head["form"] in self.replacements:
            head_result = list(self.replacements[head["form"]])
        else:
            head_result = [head["form"]]

        all_edge_rules = []
        for i, child in enumerate(children_it):
            # find matching rule
            info = EdgeRule.extract_info(head, child.token)
            function_applied = False
            if info in self.edge_rules:
                for potential_matching_rule in self.edge_rules[info]:
                    if potential_matching_rule.matches(head, child.token):
                        all_edge_rules.append(potential_matching_rule)
                        function_applied = True
                        break
            if not function_applied:
                all_edge_rules.append(None)

        for i, child in enumerate(children_it):
            child_result = self.apply(child)
            function_applied = False
            # find matching rule
            info = EdgeRule.extract_info(head, child.token)
            if info in self.edge_rules:
                for potential_matching_rule in self.edge_rules[info]:
                    if potential_matching_rule.matches(head, child.token):
                        new_head_result = potential_matching_rule(head_result, child_result, head=head,
                                                                  dep=child.token,
                                                                  dep_tree=child,
                                                                  deprel=potential_matching_rule.output_label_token or potential_matching_rule.label,
                                                                  first_child=(i == 0),
                                                                  index=i,
                                                                  all_edge_rules=all_edge_rules,
                                                                  last_child=(i == len(children_it) - 1))

                        assert all(isinstance(tok, str) for tok in new_head_result)
                        head_result = new_head_result
                        function_applied = True
                        break

            if not function_applied:
                if child.token["id"] < head["id"]:
                    head_result = child_result + head_result
                else:
                    head_result = head_result + child_result
        # apply last unary function

        return head_result

    def encode_example(self, input_tree, tokenized_input: list[str], label2i, tokenizer):
        output = self.apply(input_tree)

        assert len(self.replacements) == 0
        input = []
        # ~ for input_word, replacement in self.replacements.items():
        # ~ input.extend(tokenizer.convert_ids_to_tokens(
        # ~ tokenizer([input_word], is_split_into_words=True, add_special_tokens=False)["input_ids"]))
        # ~ input.append(SEP_1)
        # ~ input.extend(replacement)
        # ~ input.append(SEP_2)
        # ~ input.append(SEP_3)

        input.extend(tokenized_input)

        output_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(output, is_split_into_words=True, add_special_tokens=False)["input_ids"])

        return {"grammar": self.encode_as_rules(label2i), "data": [(input, output_tokens)]}

    def encode_as_rules(self, label2i) -> list[ProductionRule]:
        """
        Map non-terminals to integer ids, ensuring all ids are positive.
        :return:
        """
        out = []
        for prod in self.edge_rule_list:
            out.append(ProductionRule(label2i[prod.label] + 1, prod.function.name,
                                      prod.function.id,
                                      "", [prod.output_label_token or ""]))

        out.sort(key=lambda x: x.lhs)
        return out


def get_dependent_labels(tok_tree):
    l = []
    for child in tok_tree.children:
        l.append(child.token["deprel"])
    return tuple(l)


def tree_map(tok_tree, f, aggr):
    l = [f(tok_tree)]
    for child in tok_tree.children:
        l.append(f(child))
    return aggr(l)


def sample_replacements(tokens, token_sampler, tokenizer, poisson_lambda=2):
    replacements = dict()
    for word in tokens:
        num_repl = 1 + np.random.poisson(poisson_lambda)
        tokenized = tokenizer.tokenize(word)
        if len(tokenized) > 0:
            replacements[word] = [token_sampler.sample(tokenized[0]) for _ in range(num_repl)]
    return replacements


def parse_incr(in_file):
    if not hasattr(in_file, 'read'):
        raise FileNotFoundError("Invalid file, 'parse_incr' needs an opened file as input")

    fields = parse_conllu_plus_fields(in_file)

    def generator():
        for sentence in parse_sentences(in_file):
            try:
                yield parse_token_and_metadata(
                    sentence,
                    fields=fields,
                )
            except conllu.exceptions.ParseException as ex:
                print("Ignoring sentence because of exception", ex, file=sys.stderr)

    return SentenceGenerator(generator())

def dump_jsonl(fname, data):
    with open(fname, "w") as f:
        for dp in data:
            f.write(json.dumps(dp))
            f.write("\n")


def rule_to_dict(rule, l2i):
    # + 1 for dependency relation comes from the fact that we use 0 for PADDING.
    return {"ud_label": rule.label, "ud_id": l2i.l2i[rule.label] + 1, "function": rule.function.name, "function_id": rule.function.id}

if __name__ == "__main__":
    import random

    random.seed(6348233)
    np.random.seed(63458734)
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

    # CHANGES:
    # Removed the restriction that some edge labels cannot be assigned the IGNORE function

    with open("grammars/pt_data_step_l2i.pkl", "rb") as f:
        l2i = pickle.load(f)

    max_input_length = 40

    data = []
    data_size = 5000
    relation2instances = defaultdict(list)
    deprelcounter = Counter()

    with lzma.open(
            "data/pretrain/output_batched_00000_200k.conll.xz",
            "rt") as data_file:
        i = 0
        for tokenlist in tqdm.tqdm(parse_incr(data_file)):

            if len(tokenlist) > max_input_length:
                continue

            tokentree = tokenlist.to_tree()
            data.append(tokentree)

            deprels_present = list({token["deprel"] for token in get_all_tokens(tokentree)})

            deprelcounter.update(deprels_present)

            for deprel in deprels_present:
                relation2instances[deprel].append(i)

            i += 1

            if len(data) == data_size:
                break

    # Sample a couple of reasonably frequent relations.
    frequent_relations = [relation for relation, count in deprelcounter.most_common()
                          if count > 600 and relation != "root"] # attaching a function to root has no effect.

    print(frequent_relations)

    for task_id in range(10):

        sampled_relations = swor_gumbel_uniform(len(frequent_relations), min(len(frequent_relations), 5))
        # sampled_relations = swor_gumbel_uniform(min(len(frequent_relations), 10), min(len(frequent_relations), 10))

        sampled_relations = [frequent_relations[r] for r in sampled_relations]

        print(sampled_relations)

        rules = []
        for label in sampled_relations:
            f = None
            while f is None or f.name == "concat":
                # exclude concat because concat is the default, and we might not be able to extract this because there are two ways of expressing "concat": adding concat or simply omitting it.
                f = EDGE_FUNCTIONS_LIST[weighted_choice(EDGE_FUNCTIONS_WEIGHTS)]
            rules.append(EdgeRule(label, f, None))

        g = DepGrammarEval(rules, dict())

        output_data = []
        for tokentree in data:
            tokenized_input = tokenizer.convert_ids_to_tokens(
                tokenizer(read_off_string(tokentree), is_split_into_words=True, add_special_tokens=False)["input_ids"])
            dp = g.encode_example(tokentree, tokenized_input, l2i, tokenizer)
            output_data.append({"input": dp["data"][0][0], "output": dp["data"][0][1]})


        # Divide into train/test data.
        data_ids = list(range(len(output_data)))
        random.shuffle(data_ids)
        train_ids = data_ids[:int(0.8 * len(output_data))]
        test_ids = data_ids[int(0.8 * len(output_data)):]
        test_ids_set = set(test_ids)

        train_data = [output_data[i] for i in train_ids]
        test_data = [output_data[i] for i in test_ids]

        dump_jsonl(f"finetune_analysis_{task_id}_train.jsonl", train_data)
        dump_jsonl(f"finetune_analysis_{task_id}_test.jsonl", test_data)

        with open(f"finetune_analysis_{task_id}_transformation.json", "w") as f:
            json.dump([rule_to_dict(rule, l2i) for rule in rules], f,  indent=True)

        # Now, we pick a label and change the function that is assigned to it.

        intervened_function = None
        while intervened_function is None or intervened_function.name == rules[1].function.name:
            intervened_function = EDGE_FUNCTIONS_LIST[weighted_choice(EDGE_FUNCTIONS_WEIGHTS)]

        intervened_label = rules[1].label
        print("Changing assignment of", intervened_label, "from", rules[1].function.name, "to", intervened_function.name)
        new_rule = EdgeRule(intervened_label, intervened_function, None)
        # Document this intervention:
        with open(f"finetune_analysis_{task_id}_intervention_transformation.json", "w") as f:
            json.dump({"old": rule_to_dict(rules[1], l2i), "new": rule_to_dict(new_rule, l2i)}, f, indent=True)
        rules[1] = new_rule

        g = DepGrammarEval(rules, dict())

        intervened_data = []
        for tokentree in data:
            tokenized_input = tokenizer.convert_ids_to_tokens(
                tokenizer(read_off_string(tokentree), is_split_into_words=True, add_special_tokens=False)["input_ids"])
            dp = g.encode_example(tokentree, tokenized_input, l2i, tokenizer)
            intervened_data.append(dp["data"][0])

        intervened_subset = [{"input": intervened_data[index][0], "output": intervened_data[index][1]} for index in relation2instances[intervened_label] if index in test_ids]

        dump_jsonl(f"finetune_analysis_{task_id}_intervention.jsonl", intervened_subset)
        dump_jsonl(f"finetune_analysis_{task_id}_intervention_data_wo_intervention.jsonl", [output_data[index]
                                                                                             for index in relation2instances[intervened_label] if index in test_ids])
















