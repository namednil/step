import gzip
import json
import lzma
import pickle
import random
import sys
from typing import Tuple, List, Callable

import numpy as np
import torch
import transformers
from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler, RandomSampler, BatchSampler, DataLoader, SequentialSampler
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizerFast

import datasets

import tqdm

from STEP.data_gen.grammar_gen import ProductionRule


def load_tsv(fname, expect_first_line = None, lenient: bool = False):
    with open(fname) as f:
        it = iter(f)
        if expect_first_line is not None:
            first_line = next(it).strip()
            if expect_first_line != first_line:
                if lenient:
                    line = first_line.strip("\n").strip("\r")
                    if line:
                        yield line.split("\t")
                else:
                    raise ValueError(f"First line must be: '{expect_first_line}'")
        for line in it:
            line = line.strip("\n").strip("\r")
            if line:
                yield line.split("\t")


def weighted_dict_choice(remaining):
    Z = sum(remaining.values())
    cutoff = random.random() * Z
    z = 0
    for key in remaining:
        z += remaining[key]
        if z >= cutoff:
            return key


class BatchSamplerWithSameLength(Sampler):
    def __init__(self, data, batch_size, key="input_ids") -> None:
        super().__init__()
        self.data = data
        self.l2indices = dict()
        self.batch_size = batch_size
        for i, dp in enumerate(data):
            length = len(dp[key])
            if length not in self.l2indices:
                self.l2indices[length] = []
            self.l2indices[length].append(i)

    def __len__(self) -> int:
        return sum((len(bin_) + self.batch_size - 1) // self.batch_size for bin_ in self.l2indices)

    def __iter__(self):
        for length, indices in self.l2indices.items():
            random.shuffle(indices)
        position = {length: 0 for length in self.l2indices}
        remaining = {length: len(self.l2indices[length]) for length in self.l2indices}

        while sum(remaining.values()) > 0:
            # print("Remaining", remaining)
            length = weighted_dict_choice(remaining)
            # print("Chosen", length)
            new_position = min(position[length] + self.batch_size, len(self.l2indices[length]))
            yield self.l2indices[length][position[length]:new_position]
            remaining[length] = remaining[length] - (new_position - position[length])


def prepare_task_dataset(path:str, tokenizer: AutoTokenizer, batch_size: int, random_order: bool = True, lenient: bool=False,
                         same_length_batches: bool = False, force_output_right_padding: bool = False, prompt:str = "") -> DataLoader:
    def mapper(examples):
        d = tokenizer(prompt+examples["input"])
        padding_side = tokenizer.padding_side
        if "output" in examples:
            if force_output_right_padding:
                tokenizer.padding_side = "right"
            d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
            tokenizer.padding_side = padding_side
            if d["labels"][-1] != tokenizer.eos_token_id:
                # gpt neo tokenizer doesn't add EOS token, so do this explicitly.
                d["labels"].append(tokenizer.eos_token_id)
        return d

    keys = ["input", "output"]
    d = {k: [] for k in keys}
    for row in load_tsv(path, "input\toutput", lenient=lenient):
        for x, k in zip(row, keys):
            d[k].append(x)
    dataset = Dataset.from_dict(d)

    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])
    if same_length_batches:
        ts = BatchSamplerWithSameLength(dataset, batch_size=batch_size)
    return DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)



def prepare_emphasis_dataset(path:str, tokenizer: AutoTokenizer, batch_size: int, random_order: bool = True, lenient: bool=True) -> DataLoader:
    def mapper(examples):
        inp, emph = examples["input"].split(";")
        inp = inp.strip()
        emph = emph.strip()
        d = tokenizer(inp)
        emph_tokens = tokenizer(emph)["input_ids"]

        d["emph"] = [ tok in emph_tokens and tok != tokenizer.bos_token_id and tok != tokenizer.eos_token_id for tok in d["input_ids"]]

        padding_side = tokenizer.padding_side
        if "output" in examples:
            d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
            tokenizer.padding_side = padding_side
            if d["labels"][-1] != tokenizer.eos_token_id:
                # gpt neo tokenizer doesn't add EOS token, so do this explicitly.
                d["labels"].append(tokenizer.eos_token_id)
        return d

    keys = ["input", "output"]
    d = {k: [] for k in keys}
    for row in load_tsv(path, "input\toutput", lenient=lenient):
        for x, k in zip(row, keys):
            d[k].append(x)
    dataset = Dataset.from_dict(d)

    seq2seq_collator = DataCollatorForSeq2Seq(tokenizer)
    def collator_fn(features):
        emph = []
        for f in features:
            emph.append(torch.from_numpy(np.array(f.pop("emph"))))
        padded = torch.nn.utils.rnn.pad_sequence(emph, padding_value=False)
        d = seq2seq_collator(features)
        d["emph_mask"] = padded.transpose(0, 1) #shape (batch, seq_len)

        return d

    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])
    return DataLoader(dataset, collate_fn=collator_fn, batch_sampler=ts)



def load_cogs_dataset(path:str, tokenizer: AutoTokenizer, batch_size: int, random_order: bool = True, lenient: bool=False,
                      same_length_batches: bool = False, prompt: str = "") -> DataLoader:
    def mapper(examples):
        d = tokenizer(prompt+examples["input"])
        if "output" in examples:
            d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
            if d["labels"][-1] != tokenizer.eos_token_id:
                # gpt neo tokenizer doesn't add EOS token, so do this explicitly.
                d["labels"].append(tokenizer.eos_token_id)
        return d

    keys = ["input", "output", "gen_type"]
    d = {k: [] for k in keys}
    for row in load_tsv(path):
        for x, k in zip(row, keys):
            d[k].append(x)
    dataset = Dataset.from_dict(d)

    collator = DataCollatorForSeq2Seq(tokenizer)
    def collator_fn(features):
        gen_types = []
        for x in features:
            if "gen_type" in x:
                gen_types.append(x["gen_type"])
                del x["gen_type"]
        d = collator(features)
        if len(gen_types) > 0:
            assert len(gen_types) == len(features), "Either all or no sentences should have a gen_type."
            d["gen_type"] = gen_types
        return d

    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])
    if same_length_batches:
        ts = BatchSamplerWithSameLength(dataset, batch_size=batch_size)
    return DataLoader(dataset, collate_fn=collator_fn, batch_sampler=ts)



def prepare_task_dataset_jsonl(path:str, tokenizer: AutoTokenizer, batch_size: int, random_order: bool = True) -> DataLoader:
    def mapper(examples):
        if isinstance(examples["input"], str):
            d = tokenizer(examples["input"])
        elif isinstance(examples["input"][0], str):
            #pre-tokenized but not mapped to ints yet
            i = tokenizer.convert_tokens_to_ids(examples["input"])
            assert len(i) == len(examples["input"])
            i = tokenizer.build_inputs_with_special_tokens(i)
            d = {"input_ids": i,
                 "attention_mask": [1] * len(i)}
        else:
            d = {"input_ids": examples["input"], "attention_mask": [1] * len(examples["input"])}
        if "output" in examples:
            if isinstance(examples["output"], str):
                d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]

            elif isinstance(examples["output"][0], str):
                #pre-tokenized
                o = tokenizer.convert_tokens_to_ids(examples["output"])
                assert len(o) == len(examples["output"])
                d["labels"] = tokenizer.build_inputs_with_special_tokens(o)
            else:
                d["labels"] = examples["output"]
        return d

    keys = ["input", "output"]
    d = {k: [] for k in keys}
    with open(path) as f:
        for row in f:
            j = json.loads(row)
            for k in keys:
                d[k].append(j[k])

    dataset = Dataset.from_dict(d)

    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])
    return DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)


def prepare_multi_meta(path:str, tokenizer: AutoTokenizer, train_batch_size: int, test_batch_size: int) -> List[Tuple[DataLoader, DataLoader]]:
    datasets.disable_progress_bar()
    def mapper(examples):
        d = tokenizer(examples["input"], text_target=examples["output"] if "output" in examples else None)
        return d
    task2data = dict()
    for row in load_tsv(path, "input\toutput\ttask\tis_train"):
        input, output, task, is_train = row
        if task not in task2data:
            task2data[task] = {"train": {"input": [], "output": []}, "test": {"input": [], "output": []}}
        if int(is_train):
            d = task2data[task]["train"]
        else:
            d = task2data[task]["test"]

        d["input"].append(input)
        d["output"].append(output)

    dataloaders = []
    for task in tqdm.tqdm(task2data):
        train_data = Dataset.from_dict(task2data[task]["train"])
        test_data = Dataset.from_dict(task2data[task]["test"])
        if len(train_data) == 0:
            raise ValueError(f"Task {task} has no train data")
        if len(test_data) == 0:
            raise ValueError(f"Task {task} has no test data")
        dls = []
        for data, batch_size in zip([train_data, test_data], [train_batch_size, test_batch_size]):
            data = data.map(mapper, batched=True, remove_columns=["input", "output"])
            sampler = BatchSampler(RandomSampler(data), batch_size=batch_size, drop_last=False)
            dataloader = DataLoader(data, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=sampler)
            dls.append(dataloader)
        dataloaders.append(tuple(dls))

    return dataloaders


def encode_grammar(rules: list[ProductionRule], max_rhs_length, tokenizer, epsilon_id) -> tuple[np.array, np.array]:
    d = np.zeros((len(rules), 3 + max_rhs_length), dtype=np.int64)
    nt_mask = np.zeros((len(rules), 3 + max_rhs_length), dtype=bool)
    nt_mask[:, 0] = False
    nt_mask[:, 1] = True
    nt_mask[:, 2] = False
    for i, rule in enumerate(rules):
        d[i, 0] = rule.fint
        d[i, 1] = rule.lhs
        if rule.map_term == "":
            d[i, 2] = epsilon_id
        else:
            # d[i, 2] = tokenizer(rule.map_term, add_special_tokens=False)["input_ids"][0]
            t = tokenizer.convert_tokens_to_ids(rule.map_term)
            assert isinstance(t, int)
            d[i, 2] = t

        for j, symbol in enumerate(rule.rhs):
            if isinstance(symbol, str):
                if symbol == "":
                    d[i, j + 3] = epsilon_id
                else:
                    # d[i, j + 3] = tokenizer(symbol, add_special_tokens=False)["input_ids"][0]
                    t = tokenizer.convert_tokens_to_ids(symbol)
                    assert isinstance(t, int)
                    d[i, j + 3] = t
                nt_mask[i, j + 3] = False
            else:
                d[i, j + 3] = symbol
                nt_mask[i, j + 3] = True
    return d, nt_mask



def load_ud_grammar_pickle(path: str, tokenizer: transformers.AutoTokenizer, batch_size:int, random_order: bool = True,
                        max_n:int=None,
                        filter_f = None,
                        grammar_map_f = None, output_replace_eos: tuple[str, str] = None, reverse_file: bool = False):
    def mapper(examples):

        if isinstance(examples["input"], str):
            d = tokenizer(examples["input"])
            if "output" in examples:
                if output_replace_eos is not None:
                    text = examples["output"].replace(output_replace_eos[0], output_replace_eos[1])
                else:
                    text = examples["output"]
                d["labels"] = tokenizer(text_target=text)["input_ids"]
        else:
            # Already "pre-tokenized", i.e. converted into a list and every element is a string
            # that can be mapped to an id
            assert isinstance(examples["input"][0], str)
            i = tokenizer.convert_tokens_to_ids(examples["input"])
            assert len(i) == len(examples["input"])
            i = tokenizer.build_inputs_with_special_tokens(i)
            d = {"input_ids": i,
                 "attention_mask": [1] * len(i)}
            if "output" in examples:
                o = tokenizer.convert_tokens_to_ids(examples["output"])
                assert len(o) == len(examples["output"])
                d["labels"] = tokenizer.build_inputs_with_special_tokens(o)

        return d

    data = {"input": [], "output": [], "ud_labels": [], "function_ids": [], "task_ids": []}
    with lzma.open(path, "rb") as f:
        load_data = pickle.load(f)
        if reverse_file:
            load_data.reverse()
        i = 0
        for dp in load_data:
            if filter_f is None or filter_f(dp):
                rules: list[ProductionRule] = dp["grammar"]
                if grammar_map_f is not None:
                    rules = grammar_map_f(rules)

                for input, output in dp["data"]:
                    data["task_ids"].append(dp["task_id"])
                    data["input"].append(input)
                    data["output"].append(output)
                    data["ud_labels"].append([rule.lhs for rule in rules])
                    data["function_ids"].append([rule.fint for rule in rules])

                    if max_n is not None and i > max_n:
                        break
                    i += 1
            if max_n is not None and i > max_n:
                break

    dataset = Dataset.from_dict(data)
    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])

    seq2seq_collator = DataCollatorForSeq2Seq(tokenizer)
    def collator_fn(features):
        ud_labels = []
        function_ids = []
        for x in features:
            ud_labels.append(x.pop("ud_labels"))
            function_ids.append(x.pop("function_ids"))

        d = seq2seq_collator(features)

        max_rules = max(len(el) for el in ud_labels)

        ud_labels_tensor = np.zeros((len(ud_labels), max_rules), dtype=np.int64)
        function_ids_tensor = np.ones((len(function_ids), max_rules), dtype=np.int64)
        for i in range(len(ud_labels)):
            ud_labels_tensor[i, : len(ud_labels[i])] = ud_labels[i]
            function_ids_tensor[i, : len(function_ids[i])] = function_ids[i]

        d["ud_labels"] = torch.from_numpy(ud_labels_tensor)
        d["function_ids"] = torch.from_numpy(function_ids_tensor)

        if "task_id" in features[0]:
            d["task_ids"] = torch.from_numpy(np.array([x["task_id"] for x in features]))

        return d

    return DataLoader(dataset, collate_fn=collator_fn, batch_sampler=ts)




def load_ud_grammar_pickle_json(path: str, tokenizer: transformers.AutoTokenizer, batch_size:int, random_order: bool = True,
                        max_n:int=None, output_replace_eos: tuple[str, str] = None):
    def mapper(examples):

        if isinstance(examples["input"], str):
            d = tokenizer(examples["input"])
            if "output" in examples:
                if output_replace_eos is not None:
                    text = examples["output"].replace(output_replace_eos[0], output_replace_eos[1])
                else:
                    text = examples["output"]
                d["labels"] = tokenizer(text_target=text)["input_ids"]
        else:
            # Already "pre-tokenized", i.e. converted into a list and every element is a string
            # that can be mapped to an id
            assert isinstance(examples["input"][0], str)
            i = tokenizer.convert_tokens_to_ids(examples["input"])
            assert len(i) == len(examples["input"])
            i = tokenizer.build_inputs_with_special_tokens(i)
            d = {"input_ids": i,
                 "attention_mask": [1] * len(i)}
            if "output" in examples:
                o = tokenizer.convert_tokens_to_ids(examples["output"])
                assert len(o) == len(examples["output"])
                d["labels"] = tokenizer.build_inputs_with_special_tokens(o)

        return d
    def generator():
        with gzip.open(path, "rt") as file_obj:
            for i, line in enumerate(file_obj):
                dp = json.loads(line)
                assert len(dp["function_ids"]) == len(dp["ud_labels"])
                dp["task_ids"] = dp.pop("task_id")
                yield dp

                if max_n is not None and i > max_n:
                    break

    dataset = IterableDataset.from_generator(generator)

    if random_order:
        dataset = dataset.shuffle(buffer_size=10_000)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])

    seq2seq_collator = DataCollatorForSeq2Seq(tokenizer)
    def collator_fn(features):
        ud_labels = []
        function_ids = []
        for x in features:
            ud_labels.append(x.pop("ud_labels"))
            function_ids.append(x.pop("function_ids"))

        d = seq2seq_collator(features)

        max_rules = max(len(el) for el in ud_labels)

        ud_labels_tensor = np.zeros((len(ud_labels), max_rules), dtype=np.int64)
        function_ids_tensor = np.ones((len(function_ids), max_rules), dtype=np.int64)
        for i in range(len(ud_labels)):
            ud_labels_tensor[i, : len(ud_labels[i])] = ud_labels[i]
            function_ids_tensor[i, : len(function_ids[i])] = function_ids[i]

        d["ud_labels"] = torch.from_numpy(ud_labels_tensor)
        d["function_ids"] = torch.from_numpy(function_ids_tensor)

        if "task_id" in features[0]:
            d["task_ids"] = torch.from_numpy(np.array([x["task_id"] for x in features]))

        return d

    return DataLoader(dataset, collate_fn=collator_fn, batch_size=batch_size)

def write_tsv(fname, data):
    with open(fname, "w") as f:
        for (x,y) in data:
            f.write(x)
            f.write("\t")
            f.write(y)
            f.write("\n")


class RandomSplit:

    def __init__(self, path: str, tokenizer: AutoTokenizer, num_train:int, train_batch_size, test_batch_size = None, lenient=True):
        def mapper(examples):
            d = tokenizer(examples["input"])
            if "output" in examples:
                d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
            return d

        keys = ["input", "output"]
        data = []
        for row in load_tsv(path, "input\toutput", lenient=lenient):
            data.append(row)
        print("Random number to verify seed", random.randint(0, 100_000_000), file=sys.stderr)
        random.shuffle(data)
        self.train_data = data[:num_train]
        self.rest_data = data[num_train:]

        train_dataset = Dataset.from_list([ {k: v for k,v in zip(keys, row)} for row in self.train_data])
        rest_dataset = Dataset.from_list([ {k: v for k,v in zip(keys, row)} for row in self.rest_data])

        sampler = SequentialSampler(train_dataset)
        ts = BatchSampler(sampler, batch_size=train_batch_size, drop_last=False)
        dataset = train_dataset.map(mapper, batched=True, remove_columns=["input", "output"])
        self.train_loader = DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)


        sampler = SequentialSampler(rest_dataset)
        ts = BatchSampler(sampler, batch_size=train_batch_size if test_batch_size is None else test_batch_size, drop_last=False)
        dataset = rest_dataset.map(mapper, batched=True, remove_columns=["input", "output"])
        self.rest_loader = DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)


    def save_split(self, pathname):
        write_tsv(pathname+"_train.tsv", self.train_data)
        write_tsv(pathname+"_test.tsv", self.rest_data)

    def get_train_loader(self):
        return self.train_loader

    def get_rest_loader(self):
        return self.rest_loader


from datasets import load_dataset


def data_loader_from_dataset(dataset, tokenizer,
                             input_template:str, output_template: str,
                             batch_size: int, random_order: bool = True):
    def map_and_tokenize(row):
        input_text = input_template.format(**row)
        output_text = output_template.format(**row)
        d = tokenizer(input_text)
        d["labels"] = tokenizer(text_target=output_text)["input_ids"]
        return d

    dataset = dataset.map(map_and_tokenize, batched=False, remove_columns=dataset.column_names)

    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    ts = BatchSampler(sampler, batch_size=batch_size,
                      drop_last=False)

    return DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)



class ComposeDataLoaders:
    """
    The first iteration goes over the first data loader, the second iteration goes over the second data loader etc.
    If we have k data loaders, the k+i (i > 0) iteration will go over the last (i.e. the k-th) data loader.

    This is useful for curriculum learning.
    """
    def __init__(self, dataloaders, counts = None):
        self.dataloaders = dataloaders
        self.index = 0
        if counts is not None:
            assert len(counts) == len(dataloaders)
        self.counts = counts

    @staticmethod
    def create(**kwargs):
        return ComposeDataLoaders(**kwargs)

    def __len__(self):
        return len(self.dataloaders[self.index])

    def __iter__(self):
        r = iter(self.dataloaders[self.index])
        self.counts[self.index] -= 1
        if self.counts[self.index] == 0 and self.index+1 < len(self.dataloaders):
            self.index += 1
        return r

