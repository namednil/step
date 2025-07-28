import dataclasses
import lzma
import pickle
import sys
from typing import Callable, Optional

import conllu

from enum import Enum

import gzip

import tqdm

import numpy as np
from conllu import parse_conllu_plus_fields, parse_sentences, parse_token_and_metadata, SentenceGenerator

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


def get_ids_contained(fname):
    with lzma.open(fname, "rb") as f:
        train_data = pickle.load(f)
        for dp in tqdm.tqdm(train_data):
            yield dp["task_id"]
            
def dump_ids(ids, sents, fname):
    with gzip.open(fname, "wt") as f:
        for id in ids:
            sents[id].metadata["task_id"] = id
            f.write(sents[id].serialize())

if __name__ == "__main__":
    infile = "pt_data_step"
    
    sents = []
    with lzma.open(
            "data/pretrain/output_batched_00000_200k.conll.xz",
            "rt") as data_file:
        for i, tokenlist in tqdm.tqdm(enumerate(parse_incr(data_file))):
            
            sents.append(tokenlist)
    
    print("File read")
    
    train_ids = set(get_ids_contained(f"grammars/{infile}_train.pkl.xz"))
    print("Train ids extracted")
    easy_dev_ids = set(get_ids_contained(f"grammars/{infile}_easy_dev.pkl.xz"))
    dev_ids = set(get_ids_contained(f"grammars/{infile}_dev.pkl.xz"))
    test_ids = set(get_ids_contained(f"grammars/{infile}_test.pkl.xz"))
    
    print("All ids extracted")
    
    dump_ids(train_ids, sents, f"grammars/{infile}_used_train_with_ids.conllu.gz")
    dump_ids(easy_dev_ids, sents, f"grammars/{infile}_used_easy_dev_with_ids.conllu.gz")
    dump_ids(dev_ids, sents, f"grammars/{infile}_used_dev_with_ids.conllu.gz")
    dump_ids(test_ids, sents, f"grammars/{infile}_used_test_with_ids.conllu.gz")
