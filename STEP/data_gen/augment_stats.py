import gzip
from collections import defaultdict, Counter

import tqdm

import sys


import matplotlib.pyplot as plt

import conllu
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

def recursion_depth(tree, relation_name):
    return int(tree.token["deprel"] == relation_name) + (max(recursion_depth(c, relation_name) for c in tree.children) if tree.children else 0)

def exists_path_with_label(from_tree, to_tree, label) -> tuple[bool, bool]:
    path = False
    label_found = from_tree.token["deprel"] == label
    if from_tree == to_tree:
        path = True
    for child in from_tree.children:
        (subpath, sub_label) = exists_path_with_label(child, to_tree, label)
        path = path or subpath
        label_found = label_found or sub_label
    return (path, label_found)


def count_center_embedding_depth(tree):
    s = 0
    if tree.token["deprel"] == "acl:relcl" and tree.token["lemma"] != "be" and tree.token["upos"] != "ADJ":
        # the subject of this verb needs again to be modified by a relative clause
        found_center_embedding = False
        for child in tree.children:
            if child.token["deprel"] == "nsubj":
                for grandchild in child.children:
                    if grandchild.token["deprel"] == "acl:relcl":
                        found_center_embedding = True
                        break
        s += found_center_embedding
    return s + (max(count_center_embedding_depth(child) for child in tree.children) if tree.children else 0)



if __name__ == "__main__":
    import lzma
    with lzma.open(f"data/output_batched_00000_200k.conll.xz", "rt") as f:
        l2aux = defaultdict(list)
        depth = Counter()
        xcomp_depth = Counter()
        center_depth = Counter()

        for i, tokenlist in enumerate(tqdm.tqdm(parse_incr(f))):
            tokentree = tokenlist.to_tree()

            if len(tokenlist) > 90:
                continue

            d= recursion_depth(tokentree, "nmod")
            dxcomp= recursion_depth(tokentree, "xcomp")
            depth.update([d])
            xcomp_depth.update([dxcomp])

            cdepth = count_center_embedding_depth(tokentree)
            center_depth.update([cdepth])
            if cdepth >= 2:
                print(cdepth)
                print(tokenlist.serialize())
            # if d >= 5 or dxcomp >= 5:
            #     pass
            #     print()
            #     print(tokenlist.serialize())
            #     print("==")

            if i % 5000 == 0:
                print("nmod", depth)
                print("xcomp", xcomp_depth)
                print("center depth", center_depth)

        print("nmod", depth)
        print("xcomp", xcomp_depth)
        print("center depth", center_depth)
