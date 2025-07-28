import gzip

from STEP.data_gen.create_ud_tasks_step import parse_incr, read_off_string

import tqdm

import random
def linearize_tree(token_tree):
    if len(token_tree.children) == 0:
        return [token_tree.token["form"]]
    r = ["(", token_tree.token["form"]]
    for i, child in enumerate(token_tree.children):
        r.append(child.token["deprel"])
        r.extend(linearize_tree(child))
    r.append(")")
    return r

def write_to_file(fname, data):
    with open(fname, "w") as f:
        for x,y in data:
            f.write(x)
            f.write("\t")
            f.write(y)
            f.write("\n")


DUMMY_PARSE_SYMBOL = "<extra_id_97> "

def process_examples(fname):
    with gzip.open(
            fname,
            "rt") as data_file:
        for i, tokenlist in tqdm.tqdm(enumerate(parse_incr(data_file))):

            if i % 5000 == 0:
                print()

            tokentree = tokenlist.to_tree()
            
            inp = DUMMY_PARSE_SYMBOL+ " ".join([x["form"] for x in tokenlist])
            outp = " ".join(linearize_tree(tokentree))
            
            yield inp, outp
                
if __name__ == "__main__":
    random.seed(23846234)
    outf = "pt_data_t5_plus_dep_parse"

    with open(f"grammars/{outf}_train.tsv", "w") as f:
        for x, y in process_examples("grammars/pt_data_step_used_train_with_ids.conllu.gz"):
            f.write(x)
            f.write("\t")
            f.write(y)
            f.write("\n")

    with open(f"grammars/{outf}_test.tsv", "w") as f:
        for x, y in process_examples("grammars/pt_data_step_used_test_with_ids.conllu.gz"):
            f.write(x)
            f.write("\t")
            f.write(y)
            f.write("\n")

    print("done.")

