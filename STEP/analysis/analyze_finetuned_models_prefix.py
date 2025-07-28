"""
See beginning of Section 5.2 in EMNLP paper. See create_task_for_finetune_analysis.py for data generation.
"""

import json

import pandas as pd
import transformers

from STEP.analysis.head_multi_masking import MaskedLM, estimate_masking_acc_from_heads, \
    IncrementalVocab, get_model_greedy_acc, interpretable_heads, all_interpretable_heads, create_head_masked_model, \
    estimate_masking_acc
from STEP.analysis.head_multi_masking_prefix import PrefixMaskedLM, \
    estimate_prefix_masking_acc_from_heads, estimate_masking_acc_random_tokens
from STEP.data_loading import prepare_task_dataset_jsonl
from STEP.sip_grammar import StructuredPrefixEmbeddingModelForCFG


import pickle
from STEP.sip_grammar import UDPretrainingModel
from STEP.data_loading import load_ud_grammar_pickle
import torch


def argmax2d(tensor):
    # runs argmax along the two last dimensions
    dim, row, col = tensor.shape
    flat_tensor = torch.flatten(tensor, 1, 2) #shape (dim, N)
    flat_indices = torch.argmax(flat_tensor, dim=-1) #shape (dim,)
    return torch.stack([flat_indices // col, flat_indices % col], -1)

def argmin2d(tensor):
    # runs argmax along the two last dimensions
    dim, row, col = tensor.shape
    flat_tensor = torch.flatten(tensor, 1, 2) #shape (dim, N)
    flat_indices = torch.argmin(flat_tensor, dim=-1) #shape (dim,)
    return torch.stack([flat_indices // col, flat_indices % col], -1)


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

if __name__ == "__main__":
    with open("grammars/pt_data_step_l2i.pkl",
              "rb") as f:
        l2i = pickle.load(f)

    index_to_ud_label = ["PAD"] + l2i.i2l

    sip_pretrained_model = UDPretrainingModel.from_pretrained(
        "models/step_model").to(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

    df = {"id": [], "no_masking_acc": [], "masking_interpretable": [], "other_interpretable": [], "random_masking": [],
          "prefix_p": [], "prefix_r": [], "prefix_f": []}

    # We compute the embeddings of all edge-wise transformations (for extracting what ended up in the prefix)
    with torch.no_grad():
        ud_embeddings = sip_pretrained_model.ud_embeddings.weight.data[:len(index_to_ud_label),
                        :]  # shape (num UD labels, hidden dim)
        function_embeddings = sip_pretrained_model.function_embeddings.weight.data[:len(EDGE_FUNCTIONS_LIST),
                              :]  # shape (num functions, hidden dim)

        added_embeddings = ud_embeddings.unsqueeze(1) + function_embeddings.unsqueeze(
            0)  # shape (num UD labels, num functions, hidden dim)

    for id in range(10):
        data_loader = prepare_task_dataset_jsonl(f"data/analysis/finetune_analysis/finetune_analysis_{id}_test.jsonl",
                                                 tokenizer, 24)
        model = StructuredPrefixEmbeddingModelForCFG.from_pretrained(f"models/step_model_full_finetune_analysis_{id}").to(0)

        print("Accuracy without masking")
        acc = get_model_greedy_acc(model, data_loader)
        print(acc)
        df["id"].append(id)
        df["no_masking_acc"].append(acc)

        with open(f"data/analysis/finetune_analysis/finetune_analysis_{id}_transformation.json") as f:
            transformation = json.loads(f.read())

        relations_needed = [edgewise["ud_label"] for edgewise in transformation]

        heads_needed = set()
        for rel in relations_needed:
            heads_needed.update(interpretable_heads.get(rel, []))

        masked_important_heads = get_model_greedy_acc(PrefixMaskedLM(model, model.prefix_embedding.shape[1], list(heads_needed), num_layer=12,
                                                num_heads=12), data_loader)

        df["masking_interpretable"].append(masked_important_heads)

        print("Masking heads that are likely involved")
        print(masked_important_heads)

        print("Masking other random lookup heads")
        other_lookup_heads = list(all_interpretable_heads - heads_needed)

        other_masking = estimate_prefix_masking_acc_from_heads(model, model.prefix_embedding.shape[1], data_loader, len(heads_needed), other_lookup_heads, n=50)
        print(other_masking)
        df["other_interpretable"].append(other_masking)

        print("Masking random heads")
        random_masking_acc = estimate_masking_acc_random_tokens(model, model.prefix_embedding.shape[1], data_loader, len(heads_needed), n=50)
        print(random_masking_acc)
        df["random_masking"].append(random_masking_acc)

        # Now we want try to extract the information from the learned prefix.

        with torch.no_grad():
            prefix = model.prefix_embedding.unsqueeze(0)
            added_embeddings_norm = added_embeddings / added_embeddings.square().sum(dim=-1, keepdims=True).sqrt()
            prefix_norm = prefix / prefix.square().sum(dim=-1, keepdims=True).sqrt()

            added_embeddings_norm = added_embeddings_norm.squeeze()
            prefix_norm = prefix_norm.squeeze()
            similarities = torch.einsum("ufh, ph -> puf", added_embeddings_norm, prefix_norm)
            similarities_np = similarities.cpu().numpy()

        trafo_short = {(d["ud_label"], d["function"]) for d in transformation}
        predicted_assignment = set()
        for i, row in enumerate(argmax2d(similarities).cpu().numpy()):
            ud, f = row
            a = (index_to_ud_label[ud], EDGE_FUNCTIONS_LIST[f])
            if a[0] != "PAD":
                predicted_assignment.add(a)
            print(i, index_to_ud_label[ud], EDGE_FUNCTIONS_LIST[f], "sim", similarities[i, ud, f], "OK?",
                  a in trafo_short)

        P = 1 if len(predicted_assignment) == 0 else len(predicted_assignment & trafo_short) / len(predicted_assignment)
        R = len(predicted_assignment & trafo_short) / len(trafo_short)
        F = 2 * P * R / (P + R)
        print("Precision", P)
        print("Recall", R)
        print("F", F)
        df["prefix_f"].append(F)
        df["prefix_p"].append(P)
        df["prefix_r"].append(R)

    df = pd.DataFrame(df)
    df.to_csv("finetune_analysis_prefix.csv")



