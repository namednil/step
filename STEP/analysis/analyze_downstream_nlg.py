"""
Masking experiment for figure 5 in the EMNLP paper.
"""

from collections import defaultdict

import pandas as pd
import torch
import transformers

from STEP.data_loading import prepare_task_dataset
from STEP.sip_grammar import StructuredPrefixEmbeddingModelForCFG


from STEP.nlg_eval import nlg_eval

from STEP.analysis.head_multi_masking_prefix import PrefixMaskedLM, \
    estimate_prefix_masking_acc_from_heads, RandomPositionMaskedLM

from STEP.analysis.head_multi_masking import all_interpretable_heads


def estimate_masking_random_tokens_nlg_eval(sip_pretrained_model, num_tokens, tokenizer, data_loader, num_heads, n=5):
    total_metrics = defaultdict(float)
    for i in range(n):
        masked_model = RandomPositionMaskedLM(sip_pretrained_model, num_tokens, num_heads, num_layer=12, num_heads=12)
        metrics = nlg_eval(masked_model, tokenizer, data_loader)
        for m, val in metrics.items():
            total_metrics[m] += val
    return {m: val / n for m, val in total_metrics.items()}


class TakeNLoop:
    def __init__(self, iterator, n):
        self.iterator = iterator
        self.n = n
        self.i = 0

    def __iter__(self):
        self.i = 0
        self.it = iter(self.iterator)
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.n:
            raise StopIteration()
        return next(self.it)

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
    import sys

    df = {"task": [], "run": [], "masking": [], "metric": [], "value": []}

    run = sys.argv[1]

    for task in ["ATP", "AEM", "VEM"]:
        # for run in range(1, 6):
        model = StructuredPrefixEmbeddingModelForCFG.from_pretrained(f"models/ud_repl2_{task}_all_{run}").to(0)
        data_loader = prepare_task_dataset(f"data/finetune/{task}/test_all_100_{run}.tsv", tokenizer, 24, lenient=True)

        # data_loader = TakeNLoop(data_loader, 1)

        nlg_results = nlg_eval(model, tokenizer, data_loader)
        print("Without masking", nlg_results)
        for metric, val in nlg_results.items():
            df["task"].append(task)
            df["run"].append(run)
            df["masking"].append("no_masking")
            df["metric"].append(metric)
            df["value"].append(val)

        print(f"Masking interpretable heads, specificically to the prefix of length {model.prefix_embedding.shape[1]}")
        masked_model = PrefixMaskedLM(model, model.prefix_embedding.shape[1], list(all_interpretable_heads), num_layer=12,
                                      num_heads=12)
        nlg_results = nlg_eval(masked_model, tokenizer, data_loader)
        print(nlg_results)
        for metric, val in nlg_results.items():
            df["task"].append(task)
            df["run"].append(run)
            df["masking"].append("interpretable_masking_prefix")
            df["metric"].append(metric)
            df["value"].append(val)

        # print(f"Masking interpretable heads completely {model.prefix_embedding.shape[1]}")
        # masked_model = create_head_masked_model(model, list(all_interpretable_heads), num_layer=12,
        #                                     num_heads=12)
        # nlg_results = nlg_eval(masked_model, tokenizer, data_loader)
        # print(nlg_results)
        # for metric, val in nlg_results.items():
        #     df["task"].append(task)
        #     df["run"].append(run)
        #     df["masking"].append("interpretable_masking_complete")
        #     df["metric"].append(metric)
        #     df["value"].append(val)


        print("Masking random heads")
        nlg_results = estimate_masking_random_tokens_nlg_eval(model, model.prefix_embedding.shape[1], tokenizer, data_loader, len(all_interpretable_heads),
                                                              n=20)
        print(nlg_results)
        for metric, val in nlg_results.items():
            df["task"].append(task)
            df["run"].append(run)
            df["masking"].append("random_masking")
            df["metric"].append(metric)
            df["value"].append(val)

    df = pd.DataFrame(df)
    df.to_csv(f"finetune_analysis_downstream_nlg_{run}.csv")
