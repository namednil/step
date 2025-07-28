
import re
from typing import Optional, List, Tuple, Dict


from config_evaluator import Lazy
from logger import Logger, TqdmLogger

import torch
import numpy as np

from typing import Optional, Dict, Iterable, List, Tuple

import torch

import Levenshtein


def evaluate_on(model, tokenizer, dataloader):
  correct, total, edit_dist, per = 0,0,0,0
  model.eval()
  for test_batch in dataloader:
    test_batch = {k: v.to(model.device) for k,v in test_batch.items()}
    test_batch_inputs = dict(test_batch)
    del test_batch_inputs["labels"]
    r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1]+2,
                                              early_stopping="never", num_beams=1, no_repeat_ngram_size=0), skip_special_tokens=True)
    gold = tokenizer.batch_decode(100*(test_batch["labels"] == -100) + test_batch["labels"], skip_special_tokens=True) # replace -100 by 0
    for p, g in zip(r, gold):
      print(p, "\t|\t", g)
    correct += sum( [x == y for x,y in zip(r, gold)])
    total += len(gold)
    edit_dist += sum( Levenshtein.distance(x,y) for x,y in zip(r, gold))
    per += sum(Levenshtein.distance(x,y)/max(1, len(y)) for x,y in zip(r, gold))
  return correct/total, edit_dist/total, per/total                                                                                 

#################
from math import ceil
def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
    
def hack_t5_parallelize(model):
   model.encoder.parallelize(get_device_map(len(model.encoder.block), range(torch.cuda.device_count())))
   model.decoder.parallelize(get_device_map(len(model.decoder.block), range(torch.cuda.device_count())))
   model.lm_head = model.lm_head.to(model.decoder.first_device)
   model.model_parallel = True

   return model


def get_optimizer(model, optimizer, optimizer_groups):
    if optimizer is None:
        optimizer = Lazy(dict(), torch.optim.Adam)

    if optimizer_groups:
        # Example config
        # "optimizer_groups": [
        #     [".*prefix_embedding.*", {"lr": 1.0}],
        #     [".*lm_head.*", {"lr": 1e-5}],
        #     [".*", {"lr": 0.0}]  # all other parameters are frozen
        # ]

        groups = []
        for regex, hyperparam in optimizer_groups:
            h = dict(hyperparam)
            h["params"] = []
            groups.append(h)

        for name, param in model.named_parameters():
            for (regex, _), group in zip(optimizer_groups, groups):
                if re.match(regex, name):
                    group["params"].append(param)
                    break
        # Exclude groups with learning rate 0
        new_groups = []
        for d in groups:
            if "lr" in d and d["lr"] == 0.0:
                for param in d["params"]:
                    param.requires_grad_(False)
            else:
                new_groups.append(d)
        optimizer = optimizer.run(params=groups)
    else:
        optimizer = optimizer.run(params=model.parameters())

    return optimizer
