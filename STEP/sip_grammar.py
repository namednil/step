import os
from abc import ABC
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch.nn
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizerFast, AutoModelForSeq2SeqLM

from config_evaluator import Lazy

### Stricter UD version
class UDPretrainingModel(torch.nn.Module):

    def __init__(self, model: PreTrainedModel,
                 num_nts: int,
                 num_functions: int,
                 freeze_embeddings: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

        self.trafo_embedding_dim = self.model.get_input_embeddings().embedding_dim
        self.ud_embeddings = torch.nn.Embedding(num_nts, self.trafo_embedding_dim)
        self.function_embeddings = torch.nn.Embedding(num_functions, self.trafo_embedding_dim)

        if freeze_embeddings:
            self.model.get_input_embeddings().requires_grad_(False)
            self.model.get_output_embeddings().requires_grad_(False)

    @property
    def device(self):
        return self.model.device

    def prepare_grammar_rep(self, kwargs):
        ud_labels = kwargs.pop("ud_labels") #shape (batch, len labels)
        function_ids = kwargs.pop("function_ids") #shape (batch, len functions = len labels)

        fun_embeddings = self.function_embeddings(function_ids)
        ud_embeddings = self.ud_embeddings(ud_labels)

        full_rep = fun_embeddings + ud_embeddings # shape (batch, transition, trafo embedding dim)

        return full_rep, kwargs

    def prepare_input(self, kwargs):
        embedded_inputs = self.model.get_input_embeddings()(kwargs["input_ids"])
        batch_size = embedded_inputs.shape[0]

        grammar_rep, kwargs = self.prepare_grammar_rep(kwargs) #shape(

        embedded_inputs = torch.cat([grammar_rep, embedded_inputs], dim=1)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "task_ids" in kwargs:
            del kwargs["task_ids"]

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, grammar_rep.shape[1]), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        if "ud_labels" not in kwargs:
            return self.model(**kwargs)
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        if "ud_labels" not in kwargs:
            return self.model.generate(**kwargs)

        return self.model.generate(**self.prepare_input(kwargs))

    def save_pretrained(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self, os.path.join(path, "sip_grammar.pt"))

    @staticmethod
    def from_pretrained(path, unfreeze_embeddings: bool = False, **kwargs):
        m = torch.load(os.path.join(path, "sip_grammar.pt"), **kwargs)
        if unfreeze_embeddings:
            m.model.get_input_embeddings().requires_grad_(True)
            m.model.get_output_embeddings().requires_grad_(True)
        return m
###

class StructuredPrefixEmbeddingModelForCFG(torch.nn.Module):

    def __init__(self, model: transformers.AutoModelForSeq2SeqLM,
                 prefix_length: int,
                 ignore_mismatched_sizes: bool = False,
                 freeze_embeddings: bool = False,
                 adapter_str: str = None,
                 use_ia3_adapter = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model

        if freeze_embeddings:
            self.model.get_input_embeddings().requires_grad_(False)
            self.model.get_output_embeddings().requires_grad_(False)

        self.adapter_str = adapter_str

        if adapter_str:
            self.model.add_adapter("task_adapter", adapter_str, set_active=True)

        self.use_ia3_adapter = use_ia3_adapter
        if use_ia3_adapter:
            from peft import IA3Config, TaskType, get_peft_model
            peft_config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False)
            self.model = get_peft_model(model, peft_config)

        self.prefix_length = prefix_length

        self.prefix_embedding = torch.nn.Parameter(torch.empty(1, self.prefix_length, self.model.get_input_embeddings().embedding_dim))
        torch.nn.init.normal_(self.prefix_embedding)

    @staticmethod
    def from_sip_pretrained(path: str, prefix_length: int, map_location=None,
                            data_loader=None, **kwargs):
        model = torch.load(os.path.join(path, "sip_grammar.pt"), map_location=map_location)

        if not data_loader:
            return StructuredPrefixEmbeddingModelForCFG(model.model, prefix_length, **kwargs)

        batch = {k: v.to(model.device) for k, v in next(iter(data_loader)).items()}

        activations, _ = model.prepare_grammar_rep(batch)
        # average across batch and trim to prefix length.
        init = activations.mean(dim=0).unsqueeze(0)[:, :prefix_length, :]

        # init = init.to(model.model.device)

        m = StructuredPrefixEmbeddingModelForCFG(model.model, prefix_length, **kwargs)
        m.prefix_embedding = torch.nn.Parameter(init.detach(), requires_grad=True)

        return m
        
    def save_pretrained(self, dirname):
        self.model.save_pretrained(dirname)
        torch.save(self.prefix_embedding, os.path.join(dirname, "prefix.pt"))

    @staticmethod
    def from_pretrained(dirname):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(dirname)
        prefix = torch.load(os.path.join(dirname, "prefix.pt"))
        m = StructuredPrefixEmbeddingModelForCFG(model, prefix.shape[1])
        m.prefix_embedding = prefix
        return m

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def save_pretrained(self, dirname):
        self.model.save_pretrained(dirname)
        torch.save(self.prefix_embedding, os.path.join(dirname, "prefix.pt"))

    @staticmethod
    def from_pretrained(dirname, **kwargs):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(dirname, **kwargs)
        prefix = torch.load(os.path.join(dirname, "prefix.pt"))
        m = StructuredPrefixEmbeddingModelForCFG(model, prefix.shape[1])
        m.prefix_embedding = prefix
        return m

    @property
    def device(self):
        return self.model.device

    def prepare_input(self, kwargs):
        """
        Prepends the prefix to the given input.
        :param kwargs:
        :return:
        """
        input_ids = kwargs["input_ids"]

        embedded_inputs = self.model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]

        prefix = torch.repeat_interleave(self.prefix_embedding, batch_size, 0) #shape (batch, prefix length, embed dim)

        kwargs = dict(kwargs)

        embedded_inputs = torch.cat([prefix, embedded_inputs], dim=1)  # shape (batch, prefix + seq length, embed dim)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, self.prefix_length), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        return self.model.generate(**self.prepare_input(kwargs))

