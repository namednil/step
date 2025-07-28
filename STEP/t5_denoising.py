#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from datasets import load_dataset

from transformers import (
    PreTrainedTokenizerBase, DataCollatorForLanguageModeling,
)


def load_c4_chunk(chunk_id, tokenizer, input_length, batch_size,
                  noise_density: float = 0.15, mean_noise_span_length: int = 3):

    if isinstance(chunk_id, str):
        dataset = load_dataset("allenai/c4", data_files=f"en/c4-train.{chunk_id}-of-01024.json.gz",
                               revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
                               streaming=False)
    else:
        dataset = load_dataset("allenai/c4", data_files=[f"en/c4-train.{cid}-of-01024.json.gz" for cid in chunk_id],
                               revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
                               streaming=False)

    def tok(example):
        r = tokenizer(example["text"], truncation=True, max_length=input_length)
        return r

    dataset = dataset.map(tok, batched=True)

    # There is enough text, simply filter out things that are too short:
    dataset = dataset.filter(lambda example: len(example["input_ids"]) == input_length)
    dataset.set_format("numpy")

    dataset = dataset["train"].remove_columns(["text", "url"])

    collator = DataCollatorForT5MLM(tokenizer, noise_density, mean_noise_span_length, pad_token_id=0)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1,
                                       collate_fn=collator)



def load_c4_chunk_for_lm(chunk_id, tokenizer, input_length, batch_size, streaming=False):

    if isinstance(chunk_id, str):
        dataset = load_dataset("allenai/c4", data_files=f"en/c4-train.{chunk_id}-of-01024.json.gz",
                               revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
                               streaming=streaming)
    else:
        dataset = load_dataset("allenai/c4", data_files=[f"en/c4-train.{cid}-of-01024.json.gz" for cid in chunk_id],
                               revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
                               streaming=streaming)

    def tok(example):
        r = tokenizer(example["text"], truncation=True, max_length=input_length)
        return r

    dataset = dataset.map(tok, batched=True)
    dataset = dataset.filter(lambda example: len(example["input_ids"]) == input_length)
    dataset.with_format("torch")

    dataset = dataset["train"].remove_columns(["text", "url", "timestamp"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1,
                                       collate_fn=collator)


from datasets import Dataset
import gzip

def plain_text_mlm_gzip(path, tokenizer, input_length, batch_size, noise_density: float = 0.15, mean_noise_span_length: int = 3):

        d = {"input": []}
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.rstrip("\n")
                d["input"].append(line)

        dataset = Dataset.from_dict(d)
        def mapper(examples):
            return tokenizer(examples["input"], truncation=True, padding='max_length', max_length=input_length)

        dataset = dataset.map(mapper, batched=True)
        dataset = dataset.filter(lambda example: sum(example["attention_mask"]) == input_length)
        dataset.with_format("numpy")

        dataset = dataset.remove_columns(["input"])

        collator = DataCollatorForT5MLM(tokenizer, noise_density, mean_noise_span_length, pad_token_id=0)

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1,
                                           collate_fn=collator)


def loop_iterator(iterator):
    if iterator is None:
        return None

    while True:
        for x in iterator:
            yield x


def _compose_loaders(loaders):
    for l in loaders:
        for x in l:
            yield x

def compose_loaders(loaders):
    return iter(_compose_loaders(loaders))

def combine_data_loaders(loaders, probs):
    my_loaders = [iter(loop_iterator(l)) for l in loaders]
    assert len(probs) == len(my_loaders)
    indices = list(range(len(probs)))
    while True:
        index = np.random.choice(indices, p=probs)
        yield next(my_loaders[index])



@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.

    All inputs have to be the same length before padding. This was adapted from:
    https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py


    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    pad_token_id: int

    def __call__(self, batch_list) -> dict[str, torch.Tensor]:
        batch: dict[str, np.array] = dict()
        batch["input_ids"] = np.stack([elem["input_ids"] for elem in batch_list])
        batch["attention_mask"] = np.stack([elem["attention_mask"] for elem in batch_list])

        # batch["input_ids"] = np.concatenate([elem["input_ids"] for elem in batch_list])
        # batch["attention_mask"] = np.concatenate([elem["attention_mask"] for elem in batch_list])

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        real_input_lengths = batch["attention_mask"].sum(axis=1)
        assert np.all(real_input_lengths == real_input_lengths[0]), "All inputs must have the same length (without padding)"

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = torch.from_numpy(self.filter_input_ids(input_ids, input_ids_sentinel))
        batch["labels"] = torch.from_numpy(self.filter_input_ids(input_ids, labels_sentinel))

        batch["attention_mask"] = torch.from_numpy(batch["attention_mask"][:, :batch["input_ids"].shape[1]])

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


if __name__ == "__main__":
    import transformers
    import tqdm
    tok = transformers.AutoTokenizer.from_pretrained("t5-base")
    chunk_loader = load_c4_chunk(["00002", "00003", "00004"], tok, 80, 50)
    for x in tqdm.tqdm(chunk_loader):
        # print(x)
        pass
        # print(x)
