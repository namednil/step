import gc
import lzma

import torch.cuda
from datasets import load_dataset
from tqdm import tqdm

from trankit import Pipeline
from trankit.utils.conll import CoNLL

import traceback

if __name__ == "__main__":
    gpu = True
    cache = '~/cache'
    pipeline = Pipeline(lang='english', gpu=gpu, cache_dir=cache)
    batch_size = 64
    pipeline._tokbatchsize = batch_size
    pipeline._tagbatchsize = batch_size

    chunk_id = "00000"
    with lzma.open(f"output_batched_{chunk_id}_200k.conll.xz", "wt") as f:
        dataset = load_dataset("allenai/c4", "en", data_files=f"en/c4-train.{chunk_id}-of-01024.json.gz",
                               streaming=True)
        batch = []
        for i, doc in enumerate(tqdm(dataset["train"])):
            # Let's only select documents that contain at least one question mark
            # to over-sample questions.
            if "?" not in doc["text"]:
                continue
            if i > 200_000:
            #if i > 80_000:
                break
            try:
                tokenized = pipeline.tokenize(doc["text"])
                tokenized = [[w["text"] for w in s["tokens"]] for s in tokenized["sentences"]] # only keep words
                batch.extend(tokenized)
            except torch.cuda.OutOfMemoryError as rerr:
                traceback.clear_frames(rerr.__traceback__) # necessary to free this so there are no pointers to big tensors
                gc.collect()
                print("OOM error in tokenization, skipping document.")
                
                                                                         
            if len(batch) >= batch_size:
                try:
                    parsed = pipeline(batch)
                except torch.cuda.OutOfMemoryError as rerr:
                    traceback.clear_frames(rerr.__traceback__) # necessary to free this so there are no pointers to big tensors.
                    gc.collect()
                    print("OOM error, trying to recover")
                    pipeline = Pipeline(lang='english', gpu=gpu, cache_dir=cache)
                    parsed = pipeline(batch)
                    print("Managed to recover.")
                    pipeline._tokbatchsize = batch_size
                    pipeline._tagbatchsize = batch_size

                o = [s["tokens"] for s in parsed["sentences"]]
                conll_string = CoNLL.dict2conllstring(o)
                f.write(conll_string)
                batch = []


