# STEP

This is the repository for the paper [Strengthening Structural Inductive Biases by Pre-training to Perform Syntactic Transformations](https://arxiv.org/abs/2407.04543), in which we present STEP (Syntactic Transformation Enhanced Pre-training), a new intermediate pre-training procedure.

# Fine-tuning STEP
The STEP model is based on T5-Base and can be fine-tuned as follows:

```python
import transformers, torch
tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("namednil/STEP", trust_remote_code=True)
# always make sure to check the remote code on Huggingface, i.e. check out https://huggingface.co/namednil/STEP/blob/main/step_finetune.py

# Construct an optimizer that uses the STEP fine-tuning procedure as used in the paper:
optimizer = model.get_optimizer()
# or another optimizer
optimizer = model.get_optimizer(torch.optim.Adam, prefix_lr=10.0, lr=1e-4)
# ... fine-tune the model as usual
```


# Reproducing experiments
Code and data for the full reproduction of our experiments will follow soon.
