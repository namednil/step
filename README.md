# STEP

This is the repository for our EMNLP 2024 paper on [Strengthening Structural Inductive Biases by Pre-training to Perform Syntactic Transformations](https://arxiv.org/abs/2407.04543), in which we present STEP (Syntactic Transformation Enhanced Pre-training), a new intermediate pre-training procedure.

# Simple Fine-tuning of STEP
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


# Reproducing Experiments

## Setting up Environment


```
conda create -n step python=3.10
conda activate step
# install pytorch (we used v 2.2.0 but newer versions such as 2.7.0 should work fine as well)
#e.g. via
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone git repository
git clone https://github.com/namednil/step

# Install remaining pip requirements
cd step
pip install -r requirements.txt
```

## Data Preparation

We first have to generate the pre-training data for STEP (and potentially for the baselines).

1. [Optional - output is uploaded to git repo with git lfs for your convenience] Parse a tiny portion of the T5-pretraining corpus (C4) with trankit to produce a treebank.
   - Call `python -m STEP.data_gen.parse_c4_batched` from the directory where this README is.
2. Synthetically generated syntactic transformations used to train STEP. Call `bash generate_pt_data.sh [-baseline]` - this takes a while (an hour or so) and you might see some warnings about sentences being ignored. That's expected.When provided with the flag `-baseline`, we also generate the data for intermediate pre-training of the baselines. On successful completion of the script, the `grammars` directory will contain the pre-training files:
```
grammars/
├── pt_data_simple_step_test.tsv [optional, baseline]
├── pt_data_simple_step_train.tsv [optional, baseline]
├── pt_data_step_dev.pkl.xz
├── pt_data_step_easy_dev.pkl.xz
├── pt_data_step_l2i.pkl [contains a mapping from dependency labels to vocab ids and will overwrite the existing file from repo]
├── pt_data_step_test.pkl.xz
├── pt_data_step_train.pkl.xz
├── pt_data_t5_plus_dep_parse_test.tsv [optional, baseline]
└── pt_data_t5_plus_dep_parse_train.tsv [optional, baseline]
```

## Intermediate Pre-Training

After the data has been prepared, run

```bash
python config_evaluator.py configs/STEP.jsonnet
```
You may want to adjust the `STEP.jsonnet` config, e.g. by default, it will try to log experiments to `neptune.ai`. If you prefer to log to the standard output, use `"logger": {  f: "TqdmLogger.create"  },` instead. You can also define your own logger in `logger.py`. 

If you've got more than 12 GB of GPU memory, you can reduce the number of accumulation steps and increase the batch size to speed up training.

Training baselines is analogous using their respective config files.


## Reproducing Fine-Tuning Experiments

**Note: if you've pre-trained your own version of STEP, you have to follow these steps rather than the simplified (but in principle equivalent) procedure described [above](#simple-fine-tuning-of-step). The steps presented here run the model classes as used for the paper.**

First, you'll need to unpack the training/test data in `data/finetune.zip`. To avoid accidentally contaminating LLMs pre-training data, we use a **password-protected zip archive**. The password is **STEP**.

Pick a config file, e.g. for fine-tuning and evaluating STEP on chunking data, and supply the environmental variables (`std.extVar` in the configs) and then run `python config_evaluator.py [config file]`. 

Here, we take `configs/finetune/step_chunking.jsonnet`. As before, you might want to adjust the logger to point to your neptune.ai experiment or log only locally using `"logger": {  f: "TqdmLogger.create"  },` instead. Then we can run the experiment:

```bash
export seed=2346 # random seed
export train="data/finetune/chunking/train_100_1.tsv"
export test="data/finetune/chunking/test.tsv"
# Learning rates (see also appendix of our paper)
export model_lr="1e-4"
export prefix_lr="10"
# Pre-trained model that you trained yourself.
export load_model="models/step_model"
# Initialization of prefix
export load_prefix="grammars/step/grammars/pt_data_step_dev.pkl.xz"

python config_evaluator.py configs/finetune/step_chunking.jsonnet
```

**Fine-tuning experiments with Hugging Face version of STEP**

If you want to fine-tune the STEP model we've uploaded to Hugging Face (rather than your own local model), the fine-tuning config files have to be slightly adjusted: the part that describes the model in the config has to look like this (see `configs/finetune/step_chunking_hf.jsonnet` for an example and compare with `configs/finetune/step_chunking.jsonnet`):
```
      model: {f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
                pretrained_model_name_or_path: "namednil/STEP",
                trust_remote_code: true
             },
```
 
 # Citation

 ```bibtex
@inproceedings{lindemann-etal-2024-strengthening,
    title = "Strengthening Structural Inductive Biases by Pre-training to Perform Syntactic Transformations",
    author = "Lindemann, Matthias  and
      Koller, Alexander  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.645/",
    doi = "10.18653/v1/2024.emnlp-main.645",
}
 ```
