import os.path
import random
import re
from typing import Optional, List, Dict, Tuple, Union, Callable

import torch.optim.optimizer
import transformers

import numpy as np

from config_evaluator import Lazy
from logger import Logger
from STEP.utils import get_optimizer, evaluate_on, hack_t5_parallelize

import tqdm

def scale_grad(model, scaling, eps=0.0):
    if scaling is None:
        return
    length = 0.0
    for p in model.parameters():
        if p.grad is not None:
            length += torch.square(p.grad).sum()
    length = torch.sqrt(length) + eps
    if length > scaling:
        length = length / scaling
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= length


def _read_lines_and_sample(fname, num:int, outf):
    with open(fname) as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:num]
    with open(outf, "w") as f:
        f.writelines(lines)

def set_seeds(python_seed, pytorch_seed, numpy_seed):
    random.seed(python_seed)
    torch.manual_seed(pytorch_seed)
    np.random.seed(numpy_seed)


def loop_iterator(iterator):
    if iterator is None:
        return None

    while True:
        for x in iterator:
            yield x

def pretrain(model,
             tokenizer,
             train_data_loader,
             easy_validation_data_loader,
             validation_data_loader,
             save_dir: str,
             pretrain_data_loader=None,
             p_pretrain: float = None,
             train_data_path: str = None, # for taking a sample to put into save_dir
             test_data_loader = None,
             num_epochs: int = 10,
             python_seed: int = 363166917,
             pytorch_seed: int = 682506085,
             numpy_seed: int = 161354504,
             device: str = "cuda:0",
             optimizer: Lazy[torch.optim.Optimizer] = None,
             lr_scheduler: Lazy[torch.optim.lr_scheduler.LRScheduler] = None,
             logger: Optional[Logger] = None,
             grad_scale: Optional[float] = None,
             optimizer_groups: Optional[List[Tuple[str, Dict]]] = None,
             hack_parallelize: bool = False,
             num_accumulation_steps: Union[int, Callable[[int], int]] = 1,
             sample_size_for_save_dir: int = 400,
             freq_save: int = None,
             train_mode: bool = True,
             save_checkpoints: bool = False,
             pass_num_training_steps_to_scheduler: bool = True,
             distill_model: Optional[transformers.AutoModelForSeq2SeqLM] = None,
             eps: float = 0.0,
             use_aux_optimizer: bool=False):

    set_seeds(python_seed, pytorch_seed, numpy_seed)

    optimizer_obj = get_optimizer(model, optimizer, optimizer_groups)

    if use_aux_optimizer and pretrain_data_loader is not None:
        aux_optimizer = get_optimizer(model, optimizer, optimizer_groups)
    else:
        aux_optimizer = None

    optimizer = optimizer_obj

    if train_data_path is not None:
        with open(train_data_path, "r"):
            pass


    if hack_parallelize:
        model = hack_t5_parallelize(model)
    elif device is None:
      device = model.device # get device from model
    else:
        if hasattr(model, "model_parallel"):
            model.deparallelize()

        model = model.to(device)

    if logger is None:
        logger = Logger()

    model.train(train_mode)

    if lr_scheduler is not None:
        if pass_num_training_steps_to_scheduler:
            assert isinstance(num_accumulation_steps, int)
            lr_scheduler = lr_scheduler.run(optimizer=optimizer, num_training_steps=num_epochs * len(
                train_data_loader) // num_accumulation_steps)
        else:
            lr_scheduler = lr_scheduler.run(optimizer=optimizer)

    if distill_model is not None:
        distill_model = distill_model.to(device)
        distill_model.eval()


    loss = 0
    batch_count = 0
    orig_pretrain_loss = None
    pretrain_iterator = iter(loop_iterator(pretrain_data_loader))
    for epoch in range(num_epochs):
        acc_steps = num_accumulation_steps if isinstance(num_accumulation_steps, int) else num_accumulation_steps(epoch)
        for batch_id, batch in enumerate(logger.progress_bar(train_data_loader)):
            batch = {k: v.to(device) for k,v in batch.items()}
            r = model(**batch)
            loss += r.loss.detach().cpu().numpy()
            r.loss.backward()
            batch_count += 1

            if aux_optimizer is None:
                if distill_model is not None:
                    raise NotImplementedError()
                # old-style
                if p_pretrain is not None and pretrain_iterator is not None and random.random() < p_pretrain:
                    pretrain_batch = next(pretrain_iterator)
                    pretrain_batch = {k: v.to(device) for k, v in pretrain_batch.items()}
                    r = model(**pretrain_batch)
                    orig_pretrain_loss = r.loss.detach().cpu().numpy() if orig_pretrain_loss is None else \
                                         0.95 * orig_pretrain_loss + (1-0.95) * r.loss.detach().cpu().numpy() # exponential moving average
                    r.loss.backward()

                if batch_count % acc_steps == 0:
                    scale_grad(model, grad_scale, eps=eps)
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.log_metrics("pretrain", {"loss": loss / acc_steps, "orig_pretrain_loss": orig_pretrain_loss})
                    loss = 0
                    orig_pretrain_loss = 0
                    if lr_scheduler is not None:
                        lr_scheduler.step()
            else:
                if batch_count % acc_steps == 0:
                    scale_grad(model, grad_scale, eps=eps)
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.log_metrics("pretrain", {"loss": loss / acc_steps, "orig_pretrain_loss": orig_pretrain_loss})
                    loss = 0
                    orig_pretrain_loss = 0
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    #we have now taken gradient into account from main task, potentially sample a batch for the aux task now:
                    if p_pretrain is not None and pretrain_iterator is not None and random.random() < p_pretrain:
                        pretrain_batch = next(pretrain_iterator)
                        pretrain_batch = {k: v.to(device) for k, v in pretrain_batch.items()}
                        if distill_model is not None:
                            with torch.no_grad():
                                probs = torch.softmax(distill_model(**pretrain_batch).logits, dim=-1)
                            r = model(**pretrain_batch)
                            r.loss = -(probs * torch.log_softmax(r.logits, dim=-1)).sum() / pretrain_batch["labels"].numel()
                        else:
                            r = model(**pretrain_batch)

                        orig_pretrain_loss = r.loss.detach().cpu().numpy()
                        r.loss.backward()

                        scale_grad(model, grad_scale, eps=eps)
                        aux_optimizer.step()
                        aux_optimizer.zero_grad()

        # Easy Validation
        if easy_validation_data_loader is not None:
            acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(easy_validation_data_loader))
            logger.log_metrics("pretrain_easy_dev", {"acc": acc, "edit_dist": edit, "per": per})
            print("Easy validation", {"acc": acc, "edit_dist": edit, "per": per})

        #Normal validation
        if validation_data_loader is not None:
            acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(validation_data_loader))
            logger.log_metrics("pretrain_dev", {"acc": acc, "edit_dist": edit, "per": per})
            print("Validation", {"acc": acc, "edit_dist": edit, "per": per})

        if freq_save is not None and epoch % freq_save == 0:
            model.save_pretrained(save_dir)

        if save_checkpoints:
            model.save_pretrained(save_dir.rstrip("/")+f"-epoch-{epoch}")

        model.train(train_mode)

    if hack_parallelize:
        model.deparallelize()

    if test_data_loader is not None:
        acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(test_data_loader))
        logger.log_metrics("pretrain_test", {"acc": acc, "edit_dist": edit, "per": per})
        print("Validation", {"acc": acc, "edit_dist": edit, "per": per})

    model.save_pretrained(save_dir)

    if train_data_path is not None:
        _read_lines_and_sample(train_data_path, sample_size_for_save_dir, os.path.join(save_dir, "pretraining_sample.jsonl"))

    return model

