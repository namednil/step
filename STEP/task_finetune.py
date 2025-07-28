import sys
import tempfile
from collections import deque
from typing import Optional, List, Dict, Tuple, Callable

import torch.optim.optimizer

from config_evaluator import Lazy
from logger import Logger
from STEP.utils import get_optimizer, evaluate_on, hack_t5_parallelize

from STEP.pretraining import scale_grad
from STEP.data_loading import RandomSplit


class MovingAvgDict:
    # Naive implementation
    def __init__(self, window_length):
        self.l = deque()
        self.window_length = window_length

    def get(self):
        if len(self.l) == 0:
            return dict()
        sums = dict()
        counts = dict()
        for d in self.l:
            for k, v in d.items():
                sums[k] = sums.get(k, 0) + v
                counts[k] = counts.get(k, 0) + 1
        return {k: v/counts[k] for k, v in sums.items()}

    def append(self, x):
        assert isinstance(x, dict)
        if len(self.l) == self.window_length:
            self.l.popleft()
        self.l.append(x)
        return self.get()

    def update(self, x):
        vals = self.append(x)
        d = dict(x)
        for k, v in vals.items():
            d[f"{k}_avg_{self.window_length}"] = v
        return d


def is_better_dev(current_metrics, best_metrics, metric_name) -> bool:
    assert metric_name[0] in ["+", "-"]
    best_m = best_metrics.get(metric_name[1:], None)
    if metric_name[1:] not in current_metrics:
        raise ValueError(f"Evaluation didn't produce the metric {metric_name[1:]} but instead have: {', '.join(current_metrics.keys())}")
    current_m = current_metrics[metric_name[1:]]
    is_better = best_m is None or (current_m > best_m and metric_name[0]) == "+" or (current_m < best_m and metric_name[0] == "-")
    if is_better:
        print(f"Better checkpoint ({current_m} vs previously best {best_m})", file=sys.stderr)
    return is_better


def finetune_model(model,
                  tokenizer,
                  train_data_loader,
                  validation_data_loader,
                  test_data_loader = None,
                  dataset_splitter: Optional[RandomSplit] = None,
                  num_epochs: int = 110,
                  moving_avg_steps: int = 10,
                  device: str = "cuda:0",
                  optimizer: Lazy[torch.optim.Optimizer] = None,
                  logger: Optional[Logger] = None,
                  grad_scale: Optional[float] = None,
                  optimizer_groups: Optional[List[Tuple[str, Dict]]] = None,
                  hack_parallelize: bool = False,
                  num_accumulation_steps: int = 1,
                  eval_only_last_epochs: bool = False,
                  use_deterministic_algorithms: bool = False,
                  custom_eval_on: Optional[Callable] = None,
                  eval_metric: Optional[str] = None,
                  eps = 0.0,
                batch_logging: bool = False
                  ):
    optimizer = get_optimizer(model, optimizer, optimizer_groups)

    torch.use_deterministic_algorithms(use_deterministic_algorithms)

    if hack_parallelize:
        model = hack_t5_parallelize(model)
    elif device is None:
      device = model.device # get device from model
    else:
        if hasattr(model, "model_parallel"):
            model.deparallelize()

        model = model.to(device)

    if dataset_splitter is not None:
        if train_data_loader is not None or validation_data_loader is not None:
            raise ValueError("dataset_splitter given, so train_data_loader and validation_data_loader must be None")
        train_data_loader = dataset_splitter.get_train_loader()
        validation_data_loader = dataset_splitter.get_rest_loader()

    avg = MovingAvgDict(moving_avg_steps)

    if logger is None:
        logger = Logger()


    best_metrics = dict()
    best_model_checkpoint  = None
    if eval_metric is not None:
        best_model_checkpoint = tempfile.TemporaryFile()

    batch_count = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        virtual_batch_loss = 0
        model.train()
        for batch_id, batch in enumerate(logger.progress_bar(train_data_loader)):
            batch = {k: v.to(device) for k,v in batch.items()}
            r = model(**batch)
            virtual_batch_loss += r.loss.detach().cpu().float().numpy()
            epoch_loss += r.loss.detach().cpu().float().numpy()
            r.loss.backward()
            batch_count += 1
            if batch_count % num_accumulation_steps == 0:
                scale_grad(model, grad_scale, eps=eps)
                optimizer.step()
                optimizer.zero_grad()
                if batch_logging:
                    logger.log_metrics("finetune_train", {"batch_loss": virtual_batch_loss})
                virtual_batch_loss = 0
        logger.log_metrics("finetune_train", {"loss": epoch_loss})
        print("loss", epoch_loss)

        # Evaluate
        logger.set_output_epoch_info({"epoch": epoch})
        if validation_data_loader is not None and ((not eval_only_last_epochs) or epoch >= num_epochs - moving_avg_steps):
            if custom_eval_on:
                r: dict[str, float] = custom_eval_on(model, tokenizer, logger.progress_bar(validation_data_loader), logger=logger)
                r = avg.update(r)
                logger.log_metrics("finetune_dev", r)
                print(r)
            else:
                acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(validation_data_loader), logger=logger)
                r = avg.update({"acc": acc, "edit_dist": edit, "per": per})
                logger.log_metrics("finetune_dev", r)
                print(r)

            if eval_metric is not None and is_better_dev(r, best_metrics, eval_metric):
                best_metrics = dict(r)
                # Save model
                best_model_checkpoint.seek(0)
                torch.save(model, best_model_checkpoint)
                best_model_checkpoint.seek(0)

    if eval_metric is not None:
        print("Loading best model...", file=sys.stderr)
        model = torch.load(best_model_checkpoint)

    if test_data_loader is not None:
        logger.set_output_epoch_info({"test_data": True})
        if custom_eval_on:
            r: dict[str, float] = custom_eval_on(model, tokenizer, logger.progress_bar(test_data_loader),
                                                 logger=logger)
            r = avg.update(r)
            logger.log_metrics("finetune_test", r)
            print(r)
        else:
            acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(test_data_loader), logger=logger)
            r = avg.update({"acc": acc, "edit_dist": edit, "per": per})
            logger.log_metrics("finetune_test", r)
            print(r)

    if hack_parallelize:
        model.deparallelize()

    return model


def eval_model(model,
                  tokenizer,
                  data_loader,
                  device: str = "cuda:0",
                  logger: Optional[Logger] = None,
                  hack_parallelize: bool = False,
                  use_deterministic_algorithms: bool = False,
                  custom_eval_on: Optional[Callable] = None,
                  ):

    torch.use_deterministic_algorithms(use_deterministic_algorithms)

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

    logger.set_output_epoch_info({"test_data": True})

    if custom_eval_on:
        r: dict[str, float] = custom_eval_on(model, tokenizer, logger.progress_bar(data_loader),
                                             logger=logger)
        logger.log_metrics("finetune_test", r)
        print(r)
    else:
        acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(data_loader), logger=logger)
        r = {"acc": acc, "edit_dist": edit, "per": per}
        logger.log_metrics("finetune_test", r)
        print(r)

    if hack_parallelize:
        model.deparallelize()

    return model


