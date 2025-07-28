import gzip
import json
import os
import sys
from typing import Dict

import neptune
import tqdm


class Logger:

    def ___init__(self, **kwargs):
        self.info = dict()

    @staticmethod
    def create(**kwargs):
        raise NotImplementedError()
    def progress_bar(self, data_loader):
        return data_loader
    def log_config(self, config: Dict):
        pass

    def log_jsonnet(self, text):
        pass
    def log_metrics(self, name: str, metrics: Dict[str, float]):
        pass

    def stop_logging(self):
        pass

    def set_output_epoch_info(self, info: dict):
        self.info = info

    def log_output(self, outputs: list[dict]):
        """
        Log a batch of model outputs (e.g. on validation or test data) to somewhere.
        :param output:
        :param tags:
        :return:
        """
        pass


class TqdmLogger(Logger):

    @staticmethod
    def create(**kwargs):
        return TqdmLogger()
    def progress_bar(self, data_loader):
        mininterval = 0.1
        if "TQDM_INTERVAL" in os.environ:
            mininterval = float(os.environ["TQDM_INTERVAL"])
        self.pb = tqdm.tqdm(data_loader, mininterval=mininterval)
        return self.pb
    def log_config(self, config: Dict):
        pass
        # print("CONFIG")
        # print(config)

    def log_metrics(self, name:str, metrics: Dict[str, float]):
        self.pb.set_description(name + ": " + ", ".join(f"{k}={metrics[k]}" for k in sorted(metrics)))



from neptune.utils import stringify_unsupported
class NeptuneLogger(Logger):
    def __init__(self, log_outputs: bool = False, **kwargs):
        self.run = neptune.init_run(**kwargs)

        self.log_file = None
        if log_outputs:
            os.makedirs("logged_outputs", exist_ok=True)
            self.log_file = gzip.open(os.path.join("logged_outputs", self.run.get_url().split("/")[-1]) + ".json.gz", "wt")

    def log_config(self, config: Dict):
        # Replace the list in steps by a dictionary because neptune can't handle lists... :(
        d = {f"step_{i+1}": s for i,s in enumerate(config["steps"])}
        config = dict(config)
        config["steps"] = d
        self.run["config"] = stringify_unsupported(config)
        self.run["config_text"] = json.dumps(config, indent=4)
        self.run["config/argv"] = stringify_unsupported(sys.argv)

    def log_jsonnet(self, text):
        self.run["jsonnet"] = text

    def stop_logging(self):
        self.run.stop()
    def log_metrics(self, name: str, metrics: Dict[str, float]):
        for k,v in metrics.items():
            self.run[name + "_" + k].append(v)
    @staticmethod
    def create(**kwargs):
        return NeptuneLogger(**kwargs)

    def log_output(self, outputs: list[dict]):
        if self.log_file is not None:
            for o in outputs:
                self.log_file.write(json.dumps(o | self.info))
                self.log_file.write("\n")
            self.log_file.flush()


