from dotmap import DotMap
from glob import glob
import os
import json

from utils.helpers import extract_numbers

def number_ordering(x):
    n = extract_numbers(x)
    return n[-1] if len(n) > 0 else 0

class Experiment(DotMap):
    def __init__(self, name, desc="", result_dir="./results", *args, **kwargs):
        super(Experiment, self).__init__(*args, **kwargs)

        self.name = name
        self.desc = desc
        self.result_dir = result_dir
        self.result_path = os.path.join(result_dir, name)

    def save(self):
        os.makedirs(self.result_path, exist_ok=True)
        with open(os.path.join(self.result_path, "config.json"), "w") as f:
            f.write(json.dumps(self.toDict(), indent=4, sort_keys=False))

    def load(self):
        try:
            with open(os.path.join(self.result_path, "config.json")) as f:
                data = json.load(f)
            super(Experiment, self).__init__(data)
            return self
        except:
            return None

    def exists(self):
        return os.path.exists(self.result_path)

    def get_checkpoint_path(self):
        model_path = None

        if 'checkpoint' in self and len(self.checkpoint) > 0:
            _, ext = os.path.splitext(self.checkpoint)

            if ext == ".pth":
                model_path = self.checkpoint if os.path.exists(self.checkpoint) else None
                path = os.path.join(self.result_path, self.checkpoint) if model_path is None else model_path
                model_path = path if os.path.exists(path) else None
            elif self.checkpoint == "best":
                cpts = glob(os.path.join(self.result_path, 'best_checkpoint_model*.pth'))
                cpts = sorted(cpts, key=number_ordering)
                if len(cpts) > 0:
                    model_path = cpts[-1]
            elif self.checkpoint.isdigit():
                path = os.path.join(self.result_path, 'checkpoint_model_{}.pth'.format(self.resume_from))
                model_path = path if os.path.exists(path) else None

        optim_path = model_path.replace('model','optim') if model_path else None
        return (model_path, optim_path)

    @staticmethod
    def load_from_path(path):
        with open(path) as f:
            data = json.load(f)

        return Experiment(data['name'], data['desc'], data['result_dir'], data)


    @staticmethod
    def load_by_name(name, conf_dir="./config"):
        exp = Experiment(name, result_dir=conf_dir).load()
        return(exp)
