from dotmap import DotMap
import os
import json

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

    @staticmethod
    def load_from_path(path):
        with open(path) as f:
            data = json.load(f)

        return Experiment(data['name'], data['desc'], data['result_dir'], data)


    @staticmethod
    def load_by_name(name, conf_dir="./config"):
        exp = Experiment(name, result_dir=conf_dir).load()
        return(exp)
