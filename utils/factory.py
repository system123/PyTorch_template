from .helpers import *
from torch.utils.data import DataLoader

# Deprecated
# def create_data_loader(config):
#     data_loaders = get_modules(['./data_loaders'])
#     name = config.type
#     assert name in data_loaders.keys(), "Could not find a {} data loader in the data_loaders directory".format(name)
#     return data_loaders[name](config)

def create_dataset(config):
    data_sets = get_modules(['./datasets'])
    name = config.type
    assert name in data_sets.keys(), "Could not find a {} dataset type".format(name)
    return data_sets[name](config)

def build_model(config):
    models = get_modules(['./models'])
    name = config.type
    assert name in models.keys(), "Could not find model {}".format(name)
    del config.type
    return models[name](**config.toDict())

def get_data_loader(dset, config):
    return DataLoader(dset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.workers)
