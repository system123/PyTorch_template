from .helpers import *
from torch.utils.data import DataLoader

# Deprecated
# def create_data_loader(config):
#     data_loaders = get_modules(['./data_loaders'])
#     name = config.type
#     assert name in data_loaders.keys(), "Could not find a {} data loader in the data_loaders directory".format(name)
#     return data_loaders[name](config)

def create_dataset(config):
    dataset = get_module('./datasets', config.type)
    return dataset(config)

def build_model(config):
    model = get_modules('./models', config.type)
    del config.type
    return model(**config.toDict())

def get_data_loader(dset, config):
    return DataLoader(dset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.workers)
