from .helpers import *
from torch.utils.data import DataLoader
from ignite.metrics import *
import torch.nn.functional as F

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
    model = get_module('./models', config.type)
    del config.type
    return model(**config.toDict())

def get_optimizer(params, config):
    model = str_to_class('torch.optim', config.type)
    del config.type
    return model(params, **config.toDict())

def get_data_loader(dset, config):
    return DataLoader(dset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.workers)

def get_trainer(model, optimizer, loss_fn, config):
    trainer = get_module('./trainer', config.trainer)
    return trainer(model, optimizer, loss_fn, config)

def get_experiment_logger(config):
    logger = get_module('./logger', config.logger)
    return logger(config)

def get_loss(loss_fn):
    loss = check_if_implemented(F, loss_fn)
    if loss is None:
        loss = get_function('./functional.losses', loss_fn)
    return loss

def get_lr_scheduler(optimizer, config):
    lr_scheduler = str_to_class('torch.optim.lr_scheduler', config.type)
    if lr_scheduler is None:
        lr_scheduler = get_module('./lr_scheduler', config.type)
    del config.type
    return lr_scheduler(optimizer, **config.toDict())
