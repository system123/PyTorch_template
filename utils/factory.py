from .helpers import *
from torch.utils.data import DataLoader
from ignite.metrics import *
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_schedulers

# Deprecated
# def create_data_loader(config):
#     data_loaders = get_modules(['./data_loaders'])
#     name = config.type
#     assert name in data_loaders.keys(), "Could not find a {} data loader in the data_loaders directory".format(name)
#     return data_loaders[name](config)

def create_dataset(config):
    dataset = get_module('./datasets', config.type)
    return dataset(config) or None

def build_model(config):
    model = get_module('./models', config.type)
    args = copy_and_delete(config.toDict(), 'type')
    return model(**args) or None

def get_optimizer(params, config):
    optim = str_to_class('torch.optim', config.type)
    args = copy_and_delete(config.toDict(), 'type')
    return optim(params, **args) or None

def get_data_loader(dset, config):
    return DataLoader(dset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.workers)

def get_trainer(model, optimizer, loss_fn, exp_logger, config):
    trainer = get_module('./trainer', config.trainer)
    return trainer(model, optimizer, loss_fn, exp_logger, config) or None

def get_experiment_logger(config):
    logger = get_module('./logger', config.logger)
    return logger(config) or None

def get_loss(loss_fn):
    loss = get_if_implemented(F, loss_fn)
    if loss is None:
        loss = get_function('./functional.losses', loss_fn)
    return loss

def get_lr_scheduler(optimizer, config):
    name = config.type
    args = copy_and_delete(config.toDict(), 'type')

    lr_scheduler = get_if_implemented(lr_schedulers, name)
    # lr_scheduler = str_to_class('torch.optim.lr_scheduler', config.type)
    if lr_scheduler is None:
        lr_scheduler = get_module('./schedulers', name)
    
    if lr_scheduler is None:
        fcn = get_function('./schedulers.functional', name)
        assert fcn, "No functional implementation of {} was found".format(name)
        fcn_wrapper = lambda e: fcn(e, **args)
        lr_scheduler = lr_schedulers.LambdaLR(optimizer, fcn_wrapper)
    else:
        lr_scheduler = lr_scheduler(optimizer, **args)

    return lr_scheduler






