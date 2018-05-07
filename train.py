import os
import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from experiment import Experiment
from utils.factory import *
from utils.lr_scheduler import *

logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

def validate_config(config):
    return True

def main(config):
    assert validate_config(config), "ERROR: Config file is invalid. Please see log for details."

    logger.info("INFO: {}".format(config.toDict()))

    # Set the random number generator seed for torch, as we use their dataloaders this will ensure shuffle is constant
    # Remeber to seed custom datasets etc with the same seed
    if config.seed > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.manual_seed(config.seed)

    logger.info("INFO: Creating datasets and dataloaders...")
    # Create the training dataset
    dset_train = create_dataset(config.datasets.train)
    # Esnure we have a full config for validation, this means we don't need t specify everything in the config file
    # only the differences
    config_val = config.datasets.train
    config_val.update(config.datasets.validation)

    # If the validation config has a parameter called split then we ask the training dset for the validation dataset
    # it should be noted that you shouldn't shuffle the dataset in the init of the train dataset if this is the case
    # as only on get_validation_split will we know how to split the data. Unless shuffling is deterministic.
    if 'validation' in config.datasets:
        if 'split' in config.datasets.validation:
            dset_val = dset_train.get_validation_split(config_val)
        else:
            dset_val = create_dataset(config_val)
    else:
        logger.warning("WARNING: No validation dataset was specified")
        dset_val = None
        loader_val = None

    loader_train = get_data_loader(dset_train, config.datasets.train)

    if dset_val is not None:
        loader_val = get_data_loader(dset_val, config_val)

    Create_lr_scheduler_funciton
    Create the optimizer
    Create the model
    Create the loss - if more than one loss then we need to add all the losses and everything to the training function
    Create the metrics
    Create the trainer core - pass it the optimizer, model,
    Register callbacks

    # Save the config for this experiment to the results directory, once we know the params are good
    # config.save()

    pass


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    vis = visdom.Visdom()
    if not vis.check_connection():
        raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
        model = model.to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': CategoricalAccuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)

    train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
    val_accuracy_window = create_plot_window(vis, '#Epochs', 'Accuracy', 'Validation Accuracy')
    val_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            vis.line(X=np.array([engine.state.iteration]),
                     Y=np.array([engine.state.output]),
                     update='append', win=train_loss_window)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]), win=val_accuracy_window, update='append')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_nll]), win=val_loss_window, update='append')

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='config file path (default: None)')
    args = parser.parse_args()

    config = Experiment.load_from_path(args.config)

    if config.resume:
        logger.warning("WARNING: --config specifies resuming, overriding config with exising experiment.")
        resume_config = Experiment(config.name, desc=config.desc, result_dir=config.result_dir).load()
        assert resume_config is not None, "No experiment {} exists, cannot resume training".format(config.name)
        config = resume_config
    elif config is not None:
        assert not config.exists(), "Results directory {} already exists! Please specify a new experiment name or the remove old files.".format(config.result_path)

    assert config is not None

    main(config)
