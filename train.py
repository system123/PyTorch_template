import os
import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from ignite.engine.engine import Engine, State, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite._utils import convert_tensor

from utils import Experiment
from utils.factory import *

logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

def validate_config(config):
    assert config.device in ["cpu", "cuda"], "Invalid compute device was specified. Only 'cpu' and 'cuda' are supported."
    return True

def main(config):
    assert validate_config(config), "ERROR: Config file is invalid. Please see log for details."

    logger.info("INFO: {}".format(config.toDict()))

    # Set the random number generator seed for torch, as we use their dataloaders this will ensure shuffle is constant
    # Remeber to seed custom datasets etc with the same seed
    if config.seed > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.manual_seed(config.seed)

    if config.device == "cpu" and torch.cuda.is_available():
        logger.warning("WARNING: Not using the GPU")

    logger.info("INFO: Creating datasets and dataloaders...")
    # Create the training dataset
    dset_train = create_dataset(config.datasets.train)
    # Esnure we have a full config for validation, this means we don't need t specify everything in the config file
    # only the differences
    config_val = config.datasets.train.copy()
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

    logger.info("INFO: Building the {} model".format(config.model.type))
    model = build_model(config.model)

    m_cp_path, o_cp_path = config.get_checkpoint_path()

    if config.resume_from >= 0:
        assert m_cp_path, "Could not find a checkpoint for this model, check your config and try again"
        model.load_state_dict(torch.load( m_cp_path ))

    model = model.to(config.device)

    logger.info("INFO: Using {} Optimization".format(config.optimizer.type))
    optimizer = get_optimizer(model.parameters(), config.optimizer)

    if config.resume_from >= 0:
        assert o_cp_path, "Could not find a checkpoint for the optimizer, please check your results folder"
        optimizer.load_state_dict(torch.load( o_cp_path ))

    loss_fn = get_loss(config.loss)
    assert loss_fn, "Loss function {} could not be found, please check your config".format(config.loss)

    scheduler = None
    if 'scheduler' in config:
        logger.info("INFO: Setting up LR scheduler {}".format(config.scheduler.type))
        scheduler = get_lr_scheduler(optimizer, config.scheduler)
        assert scheduler, "Learning Rate scheduler function {} could not be found, please check your config".format(config.scheduler.type)

    if 'logger' in config:
        logger.info("INFO: Initialising the experiment logger")
        exp_logger = get_experiment_logger(config)

    logger.info("INFO: Creating training manager and configuring callbacks")
    trainer = get_trainer(model, optimizer, loss_fn, exp_logger, config)

    trainer_engine = Engine(trainer.train)
    evaluator_engine = Engine(trainer.evaluate)

    trainer.attach("train_loader", loader_train)
    trainer.attach("validation_loader", loader_val)
    trainer.attach("evaluation_engine", evaluator_engine)
    trainer.attach("train_engine", trainer_engine)

    if 'metrics' in config:
        for name, metric in config.metrics.items():
            metric = get_metric(metric)
            if metric is not None:
                metric.attach(evaluator_engine, name)
            else:
                logger.warning("WARNING: Metric {} could not be created".format(name))

    # Register default callbacks
    if exp_logger is not None:
        trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, exp_logger.log_iteration, phase="train", model=model)
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, exp_logger.log_epoch, phase="train", model=model)
        evaluator_engine.add_event_handler(Events.ITERATION_COMPLETED, exp_logger.log_iteration, phase="evaluate", model=model)
        evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, exp_logger.log_epoch, phase="evaluate", model=model)

    if loader_val is not None:
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator_engine.run(loader_val))

    if scheduler is not None:
        if config.scheduler.scheme == "batch":
            scheduler_event = Events.ITERATION_COMPLETED
        elif config.scheduler.scheme == "epoch":
            scheduler_event = Events.EPOCH_COMPLETED
        else:
            logger.error("ERROR: Invalid scheduler scheme, must be either epoch or batch")
            return 0

        trainer_engine.add_event_handler(scheduler_event, lambda engine: scheduler.step())

    if config.monitor.early_stopping:
        logger.info("INFO: Enabling early stopping, monitoring {}".format(config.monitor.score))
        score_fn = lambda e: config.monitor.scale * e.state.metrics[config.monitor.score]
        es_handler = EarlyStopping(patience=config.monitor.patience, score_function=score_fn, trainer=trainer_engine)
        evaluator_engine.add_event_handler(Events.COMPLETED, es_handler)

    if config.save_freq > 0:
        ch_path = config.result_path
        ch_handler = ModelCheckpoint(config.result_path, 'checkpoint', save_interval=config.save_freq, n_saved=4, require_empty=False, save_as_state_dict=True)
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, ch_handler, {'model': model})

    # Register custom callbacks with the engines
    if check_if_implemented(trainer, "on_iteration_start"):
        trainer_engine.add_event_handler(Events.ITERATION_STARTED, trainer.on_iteration_start, phase="train")
        evaluator_engine.add_event_handler(Events.ITERATION_STARTED, trainer.on_iteration_start, phase="evaluate")
    if check_if_implemented(trainer, "on_iteration_end"):
        trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, trainer.on_iteration_end, phase="train")
        evaluator_engine.add_event_handler(Events.ITERATION_COMPLETED, trainer.on_iteration_end, phase="evaluate")
    if check_if_implemented(trainer, "on_epoch_start"):
        trainer_engine.add_event_handler(Events.EPOCH_STARTED, trainer.on_epoch_start, phase="train")
        evaluator_engine.add_event_handler(Events.EPOCH_STARTED, trainer.on_epoch_start, phase="evaluate")
    if check_if_implemented(trainer, "on_epoch_end"):
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, trainer.on_epoch_end, phase="train")
        evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, trainer.on_epoch_end, phase="evaluate")

    # Save the config for this experiment to the results directory, once we know the params are good
    config.save()

    logger.info("INFO: Starting training...")
    trainer_engine.run(loader_train, max_epochs=config.epochs)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='config file path (default: None)')
    args = parser.parse_args()

    config = Experiment.load_from_path(args.config)

    assert config, "Config could not be loaded."

    # Else load the saved config from the results dir or throw an error if one doesn't exist
    if config.resume_from >= 0:
        logger.warning("WARNING: --config specifies resuming, overriding config with exising experiment.")
        resume_config = Experiment(config.name, desc=config.desc, result_dir=config.result_dir).load()
        assert resume_config is not None, "No experiment {} exists, cannot resume training".format(config.name)
        config = resume_config
        assert config, "Config could not be loaded for resume"
    # If we have resume_from in the config but have it < 0 to start a fresh training run then throw an error if the directory already exists
    elif config.overwrite is False:
        assert not config.exists(), "Results directory {} already exists! Please specify a new experiment name or the remove old files.".format(config.result_path)
    else:
        empty_folder(config.result_path)

    main(config)
