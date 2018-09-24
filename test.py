import os
import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from torchsummary import summary

from ignite.engine.engine import Engine, State, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite._utils import convert_tensor

from utils import Experiment
from utils.factory import *
from utils.helpers import static_vars

import scipy.io as sio

logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

@static_vars(state_dict=None)
def accumulate_predictions(engine):
    y_pred, y_true = engine.state.output

    if accumulate_predictions.state_dict is None:
        accumulate_predictions.state_dict = {'y_pred': y_pred, 'y_true': y_true}
    else:
        accumulate_predictions.state_dict['y_pred'] = torch.cat((accumulate_predictions.state_dict['y_pred'], y_pred), 0)
        accumulate_predictions.state_dict['y_true'] = torch.cat((accumulate_predictions.state_dict['y_true'], y_true), 0)

def save_predictions(engine, base_path=None, state_dict=None):
    state_dict['y_pred'] = state_dict['y_pred'].cpu().numpy()
    state_dict['y_true'] = state_dict['y_true'].cpu().numpy()

    sio.savemat("{}_inference.mat".format(base_path), state_dict)

def main(config):
    assert validate_config(config), "ERROR: Config file is invalid. Please see log for details."

    logger.info("INFO: {}".format(config.toDict()))

    if config.device == "cpu" and torch.cuda.is_available():
        logger.warning("WARNING: Not using the GPU")

    assert 'test' in config.datasets, "ERROR: Not test dataset is specified in the config. Don't know how to proceed."

    logger.info("INFO: Creating datasets and dataloaders...")

    config.datasets.test.update({'shuffle': False, 'augment': False, 'workers': 1})

    # Create the training dataset
    dset_test = create_dataset(config.datasets.test)

    config.datasets.test.update({'batch_size': 1})
    loader_test = get_data_loader(dset_test, config.datasets.test)

    logger.info("INFO: Running inference on {} samples".format(len(dset_test)))

    logger.info("INFO: Building the {} model".format(config.model.type))
    model = build_model(config.model)

    m_cp_path, _ = config.get_checkpoint_path()

    assert m_cp_path, "Could not find a checkpoint for this model, check your config and try again"
    model.load_state_dict(torch.load( m_cp_path ))
    logger.info("INFO: Loaded model checkpoint {}".format(m_cp_path))

    model = model.to(config.device)

    if 'input_size' in config:
        summary(model, input_size=config.input_size, device=config.device, unpack_inputs=True)
    else:
        print(model)

    if 'logger' in config:
        logger.info("INFO: Initialising the experiment logger")
        exp_logger = get_experiment_logger(config)

    logger.info("INFO: Creating training manager and configuring callbacks")
    trainer = get_trainer(model, None, None, None, config)

    evaluator_engine = Engine(trainer.evaluate)

    trainer.attach("test_loader", loader_test)
    trainer.attach("evaluation_engine", evaluator_engine)

    if 'metrics' in config:
        for name, metric in config.metrics.items():
            metric = get_metric(metric)
            if metric is not None:
                metric.attach(evaluator_engine, name)
            else:
                logger.warning("WARNING: Metric {} could not be created".format(name))

    # Register custom callbacks with the engines
    if check_if_implemented(trainer, "on_iteration_start"):
        evaluator_engine.add_event_handler(Events.ITERATION_STARTED, trainer.on_iteration_start, phase="test")
    if check_if_implemented(trainer, "on_iteration_end"):
        evaluator_engine.add_event_handler(Events.ITERATION_COMPLETED, trainer.on_iteration_end, phase="test")
    if check_if_implemented(trainer, "on_epoch_start"):
        evaluator_engine.add_event_handler(Events.EPOCH_STARTED, trainer.on_epoch_start, phase="test")
    if check_if_implemented(trainer, "on_epoch_end"):
        evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, trainer.on_epoch_end, phase="test")

    evaluator_engine.add_event_handler(Events.ITERATION_COMPLETED, accumulate_predictions)
    # evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, save_predictions, base_path=os.path.splitext(m_cp_path)[0], state_dict=accumulate_predictions.state_dict)

    logger.info("INFO: Starting inference...")
    evaluator_engine.run(loader_test)

    save_predictions(evaluator_engine, os.path.splitext(m_cp_path)[0], accumulate_predictions.state_dict)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='config file path (default: None)')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint to use for test')
    args = parser.parse_args()

    config = Experiment.load_from_path(args.config)

    if args.checkpoint:
        config.checkpoint = args.checkpoint

    assert config, "Config could not be loaded."

    main(config)
