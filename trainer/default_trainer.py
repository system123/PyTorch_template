import torch
import torch.nn.functional as F
from ignite.engine.engine import Engine, State, Events
from ignite._utils import convert_tensor

class DefaultTrainer:
    def __init__(self, model, optimizer, loss_fn, logger, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = config.device
        self.attached = {}

    def _prepare_batch(self, batch):
        x, y = batch
        x = convert_tensor(x, self.device)
        y = convert_tensor(y, self.device)
        return x, y

    def train(self, engine, batch):
        self.model.train()

        self.optimizer.zero_grad()

        x, y = self._prepare_batch(batch)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def on_epoch_start(self, engine, phase=None):
        pass

    def on_epoch_end(self, engine, phase=None):
        if phase == "evaluate":
            metrics = engine.state.metrics
            log = ""
            for k, v in metrics.items():
                log += "{}: {:.2f}  ".format(k, v)

            print("{} Results - Epoch: {}  {}".format(phase.capitalize(), engine.state.epoch, log))


    def on_iteration_start(self, engine, phase=None):
        if phase == "train":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["train_loader"]) + 1
            if curr_iter % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(engine.state.epoch, curr_iter, len(self.attached["train_loader"]), engine.state.output))

    def on_iteration_end(self, engine, phase=None):
        pass

    def evaluate(self, engine, batch):
        self.model.eval()

        with torch.no_grad():
            x, y = self._prepare_batch(batch)
            y_pred = self.model(x)

            return y_pred, y

    def attach(self, name, obj):
        self.attached[name] = obj
