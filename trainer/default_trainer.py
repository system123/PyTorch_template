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
        self.log_freq = config.log_freq
        self.attached = {}
        self.curr_epoch = 0

    def _prepare_batch(self, batch):
        xs, ys = batch

        if isinstance(xs, list):
            xs = [convert_tensor(x, self.device).float() for x in xs]
        else:
            xs = [convert_tensor(xs, self.device).float()]

        if isinstance(ys, list):
            ys = [convert_tensor(y, self.device).float() for y in ys]
        else:
            ys = [convert_tensor(ys, self.device).float()]

        return xs, ys

    def train(self, engine, batch):
        self.model.train()

        self.optimizer.zero_grad()

        xs, ys = self._prepare_batch(batch)
        y_pred = self.model(*xs)

        if not (isinstance(y_pred, list) or isinstance(y_pred, tuple)):
            ys = ys[0]

        loss = self.loss_fn(y_pred, ys)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def on_epoch_start(self, engine, phase=None):
        if phase == "train":
            self.curr_epoch = engine.state.epoch

    def on_epoch_end(self, engine, phase=None):
        if phase in ["evaluate", "test"]:
            metrics = engine.state.metrics
            log = ""
            for k, v in metrics.items():
                log += "{}: {:.2f}  ".format(k, v)

            print("{} Results - Epoch: {}  {}".format(phase.capitalize(), self.curr_epoch, log))


    def on_iteration_start(self, engine, phase=None):
        if phase == "train":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["train_loader"]) + 1
            if curr_iter % self.log_freq == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(engine.state.epoch, curr_iter, len(self.attached["train_loader"]), engine.state.output))
        elif phase == "test":
            curr_iter = (engine.state.iteration - 1) % len(self.attached["test_loader"]) + 1
            if curr_iter % self.log_freq == 0:
                print("Iteration[{}/{}]".format(curr_iter, len(self.attached["test_loader"])))

    def on_iteration_end(self, engine, phase=None):
        pass

    def evaluate(self, engine, batch):
        self.model.eval()

        with torch.no_grad():
            xs, ys = self._prepare_batch(batch)
            y_pred = self.model(*xs)

            if not (isinstance(y_pred, list) or isinstance(y_pred, tuple)):
                ys = ys[0]

            return y_pred.float(), ys.float()

    def attach(self, name, obj):
        self.attached[name] = obj
