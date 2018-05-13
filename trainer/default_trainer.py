from ignite.engine.engine import Engine, State, Events
from ignite._utils import convert_tensor

class DefaultTrainer:
    def __init__(self, model, optimizer, loss_fn, logger, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = config.device

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
        pass

    def on_iteration_start(self, engine, phase=None):
        pass

    def on_iteration_end(self, engine, phase=None):
        pass

    def evaluate(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = self._prepare_batch(batch)
            y_pred = self.model(x)
            return y_pred, y
