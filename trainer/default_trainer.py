from ignite.engine.engine import Engine, State, Events
from ignite._utils import convert_tensor

class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = "gpu" if config.cuda else "cpu"

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

    def on_epoch_start(self, engine):
        pass

    def on_epoch_end(self, engine):
        pass

    def on_iteration_start(self, engine):
        pass

    def on_iteration_end(self, engine):
        pass

    def inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = self._prepare_batch(batch)
            y_pred = self.model(x)
            return y_pred, x



trainer(model, loss, optimizer)
trainner register lr_scheduler callback( pass the model to it )
triner register early stopping
trainer register custom callbacks
trainer register checkpoint
enine.run(loader)

on running:

- call trainer with the current batch
-
