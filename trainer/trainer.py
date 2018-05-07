from ignite.engine.engine import Engine, State, Events

class DefaultTrainer(Engine):
    def __init__(self, model, optimizer, loss_fn, config):
        super(DefaultTrainer, self).__init__(self)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        

    def __call__(self, epoch):
