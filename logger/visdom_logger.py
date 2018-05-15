import visdom
import numpy as np

class VisdomLogger:

	def __init__(self, config):
		self.vis = visdom.Visdom()

		if not self.vis.check_connection():
			raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

		if config.overwrite:
			self.vis.close()

		self.windows = {}
		self.windows["evaluate"] = None
		self.windows["train"] = None
		self.counters = {"evaluate": 0, "train": 0}
		self.epochs = {"evaluate": 0, "train": 0}

	def _create_plot_window(self, xlabel, ylabel, title):
		return self.vis.line( X=np.array([1]), Y=np.array([np.nan]), \
				opts=dict(xlabel=xlabel, ylabel=ylabel, title=title) )

	def _create_plot_windows(self, metrics, title=""):
		windows = {}

		for m, v in metrics.items():
			windows[m] = self._create_plot_window("#Iterations", m, "{} {}".format(title, m))

		return windows

	def _plot_metrics(self, iteration, metrics, windows):
		for m, v in metrics.items():
			if m in windows:
				self.vis.line(X=np.array([iteration]), Y=np.array([v]), win=windows[m], update='append')

	def log_iteration(self, engine, phase="train", model=None):
		metrics = engine.state.metrics
		self.counters[phase] += 1

		if self.counters[phase] >= 100:
			# Overwrite this because we don't actually hve any metrics but we want the train loss
			if phase == "train":
				metrics = {'train_loss': engine.state.output}

			if self.windows[phase] is None:
				self.windows[phase] = self._create_plot_windows(metrics, title=phase.capitalize())

			if len(self.windows[phase]) > 0:
				self._plot_metrics(engine.state.iteration, metrics, self.windows[phase])

			self.counters[phase] = 0

	def log_epoch(self, engine, phase="train", model=None):
		if self.windows[phase] is None:
				self.windows[phase] = self._create_plot_windows(engine.state.metrics, title=phase.capitalize())

		self._plot_metrics(self.epochs[phase], engine.state.metrics, self.windows[phase])

		if phase == "evaluate":
			self.epochs[phase] += 1
		else:
			self.epochs[phase] = engine.state.epoch

