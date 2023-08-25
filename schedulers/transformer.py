import math

class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.iters = 0
		self.lr_list = []

	def step(self):
        self.iters += 1
        if 