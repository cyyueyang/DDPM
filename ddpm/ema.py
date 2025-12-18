class EMA:
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1.0 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_param, ema_param in zip(current_model.parameters(), ema_model.parameters()):
            new, old = current_param.data, ema_param.data
            ema_param.data = self.update_average(old, new)

