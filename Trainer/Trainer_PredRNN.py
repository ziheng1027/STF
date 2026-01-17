# Trainer/Trainer_PredRNN.py
from Trainer.Trainer_IMF import Trainer_IMF


class Trainer(Trainer_IMF):
    def __init__(self, model_config, datasets_config, metric_config, dataset_name):
        super().__init__(model_config, datasets_config, metric_config, dataset_name)

    def get_model(self):
        from Model.PredRNN import PredRNN
        model = PredRNN(**self.model_config['model'])
        return model.to(self.device), model.__class__.__name__