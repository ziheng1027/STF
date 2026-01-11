# Trainer/Trainer_PredRNN.py
from Trainer.Trainer_IMF import Trainer_IMF


class Trainer(Trainer_IMF):
    def __init__(self, model_config, datasets_config, metric_config, dataset_name):
        super().__init__(model_config, datasets_config, metric_config, dataset_name)

    def get_model(self):
        from Model.PredRNN import PredRNN
        model = PredRNN(**self.model_config['model'])
        model_version = self.model_config["model"].get("model_version", "") 
        if model_version == "V2":
            return model.to(self.device), f"{model.__class__.__name__}_{model_version}"
        else:
            model_version = "V1"
            return model.to(self.device), f"{model.__class__.__name__}_{model_version}"