from Trainer.Trainer_DMF import Trainer_DMF


class Trainer(Trainer_DMF):
    def __init__(self, model_config, datasets_config, metric_config, dataset_name):
        super().__init__(model_config, datasets_config, metric_config, dataset_name)

    def get_model(self):
        from Model.SmaAtUNet import SmaAtUNet
        model = SmaAtUNet(**self.model_config['model'])
        return model.to(self.device), model.__class__.__name__