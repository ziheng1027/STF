from Trainer.Trainer_DMF import Trainer_DMF


class Trainer(Trainer_DMF):
    def __init__(self, config, datasets_config, dataset_name):
        super().__init__(config, datasets_config, dataset_name)

    def get_model(self):
        from Model.SmaAtUNet import SmaAtUNet
        model = SmaAtUNet(**self.config['model'])
        return model.to(self.device), model.__class__.__name__