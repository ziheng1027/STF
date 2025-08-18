from Trainer.Trainer_DMF import Trainer_DMF


class Trainer(Trainer_DMF):
    def __init__(self, config, datasets_config):
        super().__init__(config, datasets_config)

    def get_model(self):
        from Model.UNet import UNet
        model = UNet(**self.config['model'])
        return model.to(self.device), model.__class__.__name__