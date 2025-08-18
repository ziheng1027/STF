import yaml
from Tool.Utils import get_trainer


# 训练哪个模型?
model_name = "UNet"

if __name__ == '__main__':
    config_path = f"Config/{model_name}.yml"
    datasets_config_path = "Config/Dataset.yml"
    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    datasets_config = yaml.safe_load(open(datasets_config_path, 'r', encoding='utf-8'))

    trainer = get_trainer(model_name, config, datasets_config)
    trainer.train()