import yaml
from Tool.Utils import get_trainer, visualize_figure


# 训练哪个模型?
model_name = "UNet"
# 使用哪个数据集？
dataset_name = "TaxiBJ"
# 训练模式? "train", "test" or "visualize"
mode = "train"

if __name__ == '__main__':
    config_path = f"Config/{dataset_name}/{model_name}.yml"
    datasets_config_path = "Config/Dataset.yml"
    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    datasets_config = yaml.safe_load(open(datasets_config_path, 'r', encoding='utf-8'))

    trainer = get_trainer(model_name, dataset_name, config, datasets_config)

    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test()
    elif mode == "visualize":
        visualize_figure(model_name, dataset_name)