import yaml
from Tool.Utils import get_trainer, visualize_figure, set_seed


# 训练哪个模型?
model_name = "SimVP"
# 使用哪个数据集？
dataset_name = "MovingMNIST"
# 选择哪个模式? "train", "test" or "visualize"
mode = "train"

if __name__ == '__main__':
    # 获取配置文件
    config_path = f"Config/{dataset_name}/{model_name}.yml"
    datasets_config_path = "Config/Dataset.yml"
    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    datasets_config = yaml.safe_load(open(datasets_config_path, 'r', encoding='utf-8'))

    # 设置随机种子
    seed = config.get("seed", 42)
    set_seed(seed)

    trainer = get_trainer(model_name, dataset_name, config, datasets_config)

    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test()
    elif mode == "visualize":
        visualize_figure(model_name, dataset_name)