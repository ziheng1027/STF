import yaml
from Tool.Utils import get_trainer, visualize_figure, set_seed


# 训练哪个模型?
model_name = "UNet"
# 使用哪个数据集？
dataset_name = "SEVIR"
# 选择哪个模式? "train", "test" or "visualize"
mode = "test"


if __name__ == '__main__':
    # 模型配置
    model_config_path = f"Config/{dataset_name}/{model_name}.yml"
    model_config = yaml.safe_load(open(model_config_path, 'r', encoding='utf-8'))
    # 数据集配置
    dataset_config_path = "Config/Dataset.yml"
    dataset_config = yaml.safe_load(open(dataset_config_path, 'r', encoding='utf-8'))
    # 评估指标配置
    metric_config_path = "Config/Metric.yml"
    metric_config = yaml.safe_load(open(metric_config_path, 'r', encoding='utf-8'))

    # 设置随机种子
    seed = model_config.get("seed", 42)
    set_seed(seed)

    trainer = get_trainer(model_name, dataset_name, model_config, dataset_config, metric_config)

    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test()
    elif mode == "visualize":
        visualize_figure(model_name, dataset_name)