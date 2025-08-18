import logging
from tabulate import tabulate

class Logger:
    """训练/测试日志记录器"""
    def __init__(self, model_name, dir_log):
        log_file = f"{dir_log}/{model_name}.log"
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.INFO)

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """记录信息级别日志"""
        self.logger.info(message)

    def log_metrics(self, metrics):
        """以表格形式记录评估指标"""
        # 指标名称作为表头
        headers = list(metrics.keys())
        data = [list(metrics.values())]
        table = tabulate(data, headers=headers, tablefmt="grid")
        self.logger.info("\n" + table)
