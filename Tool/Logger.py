import logging
from tabulate import tabulate

class Logger:
    """训练/测试日志记录器"""
    def __init__(self, log_file="train.log"):
        self.logger = logging.getLogger()
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

    def log_metrics(self, metrics, epoch):
        """以表格形式记录评估指标"""
        headers = ["Metric", "Value"]
        data = [(key, value) for key, value in metrics.items()]
        
        # 如果有epoch信息, 添加到表格中
        if epoch is not None:
            data.insert(0, ("Epoch", epoch))
        
        table = tabulate(data, headers=headers, tablefmt="grid")
        self.logger.info("\n" + table)
