# Tool/Colormap.py
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm


class ColorMapBase:
    """颜色映射基类"""
    @staticmethod
    def get(data_type=None, **kwargs):
        """获取颜色映射配置"""
        raise NotImplementedError("子类必须实现get方法")


class SEVIRColormap(ColorMapBase):
    """SEVIR数据集颜色映射"""
    @staticmethod
    def get_vil_cmap():
        """获取VIL(垂直积分液态水)的颜色映射"""
        # VIL反射率颜色定义, 从无回波到强回波渐变
        cols = [
            [0, 0, 0],                   # 黑色: 无回波
            [0.30196, 0.30196, 0.30196], # 深灰: 弱回波
            [0.15686, 0.74510, 0.15686], # 绿色: 轻度回波
            [0.09804, 0.58824, 0.09804], # 深绿
            [0.03922, 0.41176, 0.03922], # 更深绿
            [0.03922, 0.29412, 0.03922], # 最深绿
            [0.96078, 0.96078, 0],       # 黄色: 中等回波
            [0.92941, 0.67451, 0],       # 橙黄
            [0.94118, 0.43137, 0],       # 橙色
            [0.62745, 0, 0],             # 深红: 强回波
            [0.90588, 0, 1.0]            # 紫色: 极强回波
        ]

        # VIL反射率边界值
        lev = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0,
               133.0, 160.0, 181.0, 219.0, 255.0]

        nil = cols.pop(0)

        cmap = mpl.colors.ListedColormap(cols)
        cmap.set_bad(nil)
        cmap.set_under(cols[0])
        cmap.set_over(cols[-1])
        norm = BoundaryNorm(lev, cmap.N)

        return cmap, norm, None, None

    @staticmethod
    def get_vis_cmap():
        """可见光通道颜色映射"""
        cols = [
            [0, 0, 0], [0.03922, 0.03922, 0.03922],
            [0.07843, 0.07843, 0.07843], [0.11765, 0.11765, 0.11765],
            [0.15686, 0.15686, 0.15686], [0.19608, 0.19608, 0.19608],
            [0.23529, 0.23529, 0.23529], [0.27451, 0.27451, 0.27451],
            [0.31373, 0.31373, 0.31373], [0.35294, 0.35294, 0.35294],
            [0.39216, 0.39216, 0.39216], [0.43137, 0.43137, 0.43137],
            [0.47059, 0.47059, 0.47059], [0.50980, 0.50980, 0.50980],
            [0.54902, 0.54902, 0.54902], [0.58824, 0.58824, 0.58824],
            [0.62745, 0.62745, 0.62745], [0.66667, 0.66667, 0.66667],
            [0.70588, 0.70588, 0.70588], [0.74510, 0.74510, 0.74510],
            [0.78431, 0.78431, 0.78431], [0.82353, 0.82353, 0.82353],
            [0.86275, 0.86275, 0.86275], [0.90196, 0.90196, 0.90196],
            [0.94118, 0.94118, 0.94118], [0.98039, 0.98039, 0.98039],
            [0.98039, 0.98039, 0.98039]
        ]

        lev = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16,
                        0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52,
                        0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.9, 1.0])
        lev *= 10000

        nil = cols[0]
        under = cols[0]
        over = cols.pop()

        cmap = mpl.colors.ListedColormap(cols)
        cmap.set_bad(nil)
        cmap.set_under(under)
        cmap.set_over(over)
        norm = BoundaryNorm(lev, cmap.N)

        return cmap, norm, 0, 10000

    @staticmethod
    def get_ir_cmap():
        """红外通道颜色映射"""
        cols = [
            [0, 0, 0], [1.0, 1.0, 1.0],
            [0.98039, 0.98039, 0.98039], [0.94118, 0.94118, 0.94118],
            [0.90196, 0.90196, 0.90196], [0.86275, 0.86275, 0.86275],
            [0.82353, 0.82353, 0.82353], [0.78431, 0.78431, 0.78431],
            [0.74510, 0.74510, 0.74510], [0.70588, 0.70588, 0.70588],
            [0.66667, 0.66667, 0.66667], [0.62745, 0.62745, 0.62745],
            [0.58824, 0.58824, 0.58824], [0.54902, 0.54902, 0.54902],
            [0.50980, 0.50980, 0.50980], [0.47059, 0.47059, 0.47059],
            [0.43137, 0.43137, 0.43137], [0.39216, 0.39216, 0.39216],
            [0.35294, 0.35294, 0.35294], [0.31373, 0.31373, 0.31373],
            [0.27451, 0.27451, 0.27451], [0.23529, 0.23529, 0.23529],
            [0.19608, 0.19608, 0.19608], [0.15686, 0.15686, 0.15686],
            [0.11765, 0.11765, 0.11765], [0.07843, 0.07843, 0.07843],
            [0.03922, 0.03922, 0.03922], [0.0, 0.80392, 0.80392]
        ]

        lev = np.array([-110., -105.2, -95.2, -85.2, -75.2, -65.2, -55.2, -45.2,
                        -35.2, -28.2, -23.2, -18.2, -13.2, -8.2, -3.2, 1.8,
                        6.8, 11.8, 16.8, 21.8, 26.8, 31.8, 36.8, 41.8,
                        46.8, 51.8, 90., 100.])
        lev *= 100

        nil = cols.pop(0)
        under = cols[0]
        over = cols.pop()

        cmap = mpl.colors.ListedColormap(cols)
        cmap.set_bad(nil)
        cmap.set_under(under)
        cmap.set_over(over)
        norm = BoundaryNorm(lev, cmap.N)

        return cmap, norm, -8000, -1000

    @staticmethod
    def get(data_type='vil', **kwargs):
        """获取SEVIR数据集的颜色映射, data_type支持 'vil'(默认), 'vis', 'ir', 'light'"""
        if data_type.lower() == 'vil':
            return SEVIRColormap.get_vil_cmap()
        elif data_type.lower() == 'vis':
            return SEVIRColormap.get_vis_cmap()
        elif data_type.lower() in ['ir069', 'ir']:
            return SEVIRColormap.get_ir_cmap()
        elif data_type.lower() == 'light':
            return 'hot', None, 0, 5
        else:
            return 'jet', None, None, None


class MovingMNISTColormap(ColorMapBase):
    """MovingMNIST数据集颜色映射"""

    @staticmethod
    def get(data_type=None, **kwargs):
        """获取MovingMNIST数据集的颜色映射"""
        return 'gray', None, 0.0, 1.0


class TaxiBJColormap(ColorMapBase):
    """TaxiBJ数据集颜色映射"""

    @staticmethod
    def get(data_type=None, **kwargs):
        """获取TaxiBJ数据集的颜色映射

        Returns:
            tuple: (cmap, norm, vmin, vmax)
        """
        return 'viridis', None, 0.0, 1.0


# 数据集颜色映射注册表
_COLORMAP_REGISTRY = {
    'SEVIR': SEVIRColormap,
    'MovingMNIST': MovingMNISTColormap,
    'TaxiBJ': TaxiBJColormap
}


def register_colormap(dataset_name, colormap_class):
    """注册新的数据集颜色映射

    Args:
        dataset_name: 数据集名称
        colormap_class: 颜色映射类(需继承ColormapBase)

    Examples:
        >>> class MyDatasetColormap(ColormapBase):
        ...     @staticmethod
        ...     def get(data_type=None, **kwargs):
        ...         return 'plasma', None, 0, 1
        >>> register_colormap('MyDataset', MyDatasetColormap)
    """
    if not issubclass(colormap_class, ColorMapBase):
        raise ValueError("颜色映射类必须继承ColormapBase")
    _COLORMAP_REGISTRY[dataset_name] = colormap_class


def get_cmap(dataset_name, data_type=None, **kwargs):
    """根据数据集名称获取对应的颜色映射"""
    if dataset_name not in _COLORMAP_REGISTRY:
        supported = list(_COLORMAP_REGISTRY.keys())
        raise ValueError(f"不支持的数据集: {dataset_name}\n支持的数据集: {supported}")

    colormap_class = _COLORMAP_REGISTRY[dataset_name]
    return colormap_class.get(data_type, **kwargs)
