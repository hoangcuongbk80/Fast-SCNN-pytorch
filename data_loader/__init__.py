from .cityscapes import CitySegmentation
from .warehouse import WarehouseSegmentation
from .ycb import ycbSegmentation

datasets = {
    'citys': CitySegmentation,
    'warehouse': WarehouseSegmentation,
    'ycb-video': ycbSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
