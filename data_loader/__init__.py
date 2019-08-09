from .cityscapes import CitySegmentation
from .warehouse import WarehouseSegmentation

datasets = {
    #'citys': CitySegmentation,
    'warehouse': WarehouseSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
