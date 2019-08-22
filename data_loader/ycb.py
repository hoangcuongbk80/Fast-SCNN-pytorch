"""Cityscapes Dataloader"""
import os
import glob
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['ycbSegmentation']


class ycbSegmentation(data.Dataset):
    BASE_DIR = 'YCB-Video'
    NUM_CLASS = 21+1 # 13 object and background

    def __init__(self, root='./datasets/YCB-Video', split='train', mode=None, rgb_transform=None, depth_transform=None,
                 base_size=520, crop_size=480, **kwargs):
        super(ycbSegmentation, self).__init__()
        self.root = '/media/aass/783de628-b7ff-4217-8c96-7f3764de70d9/RGBD_DATASETS/YCB_Video_Dataset'
        #self.root = '/YCB_Video_Dataset'
        self.split = split
        self.mode = mode if mode is not None else split
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.rgb_paths, self.depth_paths, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.rgb_paths) == len(self.mask_paths))
        
    def __getitem__(self, index):
        rgb = Image.open(self.rgb_paths[index]).convert('RGB')
        depth = Image.open(self.depth_paths[index])

        if self.mode == 'test':
            if self.rgb_transform is not None:
                rgb = self.rgb_transform(rgb)
                depth = self.depth_transform(depth)
            return rgb, depth,os.path.basename(self.rgb_paths[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            rgb, depth, mask = self._sync_transform(rgb, depth, mask)
        elif self.mode == 'val':
            rgb, depth, mask = self._val_sync_transform(rgb, depth, mask)
        else:
            assert self.mode == 'testval'
            rgb, depth, mask = self._img_transform(rgb), self._img_transform(depth), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(rgb)
            depth = self.depth_transform(depth)
        return rgb, depth, mask

    def _val_sync_transform(self, rgb, depth, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = rgb.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        rgb = rgb.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = rgb.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        rgb = rgb.crop((x1, y1, x1 + outsize, y1 + outsize))
        depth = depth.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        rgb, depth, mask = self._img_transform(img), self._img_transform(depth), self._mask_transform(mask)
        return rgb, depth, mask

    def _sync_transform(self, rgb, depth, mask):
        # random mirror
        if random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = rgb.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        rgb = rgb.resize((ow, oh), Image.BILINEAR)
        depth=depth.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            rgb = ImageOps.expand(rgb, border=(0, 0, padw, padh), fill=0)
            depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)            
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = rgb.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        rgb = rgb.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        depth = depth.crop((x1, y1, x1 + crop_size, y1 + crop_size))        
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            rgb = rgb.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        rgb, depth, mask = self._img_transform(rgb), self._depth_transform(depth), self._mask_transform(mask)
        return rgb, depth, mask

    def _img_transform(self, img):
        return np.array(img)

    def _depth_transform(self, depth):
        depth = np.array(depth)
        depth = depth / 3000
        depth[depth > 1] = 1
        return depth

    def _mask_transform(self, mask):
        #target = self._class_to_index(np.array(mask).astype('int32'))
        target = mask
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.rgb_paths)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0



def _get_city_pairs(folder, split='train'):
    val = ['0048', '0049', '0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059']
    data_dir = os.path.join(folder, 'data')
    img_paths = []
    depth_paths = []
    mask_paths = []

    if split=='train':
        for i in range(0, 93):
            seq = 10000 + i
            seq_str = str(seq)[1:]
            
            img_dir = os.path.join(data_dir, seq_str)
            label_addrs = glob.glob(img_dir + '/*-label.png')
            
            if seq_str not in val:
                for j in range(0, len(label_addrs)):
                    if j % 8 != 0:
                        continue
                    img_index = 1000000 + j + 1
                    img_index_str = str(img_index)[1:]

                    img_path = img_dir + '/' + img_index_str + '-color.png'                    
                    depth_path = img_dir + '/' + img_index_str + '-depth.png'                    
                    label_path = img_dir + '/' + img_index_str + '-label.png'
                    
                    img_paths.append(img_path)
                    depth_paths.append(depth_path)                    
                    mask_paths.append(label_path)

    return img_paths, depth_paths, mask_paths


if __name__ == '__main__':
    dataset = Warehouse()
    img, label = dataset[0]
