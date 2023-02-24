import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

import tifffile as tiff
import os
import cv2


def read_tif(filename, channels=None):
    assert os.path.exists(filename)
    img = tiff.imread(filename)
    if len(img.shape) < 3:
        img = img[..., None]
    img = img[:, :, :channels] if channels else img
    return img

def to_3dim(img):
    if len(img.shape) == 2:
        return img[..., None]
    else:
        return img

def add_img(img1, img2):
    if len(img1.shape) == 2:
        img1 = img1[:, np.newaxis]
    if len(img2.shape) == 2:
        img2 = img2[:, np.newaxis]
    return img1 + img2 / 2


@PIPELINES.register_module()
class LoadMultiSourceTifFromFile(object):
    def __init__(self, to_float32=False,
                 fusion_type='pmC',
                 src_types=['pan', 'ms', 'fusion', 'blur_pan']):
        self.to_float32 = to_float32
        self.fusion_type = fusion_type
        self.src_types = src_types

    def __call__(self, results):
        assert results['img_prefix'] is not None
        img_prefix = results['img_prefix']
        file_name = results['img_info']['filename']
        for t in self.src_types:
            org_type = None
            if t in img_prefix:
                org_type = t
                break
            if org_type == None:
                raise Exception('img_prefix: %s has no type in src_types: %s' %
                                (file_name, str(self.src_types)))
            assert org_type is not None

        img_pths = {t: img_prefix.replace(org_type, t) + '/' + file_name
                    for t in self.src_types}
        # [pan]
        if self.fusion_type == 'pan':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            img = pan_img
        elif self.fusion_type == 'fusion_PANNet':
            fusion_img = read_tif(img_pths['fusion_PANNet'])
            img = fusion_img
        elif self.fusion_type == 'fusion_PSGAN':
            fusion_img = read_tif(img_pths['fusion_PSGAN'])
            img = fusion_img
        elif self.fusion_type == 'masked_pan_15':
            pan_img = to_3dim(read_tif(img_pths['masked_pan_15']))
            img = pan_img
        elif self.fusion_type == 'masked_pan_20':
            pan_img = to_3dim(read_tif(img_pths['masked_pan_20']))
            img = pan_img
        elif self.fusion_type == 'masked_pan_25':
            pan_img = to_3dim(read_tif(img_pths['masked_pan_25']))
            img = pan_img
        elif self.fusion_type == 'masked_pan_40':
            pan_img = to_3dim(read_tif(img_pths['masked_pan_40']))
            img = pan_img
        elif self.fusion_type == 'result_mask_pan':
            pan_img = to_3dim(read_tif(img_pths['result_mask_pan']))
            img = pan_img
        # [pan, pan, pan]
        elif self.fusion_type == 'pan3c':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            pan3c_img = np.concatenate([pan_img,
                                        pan_img,
                                        pan_img], axis=-1)
            img = pan3c_img
        # [ms]
        elif self.fusion_type == 'ms':
            ms_img = read_tif(img_pths['ms'])
            img = ms_img
        # [fusion]
        elif self.fusion_type == 'fusion':
            fusion_img = read_tif(img_pths['fusion'])
            img = fusion_img
        # [fusion]
        elif self.fusion_type == 'fusion3':
            fusion_img = read_tif(img_pths['fusion'])
            img = fusion_img[:, :, :3]
        # [pan, ms]
        elif self.fusion_type == 'pmC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            ms_img = read_tif(img_pths['ms'])
            img = np.concatenate((pan_img, ms_img), axis=-1)
        # [pan+ms]
        elif self.fusion_type == 'pmA':
            pan_img = read_tif(img_pths['pan'])
            ms_img = read_tif(img_pths['ms'])
            img = add_img(pan_img, ms_img)
        ############ New in JSTARS  #####################
        # [pan, pan]
        elif self.fusion_type == 'ppC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            img = np.concatenate((pan_img, pan_img), axis=-1)
        # [ms, ms]
        elif self.fusion_type == 'mmC':
            ms_img = read_tif(img_pths['ms'])
            img = np.concatenate((ms_img, ms_img), axis=-1)
        ############ New in JSTARS  #####################

        # [pan, fusion]
        elif self.fusion_type == 'pfC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            fusion_img = read_tif(img_pths['fusion'])
            img = np.concatenate((pan_img, fusion_img), axis=-1)
        # [pan, pan + ms]
        elif self.fusion_type == 'ppmAC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            ms_img = read_tif(img_pths['ms'])
            pmA_img = add_img(pan_img, ms_img)
            img = np.concatenate((pan_img, pmA_img), axis=-1)
        # [pan, ms, pan + ms]
        elif self.fusion_type == 'pmCpmAC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            ms_img = read_tif(img_pths['ms'])
            pmA_img = add_img(pan_img, ms_img)
            img = np.concatenate((pan_img, ms_img, pmA_img), axis=-1)
        # [pan, ms, fusion]
        elif self.fusion_type == 'pmCfC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            ms_img = read_tif(img_pths['ms'])
            fusion_img = read_tif(img_pths['fusion'])
            img = np.concatenate((pan_img, ms_img, fusion_img), axis=-1)
        # [pan, blur_pan]
        elif self.fusion_type == 'pbpC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            blur_pan_img = to_3dim(read_tif(img_pths['blur_pan']))
            #print('#' * 100)
            # print(pan_img.shape, blur_pan_img.shape)
            img = np.concatenate((pan_img, blur_pan_img), axis=-1)
        elif self.fusion_type == 'p3cfC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            pan3c_img = np.concatenate([pan_img,
                                        pan_img,
                                        pan_img], axis=-1)
            fusion_img = read_tif(img_pths['fusion'])
            img = np.concatenate((pan3c_img, fusion_img), axis=-1)
        elif self.fusion_type == 'mpC':
            pan_img = to_3dim(read_tif(img_pths['pan']))
            ms_img = read_tif(img_pths['ms'])
            img = np.concatenate((ms_img, pan_img), axis=-1)
        else:
            raise Exception('Fusion type %s not supported' % self.fusion_type)

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = img_pths[org_type]
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        if len(img.shape) == 3:
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
        else:
            results['img_shape'] = (img.shape[0], img.shape[1], 1)
            results['ori_shape'] = (img.shape[0], img.shape[1], 1)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        return '{} (to_float32={},)'.format(
            self.__class__.__name__, self.to_float32)


@PIPELINES.register_module()
class LoadTifFromFile(object):
    def __init__(self, to_float32=False,
                 channels=None,
                 extra=None,
                 channels_extra=None):
        self.to_float32 = to_float32
        self.channels = channels
        self.extra = extra
        self.channels_extra = channels_extra
        assert self.extra is None or self.extra in ['pan', 'ms', 'fusion']

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = read_tif(filename, self.channels)
        if self.extra:
            ori_type = None
            for t in ['pan', 'ms', 'fusion']:
                if t in filename:
                    ori_type = t
                    break
            assert ori_type is not None
            filename_extra = filename.replace(ori_type, self.extra)
            img_extra = read_tif(filename_extra, self.channels_extra)
            results['filename_extra'] = filename_extra
            img = np.concatenate((img, img_extra), axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        return '{} (to_float32={},)'.format(
            self.__class__.__name__, self.to_float32)
