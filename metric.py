import numpy as np
import torch
import settings


def fast_hist(label_true, label_pred):
    n_class = settings.N_CLASSES
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask].int() + label_pred[mask].int(),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


label_names = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


def cal_scores(hist):
    n_class = settings.N_CLASSES
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(label_names, iu))

    return {
        'pAcc': acc,
        'mAcc': acc_cls,
        'fIoU': fwavacc,
        'mIoU': mean_iu,
    }, cls_iu

