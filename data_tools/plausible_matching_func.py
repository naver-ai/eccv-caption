"""
Plausible Matching (PM) scores
Rereference code: https://github.com/naver-ai/pcme/blob/main/evaluate_pmrp_coco.py
"""
import numpy as np
from tqdm import tqdm

try:
    import ujson as json
except ImportError:
    import warnings
    warnings.warn('failed to import ujson. ``pip install ujson`` will make the script faster.')
    import json


def get_plausible_matching_scores(coco_instance_annotation_path,
                                  orig_image_to_captions,
                                  all_image_ids):
    with open(coco_instance_annotation_path) as fin:
        instance_ann = json.load(fin)

    iid_to_cls = {}
    for ann in instance_ann['annotations']:
        iid = int(ann['image_id'])
        if iid not in all_image_ids:
            continue
        code = iid_to_cls.get(iid, [0] * 90)
        code[int(ann['category_id']) - 1] = 1
        iid_to_cls[iid] = code

    iid_to_codes = np.zeros((len(iid_to_cls), 90), dtype=np.bool)
    val_image_ids = list(iid_to_cls.keys())
    for idx, _id in enumerate(val_image_ids):
        iid_to_codes[idx] = iid_to_cls[_id]

    # Fast pairwise hamming distance (with a very large memory size)
    # pdists = np.count_nonzero(iid_to_codes[:, None, :] != iid_to_codes[None, :, :], axis=-1)

    N = iid_to_codes.shape[0]
    pdists = np.zeros((N, N))
    for idx, code in tqdm(enumerate(iid_to_codes), total=len(iid_to_codes)):
        pdists[idx] = np.count_nonzero(iid_to_codes != code, axis=1)
    return pdists, np.array(val_image_ids)


def convert_pm_scores_to_matched_pairs(pm_image_pairs,
                                       image_ids,
                                       orig_image_to_captions,
                                       omit_orig):
    image_to_captions = {}
    caption_to_images = {}

    for idx, pm_idx in enumerate(pm_image_pairs):
        _iid = int(image_ids[idx])
        _cids = orig_image_to_captions[_iid]
        _pm_iids = set(image_ids[pm_idx])
        if omit_orig:
            _pm_iids.remove(_iid)

        for __cid in _cids:
            __c2i = caption_to_images.setdefault(int(__cid), set())
            __c2i |= _pm_iids

        for __pm_iid in _pm_iids:
            __pm_cids = set(orig_image_to_captions[__pm_iid])
            __i2c = image_to_captions.setdefault(int(_iid), set())
            __i2c |= __pm_cids

    return image_to_captions, caption_to_images


def get_plausble_matching_data(coco_instance_annotation_path,
                               orig_image_to_captions,
                               all_iids,
                               pm_thres,
                               omit_orig=False):
    all_iids = set(list(all_iids))
    pdists, image_ids = get_plausible_matching_scores(coco_instance_annotation_path,
                                                      orig_image_to_captions,
                                                      all_iids)
    pm_image_pairs = (pdists <= pm_thres)
    image_to_captions, caption_to_images = \
        convert_pm_scores_to_matched_pairs(pm_image_pairs,
                                           image_ids,
                                           orig_image_to_captions,
                                           omit_orig=omit_orig)

    return image_to_captions, caption_to_images
