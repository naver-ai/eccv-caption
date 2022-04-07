"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import os
import fire
import numpy as np
try:
    import ujson as json
except ImportError:
    import json

from cxc_func import read_cxc_data
from coco_func import read_coco_data
from plausible_matching_func import get_plausble_matching_data
from human_verified_pairs_func import get_human_verified_data


def convert_cxc_scores_to_matched_pairs(cxc_scores, cxc_thres, all_iids, all_cids, dump_negative):
    image_to_captions = {}
    caption_to_images = {}
    n, m = 0, 0
    for cid, _cxc_scores in cxc_scores.items():
        for (iid, score, _) in _cxc_scores:
            cid, iid, score = int(cid), int(iid), float(score)
            n += 1
            if (iid not in all_iids) or (cid not in all_cids):
                m += 1
                continue
            flag = score >= cxc_thres if not dump_negative else score < cxc_thres
            if flag:
                i2t = image_to_captions.setdefault(iid, set())
                i2t.add(cid)
                t2i = caption_to_images.setdefault(cid, set())
                t2i.add(iid)
    print(f'[CxC] {m} / {n} are not valid iid / cid')
    return image_to_captions, caption_to_images


def merge(coco_annotation_path,
          test_ids_path,
          output_dir,
          cxc_path=None,
          coco_instance_annotation_path=None,
          human_verified_annotation_path=None,
          cxc_thres=3,
          pm_thres=2,
          strict_pos=False,
          dump_negative=False,
          ):
    """Merge COCO / CxC annotations / PM annotations / Human-verificed annotations to a single json

    Args:
        cxc_path: Path to the CSV file containing CxC scores.
        cxc_thres: Threshold for cxc scores to positive (default: 3, following the paper)
            > Section 3 of CxC paper: Figure 4 gives the distribution of counts of positive examples
            > in each task (vali- dation split), where a score ≥ 3 (for STS, SITS)
            > and a score ≥ 2.5 (for SIS) is considered positive.

        pm_thres: Threshold for PM pairs to positive (default: 2, following the paper)

    Returns:
        Updated list of cxc_scores.
    """
    image_to_captions = {}
    caption_to_images = {}

    test_cids = np.load(test_ids_path)

    print('Loading original COCO ...')
    orig_image_to_captions, orig_caption_to_images = read_coco_data(
        coco_annotation_path, test_cids
    )
    image_to_captions['original'] = orig_image_to_captions
    caption_to_images['original'] = orig_caption_to_images
    all_iids = set(orig_image_to_captions.keys())
    all_cids = set(orig_caption_to_images.keys())

    if cxc_path:
        print('Loading CxC scores ...')
        cxc_scores = read_cxc_data(cxc_path)
        cxc_image_to_captions, cxc_caption_to_images = \
            convert_cxc_scores_to_matched_pairs(
                cxc_scores, cxc_thres, all_iids, all_cids, dump_negative,
            )
        image_to_captions['cxc'] = cxc_image_to_captions
        caption_to_images['cxc'] = cxc_caption_to_images

    if coco_instance_annotation_path:
        print('Loading PM scores ...')
        pm_image_to_captions, pm_caption_to_images = \
            get_plausble_matching_data(
                coco_instance_annotation_path,
                orig_image_to_captions, all_iids, pm_thres, omit_orig=True
            )
        image_to_captions['pm'] = pm_image_to_captions
        caption_to_images['pm'] = pm_caption_to_images

    if human_verified_annotation_path:
        print('Loading human-verified pairs ...')
        hv_image_to_captions, hv_caption_to_images = \
            get_human_verified_data(human_verified_annotation_path,
                                    strict_pos=strict_pos,
                                    dump_negative=dump_negative)
        image_to_captions['human_verified'] = hv_image_to_captions
        caption_to_images['human_verified'] = hv_caption_to_images

    print('Dumping to local ...')
    for datakey, i2t in image_to_captions.items():
        if dump_negative:
            if datakey == 'original':
                continue
            fname = os.path.join(output_dir, f'{datakey}_negative_image_to_caption.json')
        else:
            fname = os.path.join(output_dir, f'{datakey}_image_to_caption.json')
        print(f'Dumping to {fname} ...')
        with open(fname, 'w') as fout:
            _i2t = {str(k): list(v) for k, v in i2t.items()}
            fout.write(json.dumps(_i2t))

    for datakey, t2i in caption_to_images.items():
        if dump_negative:
            if datakey == 'original':
                continue
            fname = os.path.join(output_dir, f'{datakey}_negative_caption_to_image.json')
        else:
            fname = os.path.join(output_dir, f'{datakey}_caption_to_image.json')
        print(f'Dumping to {fname} ...')
        with open(fname, 'w') as fout:
            _t2i = {str(k): [int(_v) for _v in v] for k, v in t2i.items()}
            fout.write(json.dumps(_t2i))

    print('Done.')


if __name__ == '__main__':
    fire.Fire(merge)
