"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
try:
    import ujson as json
except ImportError:
    import warnings
    warnings.warn('failed to import ujson. ``pip install ujson`` will make the script faster.')
    import json


def read_coco_data(coco_annotation_path, test_cids):
    with open(coco_annotation_path) as fin:
        # coco_ann.keys() => dict_keys(['info', 'images', 'licenses', 'annotations'])
        # coco_ann['images'] and coco_ann['annotations'] => list of items
        coco_ann = json.load(fin)
    test_cids = set(list(test_cids))

    image_to_captions = {}
    caption_to_images = {}
    n, m = 0, 0
    for _ann in coco_ann['annotations']:
        # _ann.keys() => dict_keys(['image_id', 'id', 'caption'])
        _iid = int(_ann['image_id'])
        _cid = int(_ann['id'])
        n += 1
        if _cid not in test_cids:
            m += 1
            continue
        image_to_captions.setdefault(_iid, set()).add(_cid)
        caption_to_images.setdefault(_cid, set()).add(_iid)
    print(f'[COCO] {m} / {n} are not target annotations')
    return image_to_captions, caption_to_images
