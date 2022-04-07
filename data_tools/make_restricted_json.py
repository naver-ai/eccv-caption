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
from collections import Counter


def counter(keyval):
    return Counter([len(v) for k, v in keyval.items()])


def filter_annotation(valid_iids_path,
                      valid_cids_path,
                      human_verified_i2t_json,
                      human_verified_t2i_json,
                      invalid_images_path,
                      invalid_captions_path,
                      wrong_captions_path,
                      cxc_i2t_json,
                      cxc_t2i_json,
                      original_t2i_json,
                      new_json_dump_to
                      ):
    iids = set(np.load(valid_iids_path))
    cids = set(np.load(valid_cids_path))

    if invalid_images_path:
        with open(invalid_images_path) as fin:
            invalid_iids = set([int(line.strip()) for line in fin.readlines()])
        iids = iids - invalid_iids

    if invalid_captions_path:
        with open(invalid_captions_path) as fin:
            invalid_cids = set([int(line.strip()) for line in fin.readlines()])
        cids = cids - invalid_cids

    if wrong_captions_path:
        with open(invalid_captions_path) as fin:
            wrong_cids = set([line.strip() for line in fin.readlines()])
        with open(original_t2i_json) as fin:
            orig_t2i = json.load(fin)

        wrong_pairs = set()
        for cid in wrong_cids:
            if len(orig_t2i[cid]) != 1:
                raise ValueError(orig_t2i[cid])
            wrong_pairs.add(f'{orig_t2i[cid][0]}_{cid}')

    with open(human_verified_i2t_json) as fin:
        i2t = json.load(fin)
    with open(human_verified_t2i_json) as fin:
        t2i = json.load(fin)

    i2t = {k: v for k, v in i2t.items() if int(k) in iids}
    t2i = {k: v for k, v in t2i.items() if int(k) in cids}

    print('I2T', len(i2t), sorted(counter(i2t)), counter(i2t), sum([len(v) for v in i2t.values()]))
    print('T2I', len(t2i), sorted(counter(t2i)), counter(t2i), sum([len(v) for v in t2i.values()]))

    with open(cxc_i2t_json) as fin:
        cxc_i2t = json.load(fin)
    with open(cxc_t2i_json) as fin:
        cxc_t2i = json.load(fin)

    new_i2t = {}
    n, m, s = 0, 0, 0
    for iid, _cids in i2t.items():
        n += 1
        if iid not in cxc_i2t:
            m += 1
            continue
        new_cids = set(_cids).union(set(cxc_i2t[iid]))
        filtered_cids = new_cids - invalid_cids
        _filtered_cids = []
        for _cid in filtered_cids:
            if f'{iid}_{_cid}' not in wrong_pairs:
                _filtered_cids.append(_cid)
        filtered_cids = _filtered_cids
        new_i2t[iid] = list(filtered_cids)
        s += len(new_cids) - len(filtered_cids)
    i2t = new_i2t
    print(f'skip {m}/{n}, filtered {s}')

    new_t2i = {}
    n, m, s = 0, 0, 0
    for cid, _iids in t2i.items():
        n += 1
        if cid not in cxc_t2i:
            m += 1
            continue
        new_iids = set(_iids).union(set(cxc_t2i[cid]))
        filtered_iids = new_iids - invalid_iids
        _filtered_iids = []
        for _iid in filtered_iids:
            if f'{_iid}_{cid}' not in wrong_pairs:
                _filtered_iids.append(_iid)
        filtered_iids = _filtered_iids
        new_t2i[cid] = list(filtered_iids)
        s += len(new_iids) - len(filtered_iids)
    t2i = new_t2i
    print(f'skip {m}/{n}, filtered {s}')

    print('I2T', len(i2t), sorted(counter(i2t)), counter(i2t), sum([len(v) for v in i2t.values()]))
    c = counter(i2t)
    for k in sorted(c):
        print(k, c[k])
    print('T2I', len(t2i), sorted(counter(t2i)), counter(t2i), sum([len(v) for v in t2i.values()]))
    c = counter(t2i)
    for k in sorted(c):
        print(k, c[k])

    with open(os.path.join(new_json_dump_to, 'verified_image_to_caption.json'), 'w') as fout:
        json.dump(i2t, fout)

    with open(os.path.join(new_json_dump_to, 'verified_caption_to_image.json'), 'w') as fout:
        json.dump(t2i, fout)


if __name__ == '__main__':
    fire.Fire(filter_annotation)
