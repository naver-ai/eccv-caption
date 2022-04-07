"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import os
import fire
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

from eccv_caption import Metrics


def run(
    coco_img_root='/path/to/coco/images/trainval35k',
    coco_ann_path='/path/to/captions_val2014.json',
):
    # Prepare metric
    metric = Metrics()

    # Prepare the inputs
    coco = COCO(coco_ann_path)
    test_cids = metric.coco_ids

    # Load the model
    device = "cuda"
    model, preprocess = clip.load('ViT-L/14', device)

    all_image_features = []
    all_text_features = []
    all_iids, all_cids = [], []
    seen_iids = set()
    with torch.no_grad():
        for cid in tqdm(test_cids):
            iid = int(coco.anns[cid]['image_id'])
            if iid not in seen_iids:
                path = coco.imgs[iid]['file_name']
                image = Image.open(os.path.join(coco_img_root, path)).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)

                # Calculate features
                image_features = model.encode_image(image_input)
                all_image_features.append(image_features[0])
                all_iids.append(iid)
                seen_iids.add(iid)

            text_inputs = clip.tokenize(coco.anns[cid]['caption']).to(device)

            # Calculate features
            text_features = model.encode_text(text_inputs)
            all_text_features.append(text_features[0])
            all_cids.append(int(cid))

    image_features = torch.stack(all_image_features, dim=0)
    text_features = torch.stack(all_text_features, dim=0)
    print(image_features.shape, text_features.shape)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    sims = image_features @ text_features.T

    i2t = {}
    t2i = {}

    all_cids = np.array(all_cids)
    all_iids = np.array(all_iids)

    # 50 is enough for ECCV metrics (max ECCV t2i/i2t positives = 19/48)
    # In the main paper, we use the modified PMRP by using K = 50.
    # If you want to use the original PMRP, then K should be larger than 13380
    # (max PM t2i/i2t positives = 2676/13380)

    K = 50
    for idx, iid in enumerate(all_iids):
        values, indices = sims[idx, :].topk(K)
        indices = indices.detach().cpu().numpy()
        i2t[iid] = [int(cid) for cid in all_cids[indices]]

    for idx, cid in enumerate(all_cids):
        values, indices = sims[:, idx].topk(K)
        indices = indices.detach().cpu().numpy()
        t2i[cid] = [int(iid) for iid in all_iids[indices]]

    scores = metric.compute_all_metrics(
        i2t, t2i,
        target_metrics=('eccv_r1', 'eccv_map_at_r', 'eccv_rprecision',
                        'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls'),
        Ks=(1, 5, 10),
        verbose=False
    )
    print(scores)
    # It will return the following scores
    # {
    #  'coco_1k_r1': {'i2t': 0.7425999999999999, 't2i': 0.5544},
    #  'coco_1k_r5': {'i2t': 0.9282, 't2i': 0.82324},
    #  'coco_1k_r10': {'i2t': 0.9658, 't2i': 0.90152},
    #  'coco_5k_r1': {'i2t': 0.5636, 't2i': 0.3658},
    #  'coco_5k_r5': {'i2t': 0.7928, 't2i': 0.61048},
    #  'coco_5k_r10': {'i2t': 0.867, 't2i': 0.71148},
    #  'cxc_r1': {'i2t': 0.5804, 't2i': 0.38314912702226495},
    #  'cxc_r5': {'i2t': 0.8142, 't2i': 0.6385151369533878},
    #  'cxc_r10': {'i2t': 0.8862, 't2i': 0.7425116130065673},
    #  'eccv_map_at_r': {'i2t': 0.23988490345859761, 't2i': 0.3195708452398554},
    #  'eccv_rprecision': {'i2t': 0.3381634023972558, 't2i': 0.4177097641300428},
    #  'eccv_r1': {'i2t': 0.7129262490087233, 't2i': 0.7297297297297297}
    # }


if __name__ == '__main__':
    fire.Fire(run)
