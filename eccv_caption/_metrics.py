"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    import warnings
    warnings.warn('failed to import `tqdm`. verbose option has been disabled.')

    def tqdm(x, total=None, disable=None):
        return x
from typing import List, Dict


def rprecision(target_items: List[int],
               gt_items: List[int],
               r: int = None) -> float:
    """ compute R-Precision between two ranked lists

    Parameters
    ----------
    target_items: list (element: int, itemid), a ranked list by retrieval similarity
    gt_items: list (element: int, itemid), a list of "ground truth" positive items
    r: int (default: len(gt_items)), the number of target items to measure R-Precision.
        r is manually assigned only for computing mAP@R. Otherwise, set r as a default value.

    Returns
    -------
    precision: float
    """
    if not r:
        r = len(gt_items)

    target_R_items = set(target_items[:r])
    non_precise = target_R_items - gt_items
    precision = 1 - len(non_precise) / len(target_R_items)
    return precision


def recall_at_k(target_items: List[int],
                gt_items: List[int],
                K: int = 1) -> float:
    """ compute R-Precision between two ranked lists

    Parameters
    ----------
    target_items: list (element: int, itemid), a ranked list by retrieval similarity
    gt_items: list (element: int, itemid), a list of "ground truth" positive items
    K: int (default: 1), K for Recall@K

    Returns
    -------
    recall: float (1 if matched, 0 otherwise)
    """
    if K == 1:
        return 1 if target_items[0] in gt_items else 0
    else:
        return 1 if set(target_items[:K]) & gt_items else 0


def compute_rprecision(target_items: Dict[int, List[int]],
                       gt_items: Dict[int, List[int]],
                       verbose: bool = False) -> float:
    """ compute R-Precision between two ranked lists

    Parameters
    ----------
    target_items: dict (key: itemid (int), value: retrived item ids sorted by similarity),
        dictionary of target retrived items to evaluate.
        Key: item id (int), Value: retrived item ids sorted by similarity scores.
    gt_items: dict (key: itemid (int), value: a list of "ground truth" positive items)
    verbose: bool (default: False), tqdm verbose option

    Returns
    -------
    rprecision (float)
    """
    precisions = []
    for _id, _matched_items in tqdm(gt_items.items(), total=len(gt_items), disable=not verbose):
        _target_items = target_items[_id]
        all_matched = set(_matched_items)
        n_matched = len(all_matched)

        _prec = rprecision(_target_items, all_matched, n_matched)
        precisions.append(_prec)

    return np.mean(precisions)


def compute_r_at_k(target_items: Dict[int, List[int]],
                   gt_items: Dict[int, List[int]],
                   K: int = 1,
                   verbose: bool = False) -> float:
    """ compute Recall@K between two ranked lists

    Parameters
    ----------
    target_items: dict (key: itemid (int), value: retrived item ids sorted by similarity),
        dictionary of target retrived items to evaluate.
        Key: item id (int), Value: retrived item ids sorted by similarity scores.
    gt_items: dict (key: itemid (int), value: a list of "ground truth" positive items)
    K: int (default: 1), K for Recall@K
    verbose: bool (default: False), tqdm verbose option

    Returns
    -------
    recall_at_k (float)
    """
    recalls = []
    for _id, _matched_items in tqdm(gt_items.items(), total=len(gt_items), disable=not verbose):
        _gt_items = set(gt_items[int(_id)])
        _target_items = target_items[_id]
        if K == 1:
            _recall = 1 if _target_items[0] in _gt_items else 0
        else:
            _recall = 1 if set(_target_items[:K]) & _gt_items else 0
        recalls.append(_recall)

    return np.mean(recalls)


def compute_coco1k_r_at_k(target_items: Dict[int, List[int]],
                          gt_items: Dict[int, List[int]],
                          nfold_itemids: List[int],
                          K: int = 1,
                          verbose: bool = False) -> float:
    """ compute Recall@K between two ranked lists with restricted item ids

    Parameters
    ----------
    target_items: dict (key: itemid (int), value: retrived item ids sorted by similarity),
        dictionary of target retrived items to evaluate.
        Key: item id (int), Value: retrived item ids sorted by similarity scores.
    gt_items: dict (key: itemid (int), value: a list of "ground truth" positive items)
    nfold_itemids: list, target item ids (a subset of the entire item ids)
    K: int (default: 1), K for Recall@K
    verbose: bool (default: False), tqdm verbose option

    Returns
    -------
    recall_at_k (float)
    """
    gt_recalls = []
    for _id, _matched_items in tqdm(gt_items.items(), total=len(gt_items), disable=not verbose):
        _gt_items = set(gt_items[int(_id)])
        _target_items = [_itemid for _itemid in target_items[_id] if _itemid in nfold_itemids]
        if K == 1:
            _recall = 1 if _target_items[0] in _gt_items else 0
        else:
            _recall = 1 if set(_target_items[:K]) & _gt_items else 0
        gt_recalls.append(_recall)

    return np.mean(gt_recalls)


def compute_eccv_metrics(target_items: Dict[int, List[int]],
                         gt_items: Dict[int, List[int]],
                         verbose: bool = False) -> float:
    """ compute mAP@R, R-Precision, Recall@1 between two ranked lists

    Parameters
    ----------
    target_items: dict (key: itemid (int), value: retrived item ids sorted by similarity),
        dictionary of target retrived items to evaluate.
        Key: item id (int), Value: retrived item ids sorted by similarity scores.
    gt_items: dict (key: itemid (int), value: a list of "ground truth" positive items)
    verbose: bool (default: False), tqdm verbose option

    Returns
    -------
    scores (dict): key = ["map_at_r", "rprecision", "r1"]
    """
    recalls = []
    precisions = []
    maps = []
    for _id, _matched_items in tqdm(gt_items.items(), total=len(gt_items)):
        _target_items = target_items[_id]
        all_matched = set(_matched_items)
        n_matched = len(all_matched)

        # compute recall@1
        _recall = 1 if _target_items[0] in all_matched else 0
        recalls.append(_recall)

        # compute R-Precision and MAP@R
        _precs = []
        for r in range(1, n_matched + 1):
            _prec = rprecision(_target_items, all_matched, r)
            if _target_items[r-1] in all_matched:
                _precs.append(_prec)
            else:
                _precs.append(0)
        precisions.append(_prec)
        maps.append(np.mean(_precs))

    return {
        "eccv_map_at_r": np.mean(maps),
        "eccv_rprecision": np.mean(precisions),
        "eccv_r1": np.mean(recalls),
    }
