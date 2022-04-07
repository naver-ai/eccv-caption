"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import os
import warnings

import numpy as np

from typing import List, Dict, Set, Callable

from ._metrics import compute_coco1k_r_at_k
from ._metrics import compute_eccv_metrics
from ._metrics import compute_rprecision
from ._metrics import compute_r_at_k

try:
    import ujson as json
except ImportError:
    warnings.warn('failed to import `ujson`. use `json` instead.')
    import json


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class Metrics():
    def __init__(self, extra_file_dir: str = None,
                 verbose: bool = False):
        self.set_coco_gts(
            os.path.join(CUR_DIR, 'data/original_image_to_caption.json'),
            os.path.join(CUR_DIR, 'data/original_caption_to_image.json'),
        )

        self.set_cxc_gts(
            os.path.join(CUR_DIR, 'data/cxc_image_to_caption.json'),
            os.path.join(CUR_DIR, 'data/cxc_caption_to_image.json'),
        )

        self.set_eccv_gts(
            os.path.join(CUR_DIR, 'data/eccv_image_to_caption.json'),
            os.path.join(CUR_DIR, 'data/eccv_caption_to_image.json'),
        )

        if extra_file_dir:
            self.set_pm_gts(
                os.path.join(extra_file_dir, 'pm_image_to_caption.json'),
                os.path.join(extra_file_dir, 'pm_caption_to_image.json'),
            )
        else:
            self.pm_gts = {}

        self.coco_ids = np.load(os.path.join(CUR_DIR, 'data/coco_test_ids.npy'))
        self.verbose = verbose

    def set_i2t_retrieved_items(self, i2t_retrieved_items: Dict[int, List[int]]):
        """ Set default i2t_retrieved_items """
        if not isinstance(i2t_retrieved_items, dict):
            raise TypeError(f'`i2t_retrieved_items` should be `Dict`, not {type(i2t_retrieved_items)}')
        self.i2t_retrived_items = i2t_retrieved_items

    def set_t2i_retrieved_items(self, t2i_retrieved_items: Dict[int, List[int]]):
        """ Set default t2i_retrieved_items """
        if not isinstance(t2i_retrieved_items, dict):
            raise TypeError(f'`t2i_retrieved_items` should be `Dict`, not {type(t2i_retrieved_items)}')
        self.t2i_retrived_items = t2i_retrieved_items

    def __parse_json(self, i2t_json_path: str,
                     t2i_json_path: str) -> Dict[str, Dict[int, List[int]]]:
        """ Parse json dumped GT files """
        with open(i2t_json_path) as fin:
            i2t = json.load(fin)
            i2t = {int(k): [int(_v) for _v in v] for k, v in i2t.items()}
        with open(t2i_json_path) as fin:
            t2i = json.load(fin)
            t2i = {int(k): [int(_v) for _v in v] for k, v in t2i.items()}
        return {'i2t': i2t, 't2i': t2i}

    def set_coco_gts(self, coco_i2t_json_path: str, coco_t2i_json_path: str):
        """ Set COCO GTs from the local json files """
        self.coco_gts = self.__parse_json(coco_i2t_json_path, coco_t2i_json_path)

    def set_cxc_gts(self, cxc_i2t_json_path: str, cxc_t2i_json_path: str):
        """ Set CXC GTs from the local json files """
        self.cxc_gts = self.__parse_json(cxc_i2t_json_path, cxc_t2i_json_path)

    def set_eccv_gts(self, eccv_i2t_json_path: str, eccv_t2i_json_path: str):
        """ Set ECCV Caption GTs from the local json files """
        self.eccv_gts = self.__parse_json(eccv_i2t_json_path, eccv_t2i_json_path)

    def set_pm_gts(self, pm_i2t_json_path: str, pm_t2i_json_path: str):
        """ Set Plausible Match GTs from the local json files
        Note: You have to download PM GTs (> 140MB) first.
        """
        self.pm_gts = self.__parse_json(pm_i2t_json_path, pm_t2i_json_path)

    def __check_arguments(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                          modality: str):
        """ Check whether the given arguments are valid or not """
        if not isinstance(retrieved_items, dict):
            raise TypeError(f'`retrieved_items` should be `Dict`, not {type(retrieved_items)}')
        if modality not in {'i2t', 't2i', 'all'}:
            raise ValueError(f'`modality` should be in ("i2t", "t2i", "all"), not {modality}')
        if modality == 'all':
            if 'i2t' not in retrieved_items or 't2i' not in retrieved_items:
                raise ValueError(f'If `modality` == "all", `retrieved_items` should have both "i2t" and "t2i" as key, but only have ({retrieved_items.keys()})')
        else:
            if modality not in retrieved_items:
                raise ValueError(f'`modality` {modality} is given but, `retrieved_items` does not have {modality}, but ({retrieved_items.keys()})')

    def __compute_metric(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                         gts: Dict[str, Dict[int, List[int]]],
                         metric_fn: Callable[..., float],
                         modality: str,
                         **kwargs) -> Dict[str, float]:
        """ Compute i2t and t2i metrics for the given retrived_items, gts and metric_fn.

        Parameters
        ----------
        retrived_items: dict (key: item id (int), value: retrived item ids sorted by similarity),
        gts: dict (key: item id (int), value: ground truth item ids),
        metric_fn: function, the target metric function to compute
        modality: str, should be in {'all', 'i2t', 't2i'}

        Returns
        -------
        scores (dict): key: the name of metrics, value: {'i2t': score (float), 't2i': score (float)}
        """
        if modality not in {'i2t', 't2i', 'all'}:
            raise ValueError(f'`modality` should be in ("i2t", "t2i", "all"), not {modality}')
        if modality == 'all':
            score = {}
            for _modality in ['i2t', 't2i']:
                _metric = metric_fn(retrieved_items[_modality],
                                    gts[_modality],
                                    verbose=self.verbose,
                                    **kwargs)
                score[_modality] = _metric
        else:
            _metric = metric_fn(retrieved_items[modality],
                                gts[modality],
                                verbose=self.verbose,
                                **kwargs)
            score = {modality: _metric}
        return score

    def coco_1k_recalls(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                        modality: str,
                        K: int = 1) -> Dict[str, float]:
        """ Compute COCO 1K (5 fold validation of COCO 5K) Recalls """
        self.__check_arguments(retrieved_items, modality)
        N = len(self.coco_ids) // 5

        i2t_gt_1k_recalls = []
        t2i_gt_1k_recalls = []

        for idx in range(5):
            nfold_cids = self.coco_ids[idx * N: (idx + 1) * N]
            nfold_coco_t2i = {cid: self.coco_gts['t2i'][cid] for cid in nfold_cids}

            nfold_iids = set([iid[0] for iid in nfold_coco_t2i.values()])
            nfold_coco_i2t = {iid: self.coco_gts['i2t'][iid] for iid in nfold_iids}

            nfold_cids = set([int(_id) for _id in nfold_cids])
            nfold_iids = set([int(_id) for _id in nfold_iids])

            i2t_gt_1k_recalls.append(compute_coco1k_r_at_k(retrieved_items['i2t'], nfold_coco_i2t, nfold_cids, K, self.verbose))
            t2i_gt_1k_recalls.append(compute_coco1k_r_at_k(retrieved_items['t2i'], nfold_coco_t2i, nfold_iids, K, self.verbose))
        i2t_gt_1k_recall = np.mean(i2t_gt_1k_recalls)
        t2i_gt_1k_recall = np.mean(t2i_gt_1k_recalls)

        score = {'i2t': i2t_gt_1k_recall, 't2i': t2i_gt_1k_recall}
        if modality != 'all':
            score = score[modality]
        return score

    def coco_5k_recalls(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                        modality: str,
                        K: int = 1) -> Dict[str, float]:
        """ Compute COCO 5K Recalls """
        self.__check_arguments(retrieved_items, modality)
        return self.__compute_metric(retrieved_items, self.coco_gts,
                                     compute_r_at_k,
                                     modality, K=K)

    def cxc_recalls(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                    modality: str,
                    K: int = 1) -> Dict[str, float]:
        """ Compute CxC Recalls """
        self.__check_arguments(retrieved_items, modality)
        return self.__compute_metric(retrieved_items, self.cxc_gts,
                                     compute_r_at_k,
                                     modality, K=K)

    def pmrp(self, retrieved_items: Dict[str, Dict[int, List[int]]],
             modality: str) -> Dict[str, float]:
        """ Compute Plausible Match R-Precision (PMRP) """
        self.__check_arguments(retrieved_items, modality)
        if not self.pm_gts:
            warnings.warn('PM GTs are not found. Please follow https://github.com/naver-ai/eccv-caption to download PM GTs.')
            return
        return self.__compute_metric(retrieved_items, self.pm_gts,
                                     compute_rprecision,
                                     modality)

    def eccv_recalls(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                     modality: str,
                     K: int = 1) -> Dict[str, float]:
        """ Compute ECCV Caption Recalls """
        self.__check_arguments(retrieved_items, modality)
        return self.__compute_metric(retrieved_items, self.eccv_gts,
                                     compute_r_at_k,
                                     modality, K=K)

    def eccv_metrics(self, retrieved_items: Dict[str, Dict[int, List[int]]],
                     modality: str) -> Dict[str, Dict[str, float]]:
        """ Compute ECCV Caption metrics including R1, mAP@R, R-Precision """
        self.__check_arguments(retrieved_items, modality)
        _metrics = self.__compute_metric(retrieved_items, self.eccv_gts,
                                         compute_eccv_metrics,
                                         modality)
        metrics = {}
        for modality, _modality_metric in _metrics.items():
            for metric_key, metric_val in _modality_metric.items():
                _metric_dict = metrics.setdefault(metric_key, {})
                _metric_dict[modality] = metric_val
        return metrics

    def __update_recalls(self, metrics: Dict[str, Dict[str, float]],
                         target_metrics: Set[str],
                         retrieved_items: Dict[str, Dict[int, List[int]]],
                         dataset_name: str,
                         Ks: List[int]):
        recall_fn = getattr(self, f'{dataset_name}_recalls')
        if f'{dataset_name}_recalls' in target_metrics:
            for K in Ks:
                _scores = recall_fn(retrieved_items, 'all', K=K)
                metrics.update({f'{dataset_name}_r{K}': _scores})
        elif f'{dataset_name}_r1' in target_metrics:
            _scores = recall_fn(retrieved_items, 'all', K=1)
            metrics.update({f'{dataset_name}_r1': _scores})

    def compute_all_metrics(self, i2t_retrieved_items: Dict[int, List[int]] = None,
                            t2i_retrieved_items: Dict[int, List[int]] = None,
                            target_metrics: List[str] = ('coco_1k_r1', 'coco_5k_r1',
                                                         'cxc_r1', 'eccv_r1', 'eccv_map_at_r'),
                            Ks: List[int] = (1, 5, 10),
                            verbose: bool = None):
        """ Compute various evaluation metrics for the given image-to-text & text-to-image retrieved items.

        Parameters
        ----------
        i2t_retrived_items: dict (key: image id (int), value: retrived caption ids sorted by similarity),
            (default: `self.i2t_retrived_items` set by `self.set_i2t_retrieved_items`)
            dictionary of target retrived caption ids to evaluate.
        t2i_retrived_items: dict (key: caption id (int), value: retrived image ids sorted by similarity),
            (default: `self.t2i_retrived_items` set by `self.set_t2i_retrieved_items`)
            dictionary of target retrived image ids to evaluate.
        target_metrics: list, the list of target metrics.
            (default: ['coco_1k_r1', 'coco_5k_r1', 'cxc_r1', 'eccv_r1', 'eccv_map_at_r'])
            Valid metrics: {'coco_1k_r1', 'coco_5k_r1', 'cxc_r1',
                            'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls',
                            'eccv_r1', 'eccv_rprecision', 'eccv_map_at_r',
                            'eccv_recalls', 'pmrp'}
            Note1: `{datasetname}_recalls` returns R@K, where K is given by `Ks`.
            Note2: to use pmrp, you have to download PM GTs first. Please refer the official document for the details.
        Ks: list, the list of target Ks for R@K.
            (default: [1, 5, 10])
            `target_metrics` should contain {datasetname}_recalls
        verbose: bool (default: self.verbose), tqdm verbose option

        Returns
        -------
        scores (dict): key: the name of metrics, value: {'i2t': score (float), 't2i': score (float)}
        """
        if i2t_retrieved_items and t2i_retrieved_items:
            self.set_i2t_retrieved_items(i2t_retrieved_items)
            self.set_t2i_retrieved_items(t2i_retrieved_items)
        elif i2t_retrieved_items or t2i_retrieved_items:
            raise ValueError('Both `i2t_retrieved_items` and `t2i_retrieved_items` should be None or Dict, but found I2T({type(i2t_retrieved_items)}) T2I({type(t2i_retrieved_items)})')
        else:
            if not hasattr(self, 'i2t_retrived_items') or not hasattr(self, 'i2t_retrived_items'):
                raise TypeError('`i2t_retrived_items` and `t2i_retrived_items` are not given. Please set retrieved items before call `compute_all_metrics`, or pass retrived_items as arguments.')

        if verbose is not None:
            self.verbose = verbose

        retrieved_items = {}
        retrieved_items['i2t'] = self.i2t_retrived_items
        retrieved_items['t2i'] = self.t2i_retrived_items

        metrics = {}
        target_metrics = set(target_metrics)

        self.__update_recalls(metrics, target_metrics,
                              retrieved_items, 'coco_1k', Ks)
        self.__update_recalls(metrics, target_metrics,
                              retrieved_items, 'coco_5k', Ks)
        self.__update_recalls(metrics, target_metrics,
                              retrieved_items, 'cxc', Ks)

        if 'pmrp' in target_metrics:
            _scores = self.pmrp(retrieved_items, 'all')
            if _scores:
                metrics.update(_scores)

        if 'eccv_map_at_r' in target_metrics or 'eccv_rpresion' in target_metrics:
            _scores = self.eccv_metrics(retrieved_items, 'all')
            _scores = {k: v for k, v in _scores.items() if k in target_metrics}
            metrics.update(_scores)
        else:
            self.__update_recalls(metrics, target_metrics,
                                  retrieved_items, 'eccv', Ks)

        return metrics
