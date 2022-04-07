# Extended COCO Validation (ECCV) Caption dataset

Official Python implementation of ECCV Caption | [Paper](https://arxiv.org/abs/2204.03359)

[Sanghyuk Chun](https://sanghyukchun.github.io/home/), [Wonjae Kim](https://wonjae.kim/), [Song Park](https://8uos.github.io/), [Minsuk Chang](https://minsukchang.com/), [Seong Joon Oh](https://coallaoh.github.io/)

[NAVER AI Lab](https://naver-career.gitbook.io/en/teams/clova-cic)

ECCV Caption contains x8.47 positive images and x3.58 positive captions compared to the original COCO Caption. The positives are verified by machines (five state-of-the-art image-text matching models) and humans. This library provides an unified interface to measure various COCO Caption retrieval metrics, such as COCO 1k Recall@K, COCO 5k Recall@K, [CxC](https://github.com/google-research-datasets/Crisscrossed-Captions) Recall@K, [PMRP](https://github.com/naver-ai/pcme), and ECCV Caption Recall@K, R-Precision and mAP@R.

For more details, please read our paper:

[ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO](https://arxiv.org/abs/2204.03359)

### Abstract

Image-Test matching (ITM) is a common task for evaluating the quality of Vision and Language (VL) models. However, existing ITM benchmarks have a significant limitation. They have many missing correspondences, originating from the data construction process itself. For example, a caption is only matched with one image although the caption can be matched with other similar images, and vice versa. To correct the massive false negatives, we construct the Extended COCO Validation (ECCV) Caption dataset by supplying the missing associations with machine and human annotators. We employ five state-of-the-art ITM models with diverse properties for our annotation process. Our dataset provides x3.6 positive image-to-caption associations and x8.5 caption-to-image associations compared to the original MS-COCO. We also propose to use an informative ranking-based metric, rather than the popular Recall@K(R@K). We re-evaluate the existing 25 VL models on existing and proposed benchmarks. Our findings are that the existing benchmarks, such as COCO 1K R@K, COCO 5K R@K, CxC R@1 are highly correlated with each other, while the rankings change when we shift to the ECCV mAP. Lastly, we delve into the effect of the bias introduced by the choice of machine annotator. Source code and dataset are available in [https://github.com/naver-ai/eccv-caption](https://github.com/naver-ai/eccv-caption)


### Dataset statistics

ECCV Caption dataset is an extended version of the COCO Caption test split by [karpathy/neuraltalk2](https://github.com/karpathy/neuraltalk2). We annotate positives the subset of the COCO Caption test set (1,332 query images, 1,261 query captions). We show the number of positive items for the subset of the COCO Caption test split.

| Dataset                  | # positive images | # positive captions |
|--------------------------|-------------------|---------------------|
| Original MS-COCO Caption | 1,332             | 6,305 (=1,261×5)   |
| [CxC](https://github.com/google-research-datasets/Crisscrossed-Captions)                    | 1,895 (×1.42) | 8,906 (×1.41) |
| Human-verified positives | 10,814 (×8.12)   | 16,990 (×2.69)     |
| ECCV Caption             | 11,279 (×8.47)   | 22,550 (×3.58)     |

## Updates

- 8 Apr, 2022: Initial upload.

## Getting Started

### Installation

```
pip3 install eccv_caption
```

### Requirements

```
numpy
ujson
tqdm
```

`ujson` and `tqdm` are not neccessary.

### Usage

```python
from eccv_caption import Metrics

metric = Metrics()

# Get i2t, t2i retrived items from your own model
# i2t = {query_image_id: [sorted_caption_id_by_similarity]}
# t2i = {query_caption_id: [sorted_image_id_by_similarity]}
# See the example code for how to compute i2t / t2i from your model

scores = metric.compute_all_metrics(
	i2t, t2i,
	target_metrics=('eccv_r1', 'eccv_map_at_r', 'eccv_rprecision',
                   'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls'),
	Ks=(1, 5, 10),
	verbose=False
)
print(scores)
```

It will return a score map, for example:
```
{
    'coco_1k_r1': {'i2t': 0.7425999999999999, 't2i': 0.5544},
    'coco_1k_r5': {'i2t': 0.9282, 't2i': 0.82324},
    'coco_1k_r10': {'i2t': 0.9658, 't2i': 0.90152},
    'coco_5k_r1': {'i2t': 0.5636, 't2i': 0.3658},
    'coco_5k_r5': {'i2t': 0.7928, 't2i': 0.61048},
    'coco_5k_r10': {'i2t': 0.867, 't2i': 0.71148},
    'cxc_r1': {'i2t': 0.5804, 't2i': 0.38314912702226495},
    'cxc_r5': {'i2t': 0.8142, 't2i': 0.6385151369533878},
    'cxc_r10': {'i2t': 0.8862, 't2i': 0.7425116130065673},
    'eccv_map_at_r': {'i2t': 0.23988490345859761, 't2i': 0.3195708452398554},
    'eccv_rprecision': {'i2t': 0.3381634023972558, 't2i': 0.4177097641300428},
    'eccv_r1': {'i2t': 0.7129262490087233, 't2i': 0.7297297297297297}
}
```

`cxc` denotes CrissCrossed Captions. You can find the details of the CxC dataset from [google-research-datasets/Crisscrossed-Captions](https://github.com/google-research-datasets/Crisscrossed-Captions)

You can find an example code from [here](./examples/clip_example_run.py)

### Arguments

- `eccv_caption.Metrics`
	- `extra_file_dir`: str (default: `None`): directory where `pm_image_to_caption.json` and `pm_caption_to_image.json` are downloaded. You don't need to specify `extra_file_dir` if you are not going to measure `pmrp`.
	- `verbose`: bool (default: False), `tqdm` verbose option

All metrics can be computed by the following method:

- `eccv_caption.Metrics.compute_all_metrics`

The details of the method arguments are as the follows:

```python
def compute_all_metrics(self, i2t_retrieved_items: Dict[int, List[int]] = None,
                        t2i_retrieved_items: Dict[int, List[int]] = None,
                        target_metrics: List[str] = ('coco_1k_r1', 'coco_5k_r1', 'cxc_r1', 'eccv_r1', 'eccv_map_at_r'),
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
```

### Download the full data

You can find the full data from the following gdrive link, including `pm_image_to_caption.json` and `pm_caption_to_image.json` for computing `pmrp`.

https://drive.google.com/drive/folders/1Sam8_Hpm4uWKB_Ehk9C_JcGNpYH7jZD2?usp=sharing

The gdrive link also includes the full data, such as the raw human annotated data (`mturk_parsed.csv`) and filtered items (`wrong_captions.txt`, `invalid_images.txt`, `invalid_captions.txt`, `verified_iids.npy`, `verified_cids.npy`). We describe the details of how the dataset is built in the next section.

## Dataset construction

Due to the nature of the dataset annotation process, widely-used Image-Text aligned datasets, such as MS-COCO, have many false negatives. We extend the MS-COCO Caption `test` split by using machine and human annotators.

Our annotation is built upon five state-of-the-art image-text matching models, such as [VSRN (ICCV'19)](https://arxiv.org/abs/1909.02701), [PVSE (CVPR'19)](https://arxiv.org/abs/1906.04402), [ViLT (ICML'21)](https://arxiv.org/abs/2102.03334), [CLIP (ICML'21)](https://arxiv.org/abs/2103.00020), [PCME (CVPR'21)](https://arxiv.org/abs/2101.05068).

We first collect top-25 retrived items for the subset of MS-COCO Caption by the five SOTA models. We verify whether the top-25 items are positive or negative by Amazon Mechanical Turk (MTurk). The full MTurk results can be found in [gdrive](https://drive.google.com/drive/folders/1Sam8_Hpm4uWKB_Ehk9C_JcGNpYH7jZD2?usp=sharing) `mturk_parsed.csv`.

We measure precision and recall of the existing benchmarks. Precision (Prec) and Recall of previous datasets measured by human verified positive pairs are shown in the followin table. A low Prec means that many positives are actually negatives, and a low Recall means that there exist many missing positives. The table shows that the existing benchmarks, such as MS-COCO Caption have many missing positive correspondences.

| Dataset                  | I2T Prec | I2T Recall | T2I Prec | T2I Recall |
|--------------------------|----------|------------|----------|------------|
| Original MS-COCO Caption | 47.3     | 20.0       | 89.4     | 12.8       |
| [CxC](https://github.com/google-research-datasets/Crisscrossed-Captions) | 39.6     | 22.0       | 81.4     | 15.0       |
| [Plausible Match](https://github.com/naver-ai/pcme) | 8.3      | 74.6       | 10.5     | 69.0       |

We combine our MTurk results and CxC results to build the final ECCV Caption dataset (by treating CxC results as the sixth machine annotator).

For more details, please read our [paper](https://arxiv.org/abs/2204.03359).

## License

```
Copyright (c) 2022-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## How to cite

```
@article{chun2022eccv_caption,
    title={ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO}, 
    author={Chun, Sanghyuk and Kim, Wonjae and Park, Song and Chang, Minsuk Chang and Oh, Seong Joon},
    journal={arXiv preprint arXiv:2204.03359},
}
```
