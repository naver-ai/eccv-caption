"""
Original code
https://github.com/google-research-datasets/Crisscrossed-Captions/blob/master/setup.py
"""
import collections
import glob
import csv


def read_cxc_data(cxc_input):
    cxc_scores = collections.defaultdict(list)
    for input_file in glob.glob(cxc_input):
        cxc_scores = _read_cxc_data(input_file, cxc_scores)
    return cxc_scores


def _read_cxc_data(cxc_path, cxc_scores):
    """Read CxC annotations from CSV file.

    Args:
        cxc_path: Path to the CSV file containing CxC scores.
        cxc_scores: Dict of CxC scores mapping caption_id->[(image_id, score,
        rating_type),...] and image_id->[(caption_id, score, rating_type),...].

    Returns:
        Updated list of cxc_scores.
    """
    reader = csv.reader(open(cxc_path, "r"), delimiter=",")
    next(reader)  # Skip header.
    for row in reader:
        # caption,image,agg_score,sampling_method
        # COCO_val2014:sentid:732091,COCO_val2014_000000365325.jpg,2.2,c2i_intrasim

        caption, image_id, score, rating_type = row
        # image_id => COCO_val2014_000000365325.jpg
        image_id = int(image_id.split('.')[0].split('_')[-1])
        sent_id = int(caption.split(":")[-1])
        cxc_scores[sent_id].append((image_id, score, rating_type))
        '''
        # cxc_scores[image_id].append((sent_id, score, rating_type))
        # If the image and caption correspond to the same example, do not append
        # CxC rating twice for the same example.
        if rating_type != "c2i_original":
            cxc_scores[sent_id].append((image_id, score, rating_type))
        '''
    return cxc_scores


def read_cxc_caption_data(cxc_input):
    cxc_scores = collections.defaultdict(list)
    for input_file in glob.glob(cxc_input):
        cxc_scores = _read_cxc_caption_data(input_file, cxc_scores)
    return cxc_scores


def _read_cxc_caption_data(cxc_path, cxc_scores):
    """Read CxC annotations from CSV file.

    Args:
        cxc_path: Path to the CSV file containing CxC scores.
        cxc_scores: Dict of CxC scores mapping caption_id->[(image_id, score,
        rating_type),...] and image_id->[(caption_id, score, rating_type),...].

    Returns:
        Updated list of cxc_scores.
    """
    reader = csv.reader(open(cxc_path, "r"), delimiter=",")
    next(reader)  # Skip header.
    for row in reader:
        # caption1,caption2,agg_score,sampling_method
        # COCO_val2014:sentid:190268,COCO_val2014:sentid:315773,1.24,c2c_isim

        caption1, caption2, score, rating_type = row
        cid1 = int(caption1.split(":")[-1])
        cid2 = int(caption2.split(":")[-1])
        key = '_'.join([str(cid) for cid in sorted([cid1, cid2])])
        cxc_scores[key] = float(score)
    return cxc_scores
