"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import csv


def get_flag(ann, strict_pos, dump_negative):
    posflag = ann == 1 if strict_pos else ann in {1, 2}
    flag = posflag if not dump_negative else not posflag
    return flag


def get_human_verified_data(human_verified_annotation_path,
                            strict_pos=False,
                            dump_negative=False):
    image_to_captions = {}
    caption_to_images = {}

    with open(human_verified_annotation_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # model,ranking,score,cid,iid,annotation,shortcoming,shortcoming2
        for idx, row in enumerate(reader):
            try:
                iid = int(row['iid'].strip())
                cid = int(row['cid'].strip())
            except ValueError:
                print(row)
                continue

            flag = get_flag(int(row['annotation'].strip()), strict_pos, dump_negative)

            if flag:
                caption_to_images.setdefault(cid, set()).add(iid)
                image_to_captions.setdefault(iid, set()).add(cid)

    return image_to_captions, caption_to_images
