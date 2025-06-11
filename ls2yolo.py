#!/usr/bin/env python3
"""
auto_convert_split.py

1) Takes a project directory containing:
     - an interface XML
     - a Label Studio JSON export
     - an images/ folder
2) Converts LS → COCO-Keypoints
3) Splits dataset (~70/30) into train/val
4) Writes:
     project_dir/
       coco_keypoints.json
       yolo_pose_dataset/
         images/{train,val}/...
         labels/{train,val}/...
         pose.yaml
"""

import os
import json
import pathlib
import argparse
import random
import xml.etree.ElementTree as ET
from shutil import copy2
from math import floor
import tqdm


def parse_relations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return [(rel.get("fromName"), rel.get("toName"))
            for rel in root.findall(".//Relations/Relation")]


def convert_ls_to_coco(ls_json, xml_path):
    rules = parse_relations(xml_path)
    tasks = json.load(open(ls_json, 'r', encoding='utf-8'))
    images, annotations = [], []
    cat_map = {'face': 0, 'juice_tube': 1}
    kp_idx = {
        'face': ['left_pupil', 'right_pupil', 'nose_bridge'],
        'juice_tube': ['spout_top', 'spout_bottom', 'dummy']
    }
    ann_id = 0
    skipped_count = 0
    
    for task in tqdm.tqdm(tasks, desc="LS→COCO"):
        img_id = task['id']
        
        # Check if task has annotations
        if not task.get('annotations') or not task['annotations']:
            skipped_count += 1
            continue
            
        res = task['annotations'][0]['result']
        if not res:
            skipped_count += 1
            continue
            
        try:
            w, h = res[0]['original_width'], res[0]['original_height']
        except (IndexError, KeyError) as e:
            print(f"\nError processing task {img_id}:")
            print(f"Result data: {res}")
            raise
            
        fname = pathlib.Path(task['data']['image']).name
        images.append({'id': img_id, 'file_name': fname, 'width': w, 'height': h})
        by_tool = {}
        for r in res:
            by_tool.setdefault(r['from_name'], []).append(r)
        for box_tool, kp_tool in rules:
            for box in by_tool.get(box_tool, []):
                cls = box['value']['rectanglelabels'][0]
                x = box['value']['x'] * w / 100
                y = box['value']['y'] * h / 100
                W = box['value']['width'] * w / 100
                H = box['value']['height'] * h / 100
                trip = []
                for name in kp_idx[cls]:
                    match = next((kp for kp in by_tool.get(kp_tool, [] if None else [])
                                  if kp['value']['keypointlabels'][0] == name), None)
                    if match:
                        trip += [match['value']['x'] * w / 100,
                                 match['value']['y'] * h / 100, 2]
                    else:
                        trip += [0, 0, 0]
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_map[cls],
                    'bbox': [x, y, W, H],
                    'area': W * H,
                    'iscrowd': 0,
                    'num_keypoints': sum(1 for v in trip[2::3] if v > 0),
                    'keypoints': trip
                })
                ann_id += 1
    categories = [
        {'id': cat_map[cls], 'name': cls,
         'keypoints': kps,
         'skeleton': [[1, 2], [2, 3]] if cls == 'face' else [[1, 2]]}
        for cls, kps in kp_idx.items()
    ]
    return {'images': images, 'annotations': annotations, 'categories': categories}


def convert_coco_to_yolo(coco, images_dir, project_dir):
    # directories
    out = pathlib.Path(project_dir) / 'yolo_pose_dataset'
    splits = ['train', 'val']
    for s in splits:
        (out / 'images' / s).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / s).mkdir(parents=True, exist_ok=True)

    # split image IDs 70/30
    img_ids = [img['id'] for img in coco['images']]
    random.shuffle(img_ids)
    n = len(img_ids)
    n_train = floor(0.7 * n)
    train_ids = img_ids[:n_train]
    val_ids = img_ids[n_train:]
    dims = {img['id']: (img['width'], img['height']) for img in coco['images']}

    # copy images
    for img in coco['images']:
        if img['id'] in train_ids:
            split = 'train'
        elif img['id'] in val_ids:
            split = 'val'
        else:
            continue
        copy2(pathlib.Path(images_dir) / img['file_name'], out / 'images' / split / img['file_name'])

    # write labels
    for ann in coco['annotations']:
        if ann['image_id'] in train_ids:
            split = 'train'
        elif ann['image_id'] in val_ids:
            split = 'val'
        else:
            continue
        w, h = dims[ann['image_id']]
        x, y, bw, bh = ann['bbox']
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        kpts = []
        for i in range(0, len(ann['keypoints']), 3):
            kpts += [
                ann['keypoints'][i] / w,
                ann['keypoints'][i+1] / h,
                ann['keypoints'][i+2]
            ]
        line = ' '.join(map(str, [ann['category_id'], xc, yc, nw, nh] + kpts))
        img_entry = next(img for img in coco['images'] if img['id'] == ann['image_id'])
        stem = pathlib.Path(img_entry['file_name']).stem
        label_file = out / 'labels' / split / f"{stem}.txt"
        with open(label_file, 'a') as lf:
            lf.write(line + "\n")

    # pose.yaml
    num_kp = len(coco['categories'][0]['keypoints'])
    names = coco['categories'][0]['keypoints']
    flip_idx = []
    for kp in names:
        if kp.startswith('left_'):
            flip_idx.append(names.index(kp.replace('left_', 'right_')))
        elif kp.startswith('right_'):
            flip_idx.append(names.index(kp.replace('right_', 'left_')))
        else:
            flip_idx.append(names.index(kp))
    yaml = f"""\
train: images/train
val: images/val

channels: 1

kpt_shape: [{num_kp}, 3]
flip_idx: {flip_idx}

names:
"""
    for cat in coco['categories']:
        yaml += f"  {cat['id']}: {cat['name']}\n"
    with open(out / 'pose.yaml', 'w') as f:
        f.write(yaml)


def main():
    parser = argparse.ArgumentParser(description="Convert LS project to YOLO-Pose with 70/30 train/val splits")
    parser.add_argument('project_dir', help="Root project directory")
    parser.add_argument('xml', help="Interface XML filename")
    parser.add_argument('ls_json', help="LS export JSON filename")
    parser.add_argument('images_dir', help="Images folder name")
    args = parser.parse_args()

    proj = pathlib.Path(args.project_dir)
    xml = proj / args.xml
    jsn = proj / args.ls_json
    imgd = proj / args.images_dir

    coco = convert_ls_to_coco(jsn, xml)
    with open(proj / 'coco_keypoints.json', 'w') as f:
        json.dump(coco, f, indent=2)
    convert_coco_to_yolo(coco, imgd, proj)
    total = len(coco['images'])
    print(f"\nDataset summary:")
    print(f"  Images included in dataset: {total}")
    print(f"  Images skipped (no annotations): {skipped_count}")
    print(f"  Train split: {floor(0.7 * total)} images")
    print(f"  Val split: {total - floor(0.7 * total)} images")
    print(f"\n✓ Done: coco_keypoints.json and yolo_pose_dataset created in {args.project_dir}")


if __name__ == '__main__':
    random.seed(42)
    main()
