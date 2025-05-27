#!/usr/bin/env python3
"""
export_ls_project.py

A simple CLI script to export a Label Studio project's configuration XML,
COCO-with-images ZIP (and unzip it), and full JSON annotations to a specified folder.

Usage:
    python export_ls_project.py <project_id> <output_folder>
Example:
    python export_ls_project.py 1 /home/lab/labeling/labeled
"""

import os
import argparse
import zipfile
from label_studio_sdk import Client


def main():
    parser = argparse.ArgumentParser(
        description='Export Label Studio project assets (XML, COCO+images, JSON)'
    )
    parser.add_argument(
        'project_id',
        type=int,
        help='Numeric ID of the Label Studio project to export'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory where export files will be saved'
    )
    args = parser.parse_args()

    # Check if the output directory already exists
    if os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' already exists. Aborting to prevent overwrite.", file=sys.stderr)
        sys.exit(1)

    # Create the output directory
    os.makedirs(args.output_dir)

    # Initialize Label Studio client
    ls = Client(
        url='http://localhost:8080',
        api_key='cde00134ce9cef6811184456b4ae9e6d722ea5cd'
    )

    # Fetch the project
    project = ls.get_project(args.project_id)

    # 1. Export labeling configuration XML
    config_xml = project.get_params().get('label_config', '')
    xml_path = os.path.join(args.output_dir, 'label_config.xml')
    with open(xml_path, 'w') as f:
        f.write(config_xml)
    print(f'[✔] Label config saved to: {xml_path}')

    # 2. Export COCO annotations + images zip
    coco_zip_path = os.path.join(args.output_dir, 'coco_with_images.zip')
    project.export_tasks(
        export_type='COCO_WITH_IMAGES',
        download_resources=True,
        export_location=coco_zip_path
    )
    print(f'[✔] COCO with images exported to: {coco_zip_path}')

    # 2a. Unzip COCO zip into output directory
    with zipfile.ZipFile(coco_zip_path, 'r') as zip_ref:
        zip_ref.extractall(args.output_dir)
    print(f'[✔] Extracted COCO images and annotations to: {args.output_dir}')

    # 3. Export full JSON annotations
    json_path = os.path.join(args.output_dir, 'annotations.json')
    project.export_tasks(
        export_type='JSON',
        export_location=json_path
    )
    print(f'[✔] JSON annotations saved to: {json_path}')


if __name__ == '__main__':
    main()
