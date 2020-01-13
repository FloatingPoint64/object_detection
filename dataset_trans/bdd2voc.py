import os
import glob
import click
import json
from typing import List, Dict

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


class VOCWriter:
    def __init__(self, xml_path, image_size, img_filename):
        pass


def get_img_list(img_dir: str, file_split: str):
    img_path_list = glob.glob(os.path.join(img_dir, file_split, "*.jpg"))

    return img_path_list


def load_bdd_anns(label_path: str, file_split: str):
    path = os.path.join(label_path, f"bdd100k_labels_images_{file_split}.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as file:
        anns: List[Dict] = json.load(file)

    return anns


def write_voc_anns(bdd_anns, file_split, output_dir, img_dir):
    img_path_list = get_img_list(img_dir, file_split)

    for ann in bdd_anns:
        img_filename = ann["filename"]
        pass


@click.command()
@click.option("--bdd_img_dir", "-i", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="/Path/to/bdd100k/images/100k")
@click.option("--bdd_label_dir", "-l", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="/Path/to/bdd100k/labels")
@click.option("--output_label_dir", "-o", required=True,
              type=click.Path(exists=False, file_okay=False, writable=True),
              help="/Path/to/bdd100k/voc")
def run(bdd_img_dir, bdd_label_dir, output_label_dir):
    train_anns = load_bdd_anns(bdd_label_dir, file_split="train")
    val_anns = load_bdd_anns(bdd_label_dir, file_split="train")

    os.makedirs(output_label_dir, exist_ok=True)


if __name__ == '__main__':
    run()
