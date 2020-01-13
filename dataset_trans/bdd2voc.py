import os
import click
import json
from typing import List, Dict

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

from collections import namedtuple
from PIL import Image


IMG_ATTRIBUTES = namedtuple(
    "IMG_ATTRIBUTES",
    [
        "weather",
        "scene",
        "timeofday"
    ]
)

BOX_ATTRIBUTES = namedtuple(
    "BOX_ATTRIBUTES",
    [
        "occluded",
        "truncated",
        "trafficLightColor"
    ]
)


class VOCWriter:
    def __init__(self, xml_path, image_hw, filename, file_split, img_attributes: IMG_ATTRIBUTES):
        self.filename = filename
        self.file_split = file_split
        self.image_hw = image_hw

        self._top = self._gen_xml(img_attributes)

        self.xml_path = xml_path

        self._box_list = []

    def _gen_xml(self, attr: IMG_ATTRIBUTES):
        """
            Return XML root
        """
        top = Element("annotation")
        top.set("verified", "yes")

        folder = SubElement(top, "folder")
        folder.text = "bdd100k/labels"

        filename = SubElement(top, "filename")
        filename.text = self.filename

        local_img_path = SubElement(top, "path")
        local_img_path.text = f"bdd100k/images/100k/{self.file_split}/{self.filename}.jpg"

        source = SubElement(top, "source")
        database = SubElement(source, "database")
        database.text = "Unknown"

        size_part = SubElement(top, "size")
        width = SubElement(size_part, "width")
        height = SubElement(size_part, "height")
        depth = SubElement(size_part, "depth")
        width.text = str(self.image_hw[1])
        height.text = str(self.image_hw[0])
        depth.text = "3"

        segmented = SubElement(top, "segmented")
        segmented.text = "0"

        attributes = SubElement(top, "attributes")
        weather = SubElement(attributes, "weather")
        weather.text = attr.weather
        scene = SubElement(attributes, "scene")
        scene.text = attr.scene
        timeofday = SubElement(attributes, "timeofday")
        timeofday.text = attr.timeofday

        return top

    def add_bbox(self, xmin, ymin, xmax, ymax, class_name, box_attributes: BOX_ATTRIBUTES):
        bbox = {
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "class_name": class_name,
            "attributes": box_attributes
        }
        self._box_list.append(bbox)

    def _append_object(self):
        for each_object in self._box_list:
            object_item = SubElement(self._top, "object")
            name = SubElement(object_item, "name")
            name.text = each_object["class_name"]

            pose = SubElement(object_item, "pose")
            pose.text = "Unspecified"

            attr: BOX_ATTRIBUTES = each_object["attributes"]

            occluded = SubElement(object_item, "occluded")
            occluded.text = "1" if attr.occluded else "0"

            truncated = SubElement(object_item, "truncated")
            truncated.text = "1" if attr.truncated else "0"

            traffic_light_col = SubElement(object_item, "trafficLightColor")
            traffic_light_col.text = attr.trafficLightColor

            difficult = SubElement(object_item, "difficult")
            difficult.text = "0"
            bndbox = SubElement(object_item, "bndbox")
            xmin = SubElement(bndbox, "xmin")
            xmin.text = str(each_object["xmin"])
            ymin = SubElement(bndbox, "ymin")
            ymin.text = str(each_object["ymin"])
            xmax = SubElement(bndbox, "xmax")
            xmax.text = str(each_object["xmax"])
            ymax = SubElement(bndbox, "ymax")
            ymax.text = str(each_object["ymax"])

    @staticmethod
    def _prettify(elem):
        rough_string = ElementTree.tostring(elem, "utf8")
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding="utf-8").replace("  ".encode(), "\t".encode())

    def save(self):
        self._append_object()
        prettify_top = self._prettify(self._top)
        xml_file = codecs.open(self.xml_path, 'w', encoding="utf-8")
        xml_file.write(prettify_top.decode("utf8"))
        xml_file.close()


def load_bdd_anns(label_path: str, file_split: str):
    path = os.path.join(label_path, f"bdd100k_labels_images_{file_split}.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as file:
        anns: List[Dict] = json.load(file)

    return anns


def write_voc_anns(bdd_anns, file_split, output_dir, img_dir, output_size, class_names):
    output_image_dir = os.path.join(output_dir, "images", file_split)
    output_label_dir = os.path.join(output_dir, "labels", file_split)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for ann in bdd_anns:
        filename = ann["name"]
        image_path = os.path.join(img_dir, file_split, filename)
        img: Image.Image = Image.open(image_path)

        # 32 alignment
        img = img.crop((0, 8, 1280, 712))
        img = img.resize(output_size)

        img.save(os.path.join(output_image_dir, filename))

        image_hw = output_size[::-1]

        img_attr = ann["attributes"]

        filename = os.path.splitext(filename)[0]
        xml_path = os.path.join(output_label_dir, f"{filename}.xml")
        writer = VOCWriter(xml_path, image_hw, filename, file_split,
                           IMG_ATTRIBUTES(img_attr["weather"], img_attr["scene"], img_attr["timeofday"]))

        boxes = ann["labels"]
        for box in boxes:
            # ignore drivable area and lane
            if "poly2d" in box:
                continue
            class_name = box["category"]
            box_attr = box["attributes"]
            box2d = box["box2d"]

            writer.add_bbox(box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"], class_name,
                            BOX_ATTRIBUTES(box_attr["occluded"], box_attr["truncated"], box_attr["trafficLightColor"]))
            class_names.add(class_name)

        writer.save()


def save_label_map(label_map: dict, output_path: str):
    label_map_list = [(idx, name) for name, idx in label_map.items()]
    label_map_list.sort(key=lambda x: x[0])

    with open(output_path, "w") as f:
        for idx, name in label_map_list:
            line = "item {\n"
            line += f"  id: {idx}\n"
            line += f"  name: '{name}'\n"
            line += "}\n\n"

            f.write(line)


@click.command()
@click.option("--bdd_img_dir", "-i", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="/Path/to/bdd100k/images/100k")
@click.option("--bdd_label_dir", "-l", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="/Path/to/bdd100k/labels")
@click.option("--output_dir", "-o", required=True,
              type=click.Path(exists=False, file_okay=False, writable=True),
              help="/Path/to/voc")
@click.option("--output_size", "-s", required=True, type=click.Tuple(types=(int, int)), help="Alignment 32")
def run(bdd_img_dir, bdd_label_dir, output_dir, output_size):
    if output_size[0] < 0:
        output_size = (1280, 704)

    class_names = set()

    output_splits = ["train", "val"]
    # output_splits = ["val"]
    for file_split in output_splits:
        anns = load_bdd_anns(bdd_label_dir, file_split=file_split)
        write_voc_anns(anns, file_split, output_dir, bdd_img_dir, output_size, class_names)

    class_names = list(class_names)
    class_names.sort()
    label_map = {name: idx for idx, name in enumerate(class_names, 1)}

    save_label_map(label_map, os.path.join(output_dir, "class_map.txt"))


if __name__ == "__main__":
    run()
