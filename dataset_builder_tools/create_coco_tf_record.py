import os
import sys
import json
import collections
import multiprocessing
from PIL import Image
import io
import hashlib

import tensorflow as tf

sys.path.append(os.path.abspath("../"))

# from research.object_detection.utils import dataset_util
# from research.object_detection.utils import label_map_util
# from dataset_tools import label_map_util
# from dataset_tools import dataset_util
# import label_map_util
# import dataset_util
from object_detection_utils import label_map_util
from object_detection_utils import dataset_util

flags = tf.flags

flags.DEFINE_string("image_dir", "", "Directory containing images.")
flags.DEFINE_string("image_info_file", "", "File containing image information. "
                                           "Tf Examples in the output files correspond to the image "
                                           "info entries in this file. If this file is not provided "
                                           "object_annotations_file is used if present. Otherwise, "
                                           "caption_annotations_file is used to get image info.")
flags.DEFINE_string("object_annotations_file", "", "File containing object "
                                                   "annotations - boxes and instance masks.")
flags.DEFINE_string("output_file_prefix", "/tmp/train", "Path to output file")
flags.DEFINE_integer("num_shards", 32, "Number of shards for output file.")

FLAGS = flags.FLAGS


def _create_tf_example(image,
                       image_dir,
                       obj_annotations=None,
                       category_index=None,
                       ):
    img_height = image["height"]
    img_width = image["width"]
    filename = image["file_name"]
    image_id = image["id"]

    full_path = os.path.join(image_dir, filename)

    # jpg test
    with tf.gfile.GFile(full_path, "rb") as file:
        encoded_jpg = file.read()
    # img_width, img_height = org_img_width, org_img_height
    # Image.open(io.BytesIO(encoded_jpg))

    key = hashlib.sha256(encoded_jpg).hexdigest()

    feature_dict = {
        "image/height": dataset_util.int64_feature(img_height),
        "image/width": dataset_util.int64_feature(img_width),
        "image/filename": dataset_util.bytes_feature(filename.encode("utf8")),
        "image/source_id": dataset_util.bytes_feature(str(image_id).encode("utf8")),
        "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
        # "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature("jpeg".encode("utf8")),
    }

    num_annotations_skipped = 0
    if obj_annotations:
        y_min, x_min, y_max, x_max = [], [], [], []
        is_crowd = []
        category_names = []
        category_ids = []
        area = []

        for ann in obj_annotations:
            (x, y, w, h) = tuple(ann["bbox"])

            if (w <= 0 or h <= 0) or (x + w > img_width or y + h > img_height):
                num_annotations_skipped += 1
                continue

            y_min.append(float(y) / img_height)
            x_min.append(float(x) / img_width)
            y_max.append(float(y + h) / img_height)
            x_max.append(float(x + w) / img_width)
            is_crowd.append(ann["iscrowd"])

            category_id = int(ann["category_id"])
            category_ids.append(category_id)
            category_names.append(category_index[category_id]["name"].encode("utf8"))
            area.append(ann["area"])

        feature_dict.update(
            {
                "image/object/bbox/ymin": dataset_util.float_list_feature(y_min),
                "image/object/bbox/xmin": dataset_util.float_list_feature(x_min),
                "image/object/bbox/ymax": dataset_util.float_list_feature(y_max),
                "image/object/bbox/xmax": dataset_util.float_list_feature(x_max),
                "image/object/class/text": dataset_util.bytes_list_feature(category_names),
                "image/object/class/label": dataset_util.int64_list_feature(category_ids),
                "image/object/is_crowd": dataset_util.int64_list_feature(is_crowd),
                "image/object/area": dataset_util.float_list_feature(area),
            }
        )

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
    return _create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
    with tf.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)

    images = obj_annotations["images"]
    category_index = label_map_util.create_category_index(obj_annotations["categories"])

    img_to_obj_annotation = collections.defaultdict(list)
    tf.logging.info("Building bounding box index.")
    for annotation in obj_annotations["annotations"]:
        image_id = annotation["image_id"]
        img_to_obj_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image["id"]
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1

    tf.logging.info(f"{missing_annotation_count} images are missing bounding boxes.")

    return img_to_obj_annotation, category_index


def _load_images_info(images_info_file):
    with tf.gfile.GFile(images_info_file, 'r') as fid:
        info_dict = json.load(fid)
    return info_dict['images']


def _create_tf_record_from_coco_annotations(images_info_file: str,
                                            image_dir: str,
                                            output_path: str,
                                            num_shards: int,
                                            object_annotations_file: str = None,
                                            ):
    tf.logging.info(f"writing to output path: {output_path}")
    writers = [
        tf.python_io.TFRecordWriter(output_path + f"-{i:05d}-of-{num_shards:05d}.tfrecord")
        for i in range(num_shards)
    ]

    images = _load_images_info(images_info_file)

    img_to_obj_annotation = None
    category_index = None

    if object_annotations_file:
        img_to_obj_annotation, category_index = _load_object_annotations(object_annotations_file)

    def _get_object_annotation(image_id):
        if img_to_obj_annotation and (image_id in img_to_obj_annotation):
            return img_to_obj_annotation[image_id]
        else:
            return None

    pool = multiprocessing.Pool()
    total_num_annotations_skipped = 0
    for idx, (_, tf_example, num_annotations_skipped) in enumerate(
        pool.imap(
            _pool_create_tf_example,
            [
                (
                    image,
                    image_dir,
                    _get_object_annotation(image["id"]),
                    category_index,
                ) for image in images
            ]
        )
    ):
        if idx % 100 == 0:
            tf.logging.info(f"On image {idx} of {len(images)}.")

        total_num_annotations_skipped += num_annotations_skipped
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()

    tf.logging.info(f"Finished writing, skipped {total_num_annotations_skipped} annotations.")


def main(_):
    assert FLAGS.image_dir, "'image_dir' missing."
    assert (FLAGS.image_info_file or FLAGS.object_annotations_file or
            FLAGS.caption_annotations_file), ("All annotation files are "
                                              "missing.")
    if FLAGS.image_info_file:
        images_info_file = FLAGS.image_info_file
    elif FLAGS.object_annotations_file:
        images_info_file = FLAGS.object_annotations_file
    else:
        raise ValueError

    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.gfile.IsDirectory(directory):
        tf.gfile.MakeDirs(directory)

    _create_tf_record_from_coco_annotations(
        images_info_file,
        FLAGS.image_dir,
        FLAGS.output_file_prefix,
        FLAGS.num_shards,
        FLAGS.object_annotations_file,
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
