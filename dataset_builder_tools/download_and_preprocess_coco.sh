#!/bin/bash
# usage:
#  bash download_and_preprocess_coco.sh /path/to/data/directory

set -eu
set -x

if [ -z "$1" ]; then
  esho "usage download_and_preprocess_coco.sh [data dir]"
  exit
fi

# Install dependencies
sudo apt install unzip

protoc object_detection_utils/*.proto --python_out=.

#sudo apt install -y protobuf-compiler python-pil python-lxml python-pip python-dev git unzip
#
#pip install Cython git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI
#
#pip install --upgrade tensorflow
#
#echo "Cloning Tensorflow models directory (for conversion utilities)"
#if [ ! -e tf-models ]; then
#  git clone http://github.com/tensorflow/models tf-models
#fi
#
#(cd tf-models/research && protoc object_detection/protos/*.proto --python_out=.)

UNZIP="unzip -nq"

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
    local BASE_URL=${1}
    local FILENAME=${2}

    cd "${SCRATCH_DIR}"

    if [ ! -f "${FILENAME}" ]; then
      echo "Downloading ${FILENAME} to $(pwd)."
      wget -nd -c "${BASE_URL}/${FILENAME}"
    else
      echo "Skipping download of ${FILENAME}"
    fi

    echo "Unzipping ${FILENAME}"
    ${UNZIP} "${FILENAME}"
    cd "${CURRENT_DIR}"
}

# Download all annotations.
BASE_ANNOTATION_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval2017.zip"
download_and_unzip ${BASE_ANNOTATION_URL} ${INSTANCES_FILE}
IMAGE_INFO_FILE="image_info_test2017.zip"
download_and_unzip ${BASE_ANNOTATION_URL} ${IMAGE_INFO_FILE}
UNLABELED_IMAGE_INFO_FILE="image_info_unlabeled2017.zip"
download_and_unzip ${BASE_ANNOTATION_URL} ${UNLABELED_IMAGE_INFO_FILE}

# Setup packages
#touch tf-models/__init__.py
#touch tf-models/research/__init__.py

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BASE_IMAGE_URL="http://images.cocodataset.org/zips"

function create_dataset() {
  local IMAGE_FILENAME="${1}2017"

  local ZIP_FILENAME="${IMAGE_FILENAME}.zip"
  download_and_unzip ${BASE_IMAGE_URL} ${ZIP_FILENAME}

  local IMAGE_DIR="${SCRATCH_DIR}/${IMAGE_FILENAME}"
  local OBJ_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_${IMAGE_FILENAME}.json"

  python "$SCRIPT_DIR/create_coco_tf_record.py" \
    --logtostderr \
    --image_dir="${IMAGE_DIR}" \
    --object_annotations_file="${OBJ_ANNOTATIONS_FILE}" \
    --output_file_prefix="${OUTPUT_DIR}/${1}" \
    --num_shards=256
}

function create_testdev_and_test_dataset() {
  local IMAGE_FILE="test2017.zip"
  download_and_unzip ${BASE_IMAGE_URL} ${IMAGE_FILE}

  local IMAGE_DIR="${SCRATCH_DIR}/test2017"
  local IMAGE_INFO_FILE="${SCRATCH_DIR}/annotations/image_info_test2017.json"
  python "$SCRIPT_DIR/create_coco_tf_record.py" \
    --logtostderr \
    --image_dir="${IMAGE_DIR}" \
    --image_info_file="${IMAGE_INFO_FILE}" \
    --output_file_prefix="${OUTPUT_DIR}/test" \
    --num_shards=256

  local DEV_IMAGE_INFO_FILE="${SCRATCH_DIR}/annotations/image_info_test-dev2017.json"
  python "$SCRIPT_DIR/create_coco_tf_record.py" \
    --logtostderr \
    --image_dir="${IMAGE_DIR}" \
    --image_info_file="${DEV_IMAGE_INFO_FILE}" \
    --output_file_prefix="${OUTPUT_DIR}/test-dev" \
    --num_shards=256
}

function create_unlabeled_dataset() {
  local IMAGE_FILE="unlabeled2017.zip"
  download_and_unzip ${BASE_IMAGE_URL} ${IMAGE_FILE}

  local IMAGE_DIR="${SCRATCH_DIR}/unlabeled2017"
  local IMAGE_INFO_FILE="${SCRATCH_DIR}/annotations/image_info_unlabeled2017.json"

  python "$SCRIPT_DIR/create_coco_tf_record.py" \
    --logtostderr \
    --image_dir="${IMAGE_DIR}" \
    --image_info_file="${IMAGE_INFO_FILE}" \
    --output_file_prefix="${OUTPUT_DIR}/unlabeled" \
    --num_shards=256
}

#create_dataset "train"
create_dataset "val"
#create_testdev_and_test_dataset
#create_unlabeled_dataset
