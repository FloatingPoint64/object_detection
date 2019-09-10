import numpy as np
import tensorflow as tf
from PIL import Image

from dataset_tools import tf_example_decoder

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DataBuilderTest(tf.test.TestCase):
    def test_img_load(self):
        dataset_pattern = "N:/DNN/project/pycharm/squeeze_det/test_data/dataset/coco/val-*-of-00256.tfrecord"

        dataset = tf.data.Dataset.list_files(dataset_pattern, shuffle=False)

        def _prefetch_dataset(filename):
            return tf.data.TFRecordDataset(filename).prefetch(1)

        dataset = dataset.apply(tf.data.experimental.parallel_interleave(
            _prefetch_dataset,
            cycle_length=32,
            sloppy=False
        )
        )

        example_decoder = tf_example_decoder.TfExampleDecoder()

        def _dataset_parser(value):
            data = example_decoder.decode(value)

            image = data["image"]

            return image

        dataset = dataset.map(_dataset_parser, num_parallel_calls=64)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        dataset = dataset.batch(1, drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()
        features_op = iterator.get_next()

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                allow_growth=True
            )
        )

        with tf.Session(config=sess_config) as sess:
            tf.global_variables_initializer().run()

            count = 0
            while count < 10:
                try:
                    features: np.ndarray = sess.run(features_op)
                except tf.errors.OutOfRangeError:
                    break

                for feature in features:  # type: np.ndarray
                    img = Image.fromarray(feature.astype(np.uint8))
                    img.show()

                count += 1
