
r

from __future__ import print_function

import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def main(_):
  
  if FLAGS.wav_file:
    wav_file = FLAGS.wav_file
  else:
    
    num_secs = 5
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)
    
    samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    wav_file = six.BytesIO()
    soundfile.write(wav_file, samples, sr, format='WAV', subtype='PCM_16')
    wav_file.seek(0)
  examples_batch = vggish_input.wavfile_to_examples(wav_file)
  print(examples_batch)

  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

  writer = tf.python_io.TFRecordWriter(
      FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

  with tf.Graph().as_default(), tf.Session() as sess:
    
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    print('Embedding:')
    print(embedding_batch.shape)
    print(embedding_batch)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    print('Embebdding(post-processed):')
    print(postprocessed_batch)

    seq_example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={
                vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[embedding.tobytes()]))
                            for embedding in postprocessed_batch
                        ]
                    )
            }
        )
    )
    print(seq_example)
    if writer:
      writer.write(seq_example.SerializeToString())

  if writer:
    writer.close()

if __name__ == '__main__':
  tf.app.run()
