
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
tf.disable_v2_behavior()

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

print('\nTesting your install of VGGish\n')

checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'

rel_error = 0.1  

num_secs = 3
freq = 1000
sr = 44100
t = np.linspace(0, num_secs, int(num_secs * sr))
x = np.sin(2 * np.pi * freq * t)

input_batch = vggish_input.waveform_to_examples(x, sr)
print('Log Mel Spectrogram example: ', input_batch[0])
np.testing.assert_equal(
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

with tf.Graph().as_default(), tf.Session() as sess:
  vggish_slim.define_vggish_slim()
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  [embedding_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})
  print('VGGish embedding: ', embedding_batch[0])
  expected_embedding_mean = 0.131
  expected_embedding_std = 0.238
  np.testing.assert_allclose(
      [np.mean(embedding_batch), np.std(embedding_batch)],
      [expected_embedding_mean, expected_embedding_std],
      rtol=rel_error)

pproc = vggish_postprocess.Postprocessor(pca_params_path)
postprocessed_batch = pproc.postprocess(embedding_batch)
print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
expected_postprocessed_mean = 123.0
expected_postprocessed_std = 75.0
np.testing.assert_allclose(
    [np.mean(postprocessed_batch), np.std(postprocessed_batch)],
    [expected_postprocessed_mean, expected_postprocessed_std],
    rtol=rel_error)

print('\nLooks Good To Me!\n')
