
import numpy as np

import vggish_params

class Postprocessor(object):
  
  def __init__(self, pca_params_npz_path):
    
    params = np.load(pca_params_npz_path)
    self._pca_matrix = params[vggish_params.PCA_EIGEN_VECTORS_NAME]
    
    self._pca_means = params[vggish_params.PCA_MEANS_NAME].reshape(-1, 1)
    assert self._pca_matrix.shape == (
        vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE), (
            'Bad PCA matrix shape: %r' % (self._pca_matrix.shape,))
    assert self._pca_means.shape == (vggish_params.EMBEDDING_SIZE, 1), (
        'Bad PCA means shape: %r' % (self._pca_means.shape,))

  def postprocess(self, embeddings_batch):
    
    assert len(embeddings_batch.shape) == 2, (
        'Expected 2-d batch, got %r' % (embeddings_batch.shape,))
    assert embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE, (
        'Bad batch shape: %r' % (embeddings_batch.shape,))

    pca_applied = np.dot(self._pca_matrix,
                         (embeddings_batch.T - self._pca_means)).T

    clipped_embeddings = np.clip(
        pca_applied, vggish_params.QUANTIZE_MIN_VAL,
        vggish_params.QUANTIZE_MAX_VAL)
    
    quantized_embeddings = (
        (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL) *
        (255.0 /
         (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)))
    
    quantized_embeddings = quantized_embeddings.astype(np.uint8)

    return quantized_embeddings
