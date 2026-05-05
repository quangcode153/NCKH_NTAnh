
import os
import re
import pandas as pd
import numpy as np
import struct

def read_hog(filename, batch_size=5000):
    
    all_feature_vectors = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4)) 
        num_rows, = struct.unpack("i", f.read(4)) 
        num_channels, = struct.unpack("i", f.read(4)) 

        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features)) 
        all_feature_vectors.append(feature_vector)

        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        
        num_bytes_to_read = num_floats_to_read * 4

        while True:
            bytes = f.read(num_bytes_to_read)
            
            num_bytes_read = len(bytes)
            assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

            feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
            
            feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
            
            feature_vectors = feature_vectors[:, 3:]
            
            all_feature_vectors.append(feature_vectors)

            if num_bytes_read < num_bytes_to_read:
                break

        all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

        is_valid = all_feature_vectors[:, 0]
        feature_vectors = all_feature_vectors[:, 1:]

        return is_valid, feature_vectors

def read_csv(filename, startIdx):
    data = pd.read_csv(filename)
    all_feature_vectors = []
    for index in data.index:
        features = np.array(data.iloc[index][startIdx:])
        all_feature_vectors.append(features)
    all_feature_vectors = np.array(all_feature_vectors)
    return all_feature_vectors

