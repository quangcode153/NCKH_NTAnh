
import os
import csv
import tqdm
import torch
import pickle
import numpy as np
from copy import deepcopy
import PIL.Image

from os.path import join as pjoin

class Fer2013Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, mode='val', transform=None,
                 include_train=False):
        self.data_dir = data_dir
        self.mode = mode
        self.include_train = include_train
        self._transform = transform
        self.pkl_path = pjoin(data_dir, 'pytorch', 'data.pkl')

        if not os.path.isfile(self.pkl_path):
            self.prepare_data()

        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        
        im_data = self.data['images'][self.mode][index].astype('uint8')
        image = PIL.Image.fromarray(im_data)
        label = self.data['labels'][self.mode][index]
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def prepare_data(self):
        
        print('preparing data...')
        with open(pjoin(self.data_dir, 'fer2013.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) 
            rows = [row for row in reader]

        train_ims, val_ims, test_ims = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for row in tqdm.tqdm(rows):
            subset = row[2]
            raw_im = np.array([int(x) for x in row[1].split(' ')])
            im = np.repeat(raw_im.reshape(48,48)[:,:,np.newaxis], 3, axis=2)
            if subset == 'Training':
                train_labels.append(int(row[0]))
                train_ims.append(im)
            elif subset == 'PublicTest':
                val_labels.append(int(row[0]))
                val_ims.append(im)
            elif subset == 'PrivateTest':
                test_labels.append(int(row[0]))
                test_ims.append(im)
            else:
                raise ValueError('unrecognised subset: {}'.format(subset))

        data = {'labels': {}, 'images': {}}
        data['labels']['val'] = np.array(val_labels)
        data['labels']['test'] = np.array(test_labels)

        data['images']['val'] = np.array(val_ims)
        data['images']['test'] = np.array(test_ims)

        if self.include_train:
            data['labels']['train'] = np.array(train_labels)
            data['images']['train'] = np.array(train_ims)

        for key in 'images', 'labels':
            assert len(data[key]['val']) == 3589, 'unexpected length'
            assert len(data[key]['test']) == 3589, 'unexpected length'
            if self.include_train:
                assert len(data[key]['train']) == 28709, 'unexpected length'

        if not os.path.exists(os.path.dirname(self.pkl_path)):
                os.makedirs(os.path.dirname(self.pkl_path))

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data, f)

    def __len__(self):
        
        return self.data['labels'][self.mode].size

class Fer2013PlusDataset(Fer2013Dataset):
    
    def __init__(self, *args, **kwargs):
        super(Fer2013PlusDataset, self).__init__(*args, **kwargs)
        self.update_labels()

    def update_labels(self):
        
        with open(pjoin(self.data_dir, 'fer2013new.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) 
            rows = [row for row in reader]

        set_map = {'Training': 1, 'PublicTest': 2, 'PrivateTest': 3}
        sets = np.atleast_2d([set_map[x[0]] for x in rows]).T
        labels = [np.atleast_2d([int(x) for x in r[2:]]) for r in rows]
        labels = np.concatenate(labels, axis=0)
        orig_labels = deepcopy(labels)
        outliers = (labels <=1)
        labels[outliers] = 0 
        dropped = 1 - (labels.sum() / orig_labels.sum())
        print('dropped {:.1f}%% of votes as outliers'.format(dropped * 100))
        num_votes = np.sum(labels, 1)
        
        to_drop = np.zeros((labels.shape[0], 1))
        for ii in tqdm.tqdm(range(labels.shape[0])):
            max_vote = np.max(labels[ii,:])
            max_vote_emos = np.where(labels[ii,:] == max_vote)[0]
            drop = any([x in [8, 9] for x in max_vote_emos])
            num_max_votes = max_vote_emos.size
            drop = drop or num_max_votes >= 3
            drop = drop or (num_max_votes * max_vote <= 0.5 * num_votes[ii])
            to_drop[ii] = drop

        assert to_drop.sum() == 3079, 'unexpected number of dropped votes'
        
        val_keep_ims = np.logical_not(to_drop[sets == 2])
        test_keep_ims = np.logical_not(to_drop[sets == 3])
        val_keep_labels = np.logical_and(sets == 2,
                                      np.logical_not(to_drop)).flatten()
        test_keep_labels = np.logical_and(sets == 3,
                                      np.logical_not(to_drop)).flatten()
        val_labels = labels[val_keep_labels, :]
        test_labels = labels[test_keep_labels, :]
        print('val size: ', len(val_labels), 'test size: ', len(test_labels))
        
        self.data['images']['val'] = \
                            self.data['images']['val'][val_keep_ims,:,:,:]
        self.data['images']['test'] = \
                            self.data['images']['test'][test_keep_ims,:,:,:]

        self.data['labels']['val'] = np.argmax(val_labels, 1)
        self.data['labels']['test'] = np.argmax(test_labels, 1)
