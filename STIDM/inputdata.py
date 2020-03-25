import numpy as np

class DataSet(object):
    def __init__(self,samples,labels=None):
        assert samples.shape[0]==labels.shape[0],('samples.shape:%s\n labels.shape:%s' %(samples.shape,labels.shape))
        self._num_examples = samples.shape[0]
        self._samples = samples
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def samples(self):
        return self._samples

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoches_completed(self):
        return self._epochs_completed

    # 按照batch_size 的大小逐个生成data和label
    def next_batch(self,batch_size,shuffle=True):
        # print('batch_size:',batch_size)
        # print('num_examples:',self._num_examples)
        # print('sample_shape:',self._samples.shape)
        # print('label_shape:',self._labels.shape)
        # print('start:',self._index_in_epoch)
        start = self._index_in_epoch
        # shuffle for the first epoch  第一次的时候shuffle整个数据集
        if self._epochs_completed == 0 and start==0 and shuffle:
            # shuffle samples
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._samples = self.samples[perm0]
            self._labels = self.labels[perm0]
        # go to next epoch
        if start+batch_size>self._num_examples:
            # finish the epoch
            self._epochs_completed+=1
            # get the rest samples
            rest_num_examples = self._num_examples-start
            samples_rest_part = self._samples[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # shuffle the input
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._samples = self.samples[perm]
                self._labels = self.labels[perm]
            # start next batch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            samples_new_part = self.samples[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((samples_rest_part, samples_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._samples[start:end], self.labels[start:end]
