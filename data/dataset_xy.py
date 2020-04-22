import tensorflow as tf
import numpy as np

class TFDataset():

    def __init__(self, path, batch_size, dataset='xy', shuffle_buffer=10000):
        self.path = path
        self.batch_size = batch_size
        self.iterator = None
        self.dataset_type = dataset
        self.shuffle_buffer = shuffle_buffer
        self.dataset = self.read_tfrecord(self.path, self.batch_size)
        self.get_iterator()

    def read_tfrecord(self, path, batch_size):
        # dataset = tf.data.TFRecordDataset(path).shuffle(self.shuffle_buffer).batch(batch_size)
        dataset = tf.data.TFRecordDataset(path).shuffle(self.shuffle_buffer)

        if self.dataset_type == 'xy':
            dataset = dataset.map(self.extract_xy_data_fn)
             
        return dataset


    def extract_xy_data_fn(self, data_record):
        feature_description = {
            'start_cartesian': tf.io.FixedLenFeature([], tf.string),
            'start_cartesian_shape': tf.io.FixedLenFeature([], tf.string),
            'goal_cartesian': tf.io.FixedLenFeature([], tf.string),
            'goal_cartesian_shape': tf.io.FixedLenFeature([], tf.string),
            'voxels': tf.io.FixedLenFeature([], tf.string),
            'voxels_shape': tf.io.FixedLenFeature([], tf.string),
            'planned_result_cartesian_interpolated': tf.io.FixedLenFeature([], tf.string),
            'planned_result_cartesian_interpolated_shape': tf.io.FixedLenFeature([], tf.string),
        }

        sample = tf.io.parse_single_example(data_record, feature_description)

        return sample
        

    def get_iterator(self):
    
        self.iterator = self.dataset.__iter__()


    def reset_iterator(self):
    
        self.dataset.shuffle(self.shuffle_buffer)
        self.get_iterator()

    def get_batch(self):
        start_cartesian_batch = np.zeros((self.batch_size, 7))
        goal_cartesian_batch = np.zeros((self.batch_size, 7))
        voxels_batch = np.zeros((self.batch_size, 2048, 3))
        plan_interpolated_batch = np.zeros((self.batch_size, 11, 7))
        
        for i in range(self.batch_size):
            try:
                d = self.iterator.next()
                start_cartesian_batch[i, :] = np.frombuffer(d['start_cartesian'].numpy(), dtype=np.float64)
                goal_cartesian_batch[i, :] = np.frombuffer(d['goal_cartesian'].numpy(), dtype=np.float64)
                voxels_batch[i, :, :] = np.frombuffer(d['voxels'].numpy().reshape(2048, 3), dtype=np.float64)
                plan_interpolated_batch[i,:,:] = np.frombuffer(d['planned_result_cartesian_interpolated'].numpy().reshape(11, 7), dtype=np.float64)
                
            except:
                self.reset_iterator()

        return start_cartesian_batch, goal_cartesian_batch, voxels_batch, plan_interpolated_batch



if __name__ == "__main__":
    import os
    
    cls = TFDataset(path=os.path.join(os.getcwd(), 'xy_data_tfrecord', 'data.tfrecord'),
                    batch_size=2)

    s, g, v, p = cls.get_batch()

    print(s.shape)
    print(g.shape)
    print(v.shape)
    print(p.shape)