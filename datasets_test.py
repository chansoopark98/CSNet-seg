import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import collections
data_dir = './datasets/'
image_size = (512, 1024)

@tf.function
def preprocess_valid(sample):
    img = sample['image_left']          #
    id = sample['image_id']
    labels = sample['segmentation_label']

    return (id, img, labels)


dataset = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=data_dir, split='train')

dataset = dataset.map(preprocess_valid)
dataset = dataset.batch(16).repeat()

dataset = dataset.take(10)
id_list = []
id_dict = {}
for id, x, y in dataset:
    


    id_list.append(str(id))

id_dict = collections.Counter(id_list)
print(id_dict)

