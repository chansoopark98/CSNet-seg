import tensorflow as tf
import tensorflow_datasets as tfds

data_dir = './datasets/'
image_size = (512, 1024)

@tf.function
def preprocess_valid(sample):
    # img = sample['image_left']
    labels = sample['segmentation_label']#

    return (labels)


dataset = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=data_dir, split='train')

dataset = dataset.map(preprocess_valid)
dataset = dataset.batch(1)

# dataset = dataset.take(10)
# id_list = []
# id_dict = {}
counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for x in dataset:

    y, idx, count = tf.unique_with_counts(tf.reshape(x, [-1]))

    for i in range(len(y)):

        count_np = count.numpy()
        counts[y[i]] += count_np[i]
        #print(counts)


total = 0
for i in range(len(counts)):
    total+= counts[i]
for i in range(len(counts)):
    counts[i] = 1 / tf.math.log(1.02+(counts[i]/total))

for i in range(len(counts)):
    print(counts[i])
# class_weight = 1 / np.log(1.02 + (frequency / total_frequency))


class_weight = [
    2.8543523840037177,
    1.3501380011580169,
    4.970495934756391,
    1.8887107725102839,
    15.066038400152298,
    13.837913138463685,
    12.277478720550748,
    18.378001839340588,
    15.71545310733969,
    2.4582169536505725,
    12.555716676189862,
    6.588000419104747,
    12.318122119569152,
    19.072811312045705,
    4.500762248314975,
    17.856449045143542,
    18.13687989115995,
    18.157941306594143,
    19.434675017727237,
    16.688571815264762
]
