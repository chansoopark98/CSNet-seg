import tensorflow as tf
import tensorflow_datasets as tfds

data_dir = './datasets/'
image_size = (512, 1024)

@tf.function
def preprocess_valid(sample):
    img = sample['image_left']
    # labels = sample['segmentation_label']#

    return (img)


dataset = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=data_dir, split='validation')

dataset = dataset.map(preprocess_valid)
dataset = dataset.batch(1)

dataset = dataset.take(1)

# concat_img = tf.concat([img, labels], axis=-1)
# concat_img = tf.image.random_crop(concat_img, (self.image_size[0], self.image_size[1], 4))
#
# img = concat_img[:, :, :3]
# labels = concat_img[:, :, 3:]
#
# img = tf.cast(img, dtype=tf.float32)
# labels = tf.cast(labels, dtype=tf.int64)
#
# if tf.random.uniform([]) > 0.5:
#     # img = tf.image.random_brightness(img, max_delta=0.4)
#     img = tf.image.random_brightness(img, max_delta=0.1)
# if tf.random.uniform([]) > 0.5:
#     # img = tf.image.random_contrast(img, lower=0.7, upper=1.4)
#     img = tf.image.random_contrast(img, lower=0.1, upper=0.8)
# # if tf.random.uniform([]) > 0.5:
# #     img = tf.image.random_hue(img, max_delta=0.4)
# if tf.random.uniform([]) > 0.5:
#     # img = tf.image.random_saturation(img, lower=0.7, upper=1.4)
#     img = tf.image.random_saturation(img, lower=0.1, upper=0.8)
# # if tf.random.uniform([]) > 0.5:
# #     img = tfa.image.sharpness(img, factor=0.5)
#
# if tf.random.uniform([]) > 0.5:
#     img = tf.image.flip_left_right(img)
#     labels = tf.image.flip_left_right(labels)
path = './experiment/'
resize_path = './experiment/resize/'
crop_path = './experiment/crop/'
for x in dataset:
    x = x[0]

    tf.keras.preprocessing.image.save_img(path+"orininal"+'.png', x)

    random_crop = tf.image.random_crop(x, (512, 1024, 3))
    tf.keras.preprocessing.image.save_img(crop_path+"random_crop" + '.png', random_crop)

    # random crop
    random_brightness = tf.image.random_brightness(random_crop, max_delta=0.4)
    tf.keras.preprocessing.image.save_img(crop_path +"random_brightness" + '.png', random_brightness)

    random_contrast = tf.image.random_contrast(random_crop, lower=0.4, upper=0.41)
    tf.keras.preprocessing.image.save_img(crop_path + "random_contrast" + '.png', random_contrast)

    random_saturation = tf.image.random_saturation(random_crop, lower=0.4, upper=0.41)
    tf.keras.preprocessing.image.save_img(crop_path + "random_saturation" + '.png', random_saturation)

    flip_left_right = tf.image.flip_left_right(random_crop)
    tf.keras.preprocessing.image.save_img(crop_path + "flip_left_right" + '.png', flip_left_right)


    # random resize

    resize = tf.image.resize(x, (512, 1024))
    tf.keras.preprocessing.image.save_img(resize_path + "resize" + '.png', resize)

    random_brightness = tf.image.random_brightness(resize, max_delta=0.4)
    tf.keras.preprocessing.image.save_img(resize_path +"random_brightness" + '.png', random_brightness)

    random_contrast = tf.image.random_contrast(resize, lower=0.4, upper=0.41)
    tf.keras.preprocessing.image.save_img(resize_path + "random_contrast" + '.png', random_contrast)

    random_saturation = tf.image.random_saturation(resize, lower=0.4, upper=0.41)
    tf.keras.preprocessing.image.save_img(resize_path + "random_saturation" + '.png', random_saturation)

    flip_left_right = tf.image.flip_left_right(resize)
    tf.keras.preprocessing.image.save_img(resize_path + "flip_left_right" + '.png', flip_left_right)



#
# total = 0
# for i in range(len(counts)):
#     total+= counts[i]
# for i in range(len(counts)):
#     counts[i] = 1 / tf.math.log(1.02+(counts[i]/total))
#
# for i in range(len(counts)):
#     print(counts[i])
# class_weight = 1 / np.log(1.02 + (frequency / total_frequency))


# class_weight = [
#     2.8543523840037177,
#     1.3501380011580169,
#     4.970495934756391,
#     1.8887107725102839,
#     15.066038400152298,
#     13.837913138463685,
#     12.277478720550748,
#     18.378001839340588,
#     15.71545310733969,
#     2.4582169536505725,
#     12.555716676189862,
#     6.588000419104747,
#     12.318122119569152,
#     19.072811312045705,
#     4.500762248314975,
#     17.856449045143542,
#     18.13687989115995,
#     18.157941306594143,
#     19.434675017727237,
#     16.688571815264762
# ]
