import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13


# sound_shape (60,41,2)


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


import os


# max 0 car: 10380
# max 1 dog: 10380
def extract_features(sound_dir, file_ext="*.wav", bands=60, frames=41):
    max_len = 10380
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    dirs = glob.glob(os.path.join(sound_dir, "*"))
    dirs.sort()
    for i, label_dir in enumerate(dirs):
        label = i
        cnt = 0
        for j, fn in enumerate(glob.glob(os.path.join(label_dir, file_ext))):
            sound_clip, s = librosa.load(fn)
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.amplitude_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)
                    cnt += 1
            print(j)
            # if cnt > 25:
            #     break
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    print(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    print(one_hot_encode.shape)
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def apply_convolution(x, kernel_size, num_channels, depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')


frames = 41
bands = 60

feature_size = 2460  # 60x41
num_labels = 2
num_channels = 2

batch_size = 64
kernel_size = 30
depth = 30
num_hidden = 200

learning_rate = 0.01

sound_X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels], name="sound_x")
sound_Y = tf.placeholder(tf.float32, shape=[None, num_labels], name="sound_y")

sound_net = tf.layers.conv2d(sound_X, num_hidden, kernel_size, activation=tf.nn.relu)
# sound_net = tf.layers.conv2d(sound_net, num_hidden, 5, activation=tf.nn.relu)
# sound_net = tf.layers.conv2d(sound_net, num_hidden, 3, activation=tf.nn.relu)
sound_net = tf.reduce_mean(sound_net, [1, 2], name='last_pool')
print(sound_net)

image_X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="image_x")
image_Y = tf.placeholder(tf.float32, shape=[None, num_labels], name="image_y")

image_net = tf.layers.conv2d(image_X, 64, 7, 4, activation=tf.nn.relu)
image_net = tf.layers.max_pooling2d(image_net, 3, 2)
image_net = tf.layers.conv2d(image_net, 128, 5, activation=tf.nn.relu)
image_net = tf.layers.max_pooling2d(image_net, 3, 2)
image_net = tf.layers.conv2d(image_net, num_hidden, 3, padding="same", activation=tf.nn.relu)
image_net = tf.layers.conv2d(image_net, num_hidden, 3, padding="same", activation=tf.nn.relu)
image_net = tf.layers.conv2d(image_net, num_hidden, 3, padding="same", activation=tf.nn.relu)
image_net = tf.reduce_mean(image_net, [1, 2], name='last_pool')
print(image_net)
net = tf.concat([image_net, sound_net], axis=0)

net = tf.layers.dense(net, 256, activation=tf.nn.relu)

net = tf.layers.dense(net, 128, activation=tf.nn.relu)
net = tf.layers.dense(net, 64, activation=tf.nn.relu)
logits = tf.layers.dense(net, num_labels)
logits_l2norm = tf.nn.l2_normalize(logits, axis=1)
y_ = tf.nn.softmax(logits)
Y = tf.concat([image_Y, sound_Y], axis=0)
loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits),
    name="softmax_cross_entropy")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(logits)
# sys.exit()
if not os.path.exists("image_dataset.npy"):
    image_dir = "D:\\data\\mixnet\\dog_and_car_sound_and_image\\image"
    dirs = glob.glob(os.path.join(image_dir, "*"))
    from PIL import Image

    im_size = 224
    image_dataset = None
    image_labels = []
    dirs.sort()
    for i, d in enumerate(dirs):
        imgs = glob.glob(os.path.join(d, "*"))
        for j, file_path in enumerate(imgs):
            print(file_path)
            image_obj = Image.open(file_path)
            w = image_obj.size[0]
            h = image_obj.size[1]
            if w < im_size and h < im_size:
                continue
            if w > h:
                image_obj = image_obj.resize([round(w * (im_size / h)), im_size])
            elif w < h:
                image_obj = image_obj.resize([im_size, round(h * (im_size / w))])
            else:
                image_obj = image_obj.resize([im_size, im_size])
            width, height = image_obj.size  # Get dimensions
            left = (width - im_size) / 2
            top = (height - im_size) / 2
            right = (width + im_size) / 2
            bottom = (height + im_size) / 2

            image_obj = image_obj.crop((left, top, right, bottom))

            img_array = np.array(image_obj)
            print(img_array.shape)
            print(len(img_array.shape))
            if len(img_array.shape) < 3 or img_array.shape[2] != 3:
                print("skip")
                continue
            img_array = np.expand_dims(img_array, axis=0)

            if image_dataset is None:
                image_dataset = img_array
            else:
                image_dataset = np.concatenate((image_dataset, img_array), axis=0)

            image_labels.append(i)
            # if j > batch_size:
            #     print(image_dataset.shape)
            #     break

    image_labels = one_hot_encode(image_labels)
    np.save("image_dataset", image_dataset)
    np.save("image_labels", image_labels)
else:
    image_dataset = np.load("image_dataset.npy")
    image_labels = np.load("image_labels.npy")

from sklearn.utils import shuffle

idxs = shuffle(list(range(len(image_dataset))), random_state=100)
t_cnt = int(len(idxs) * 0.9)

img_trx = image_dataset[idxs[:t_cnt]]
img_tex = image_dataset[idxs[t_cnt:]]
img_try = image_labels[idxs[:t_cnt]]
img_tey = image_labels[idxs[t_cnt:]]

if not os.path.exists("sound_dataset.npy"):
    sound_dir = "D:\\data\\mixnet\\dog_and_car_sound_and_image\\sound"
    sound_dataset, sound_labels = extract_features(sound_dir, bands=60, frames=41)
    sound_labels = one_hot_encode(sound_labels)
    print(type(sound_labels), sound_labels.shape)
    print(type(sound_dataset), sound_dataset.shape)

    np.save("sound_dataset", sound_dataset)
    np.save("sound_labels", sound_labels)
else:
    sound_dataset = np.load("sound_dataset.npy")
    sound_labels = np.load("sound_labels.npy")
print("dd", len(image_dataset))
print(len(sound_dataset))
# f_shapes = list(np.shape(features))
# f_shapes[-1] = 1
# features = np.concatenate((features, np.zeros(f_shapes)), axis=3)
# print(type(features), features.shape)
# print(np.unique(labels, axis=1))

idxs = shuffle(list(range(len(sound_dataset))), random_state=100)
t_cnt = int(len(idxs) * 0.9)
sound_trx = sound_dataset[idxs[:t_cnt]]
sound_tex = sound_dataset[idxs[t_cnt:]]
sound_try = sound_labels[idxs[:t_cnt]]
sound_tey = sound_labels[idxs[t_cnt:]]

# training_iterations = 1600
training_iterations = 5000
# random sampling
with tf.Session() as session:
    tf.global_variables_initializer().run()
    step = 0
    for itr in range(training_iterations):
        rnd_idx = random.sample(range(len(img_trx)), batch_size)
        img_batch_x = img_trx[rnd_idx]
        img_batch_y = img_try[rnd_idx]

        rnd_idx = random.sample(range(len(sound_trx)), batch_size)
        sound_batch_x = sound_trx[rnd_idx]
        sound_batch_y = sound_try[rnd_idx]
        step += 1

        _, c, train_accuracy, sm = session.run([optimizer, loss_op, accuracy, y_],
                                               feed_dict={image_X: img_batch_x, image_Y: img_batch_y,
                                                          sound_X: sound_batch_x,
                                                          sound_Y: sound_batch_y})
        if step % 10 == 0:
            print(step, c, train_accuracy)

        if step % 100 == 0:
            test_steps = len(img_tey) // batch_size
            for i in range(test_steps):
                print('Test accuracy: ', session.run([accuracy, loss_op],
                                                     feed_dict={
                                                         image_X: img_tex[i * batch_size:i * batch_size + batch_size],
                                                         image_Y: img_tey[i * batch_size:i * batch_size + batch_size],
                                                         sound_X: sound_tex[i * batch_size:i * batch_size + batch_size],
                                                         sound_Y: sound_tey[
                                                                  i * batch_size:i * batch_size + batch_size]}))

    # test_steps = len(img_tey) // batch_size
    # test_results = []
    # for i in range(test_steps):
    #     results = session.run([accuracy, loss_op, logits_l2norm],
    #                           feed_dict={
    #                               image_X: img_tex[i * batch_size:i * batch_size + batch_size],
    #                               image_Y: img_tey[i * batch_size:i * batch_size + batch_size],
    #                               sound_X: sound_tex[i * batch_size:i * batch_size + batch_size],
    #                               sound_Y: sound_tey[
    #                                        i * batch_size:i * batch_size + batch_size]})
    #     test_results.append(results[2])
    # image_embedding = np.concatenate(
    #     (test_results[0][:batch_size // 2], test_results[1][:batch_size // 2], test_results[2][
    #                                                                            :batch_size // 2]), axis=0)
    # sound_embedding = np.concatenate(
    #     (test_results[0][batch_size // 2:], test_results[1][batch_size // 2:], test_results[2][
    #                                                                            batch_size // 2:]), axis=0)
    # image_labeling = img_tey[:batch_size * 3]
    # sound_labeling = sound_tey[:batch_size * 3]
    #
    # print(len(image_embedding))
    # print(len(sound_embedding))
    # print(len(image_labeling))
    # print(len(sound_labeling))
    #
    # from sklearn.decomposition import PCA
    #
    # pca = PCA(n_components=2)
    # image_pca = pca.fit_transform(image_embedding)
    # sound_pca = pca.fit_transform(sound_embedding)
    # fig = plt.figure(figsize=(15, 10))
    # plt.plot(cost_history)
    # plt.axis([0, training_iterations, 0, np.max(cost_history)])
    # plt.show()
