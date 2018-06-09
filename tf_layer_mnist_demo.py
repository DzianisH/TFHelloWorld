import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.examples.tutorials.mnist import input_data
import time

data = input_data.read_data_sets('data/MNIST', one_hot=True)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10
epochs = 12
batch_size = 128

# utils.plot_images(data.test.images[:9], img_shape, data.test.labels[:9])

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_class = tf.argmax(y_true, dimension=1)

model = x_image
model = layers.conv2d(inputs=model, name="conv1", padding='same', filters=32, kernel_size=3, activation=tf.nn.relu)
model = layers.conv2d(inputs=model, name="conv2", padding='same', filters=64, kernel_size=3, activation=tf.nn.relu)
model = layers.max_pooling2d(inputs=model, name="pool1", strides=2, pool_size=2)
model = layers.dropout(inputs=model, rate=0.25)
model = layers.flatten(model)
model = layers.dense(inputs=model, name='dense1', units=128, activation=tf.nn.relu)
model = layers.dropout(inputs=model, rate=0.5)
model = layers.dense(inputs=model, name='output', units=num_classes, activation=None)
logits = model
y_pred = tf.nn.softmax(logits=logits)
y_pred_class = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

optimazer = tf.train.AdagradOptimizer(learning_rate=1e-4).minimize(loss)
# merged_summary_op = tf.summary.merge_all()

# weights_conv1 = utils.get_weight_variable('conv1')
# weights_conv2 = utils.get_weight_variable('conv2')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        time_delay = time.time()
        epoch_loss_value = 0
        for step in range(0, data.train.num_examples, batch_size):
            x_batch, y_true_batch = data.train.next_batch(batch_size)
            _, batch_loss_value = sess.run([optimazer, loss],
                                                         feed_dict={x: x_batch, y_true: y_true_batch})
            epoch_loss_value += batch_loss_value

        time_delay = time.time() - time_delay
        print("epoch: {:02d} train loss: {:.3f} dt: {:.2f}s".format(epoch, epoch_loss_value, time_delay))