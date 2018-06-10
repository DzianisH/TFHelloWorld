import tensorflow as tf
from tensorflow import layers
from tensorflow.examples.tutorials.mnist import input_data

import utils

data = input_data.read_data_sets('data/MNIST', one_hot=True)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10
max_epochs = 101
train_batch_size = 100
test_batch_size = 500
max_epoch_without_improvement = 25

# utils.plot_images(data.test.images[:9], img_shape, data.test.labels[:9])

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_class = tf.argmax(y_true, axis=1, output_type=tf.int32)

model = x_image
model = model * 2. - 1.  # normalize

max = tf.reduce_max(model)
min = tf.reduce_min(model)
mean = tf.reduce_mean(model)

with utils.create_session() as sess:
    x_data, y_data = data.test.next_batch(test_batch_size)
    min, mean, max = sess.run([min, mean, max], feed_dict={x: x_data, y_true: y_data})
    print("input_data_format:\n\t\tmin: {:.2f} mean: {:.2f} max: {:.2f}".format(min, mean, max))

model = layers.conv2d(inputs=model, name="conv1", padding='same', filters=32, kernel_size=3, activation=tf.nn.relu)
model = layers.conv2d(inputs=model, name="conv2", padding='same', filters=64, kernel_size=3, activation=tf.nn.relu)
model = layers.max_pooling2d(inputs=model, name="pool1", strides=2, pool_size=2)
model = layers.conv2d(inputs=model, name="conv3", padding='same', filters=64, kernel_size=3, activation=tf.nn.relu)
model = layers.dropout(inputs=model, rate=0.25)
model = layers.flatten(model)
model = layers.dense(inputs=model, name='dense1', units=128, activation=tf.nn.relu)
model = layers.dropout(inputs=model, rate=0.5)
model = layers.dense(inputs=model, name='output', units=num_classes, activation=None)
logits = model
y_pred = tf.nn.softmax(logits=logits)
y_pred_class = tf.argmax(y_pred, axis=1, output_type=tf.int32)

guesses = tf.equal(y_pred_class, y_true_class)
guesses = tf.cast(guesses, tf.int16)
guesses = tf.reduce_sum(guesses)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

optimazer = tf.train.MomentumOptimizer(learning_rate=3e-3, momentum=0.9).minimize(loss)
# merged_summary_op = tf.summary.merge_all()
# weights_conv1 = utils.get_weight_variable('conv1')
# weights_conv2 = utils.get_weight_variable('conv2')

with utils.create_session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    best_test_accuracy = -1
    best_test_accuracy_epoch = 0
    for epoch in range(max_epochs):
        train_accuracy, train_loss, train_time_spent = utils.train(data.train, train_batch_size,
                                                                   lambda batch_x, batch_y: sess.run(
                                                                       [optimazer, guesses, loss],
                                                                       feed_dict={x: batch_x, y_true: batch_y}))
        test_accuracy, test_time_spent = utils.test(data.test, test_batch_size,
                                                    lambda batch_x, batch_y: sess.run([guesses], feed_dict={x: batch_x,
                                                                                                            y_true: batch_y}))

        print("epoch: {:02d} train_accuracy: {:3.2f}% train_loss: {:2.3f} time: {:2.2f}s\n"
              "           test_accuracy: {:3.2f}%                     time: {:2.2f}s".format(epoch + 1, train_accuracy,
                                                                                             train_loss,
                                                                                             train_time_spent,
                                                                                             test_accuracy,
                                                                                             test_time_spent))
        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_accuracy_epoch = epoch
            print("           new best accuracy")
            saver.save(sess, "data/models/mnist_{:.3f}".format(test_accuracy), latest_filename="best")

        if best_test_accuracy_epoch + max_epoch_without_improvement < epoch:
            print("\texceeded maximum amount of epoch without score improvement (max_dist: {}, current_epoch: {}, "
                  "best_epoch: {})".format(max_epoch_without_improvement, epoch, best_test_accuracy_epoch))
            break

    saver.save(sess, "data/modes/mnist_{:.3f}".format(best_test_accuracy), latest_filename="last")
    print("best_test_accuracy: {:.2f}%".format(best_test_accuracy))
