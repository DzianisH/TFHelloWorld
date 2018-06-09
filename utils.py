import matplotlib.pyplot as plt
import tensorflow as tf
import time

def get_weight_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable(layer_name)
    return variable


def plot_images(images, img_shape, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def train(data, batch_size, executor):
    time_delay = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    for step in range(0, data.num_examples, batch_size):
        x_batch, y_true_batch = data.next_batch(batch_size)
        _, batch_guesses, batch_loss = executor(x_batch, y_true_batch)
        epoch_accuracy += batch_guesses
        epoch_loss += batch_loss
    epoch_accuracy = epoch_accuracy * 100. / data.num_examples

    time_delay = time.time() - time_delay

    return epoch_accuracy, epoch_loss, time_delay

def test(data, batch_size, executor):
    time_delay = time.time()
    epoch_accuracy = 0
    for step in range(0, data.num_examples, batch_size):
        x_batch, y_true_batch = data.next_batch(batch_size)
        epoch_accuracy += executor(x_batch, y_true_batch)[0]
    epoch_accuracy = epoch_accuracy * 100. / data.num_examples

    time_delay = time.time() - time_delay
    return epoch_accuracy, time_delay


def create_session(gpu_mem_fraction=0.73):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction

    return tf.Session(config=config)