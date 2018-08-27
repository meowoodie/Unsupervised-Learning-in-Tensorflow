Deep Learning Models in tensorflow
---

### Introduction
It is a repository for compiling most of my deep learning implementations in Tensorflow, which includes Stacked Denoising Autoencoders, and so on.

### Stacked Denoising Autoencoders
Below is a simple example for fitting a vanilla Denoising Autoencoder. Also the `dA` provides methods for reconstructing and calculating the hidden output (encode). Moreover, we use the MNIST dataset for the unit testing in order to compare the performance.
```python
# An unittest on MNIST data for Denoising Autoencoder

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
train_x = np.vstack([img.reshape(-1,) for img in mnist.train.images])
train_y = np.reshape(mnist.train.labels, (train_x.shape[0], 1))
test_x  = np.vstack([img.reshape(-1,) for img in mnist.test.images])
test_y  = np.reshape(mnist.test.labels, (test_x.shape[0], 1))
test_sample = test_x[0:10]

n_visible = train_x.shape[1]
n_hidden  = 700

from utils import show_mnist_images
with tf.Session() as sess:
    # Single Denoising Autoencoder
    # - This is the best params that I can find for reconstructing the
    #   best quality of images
    da = dA('test', n_visible, n_hidden,
             keep_prob=0.05, lr=0.01, batch_size=1000, n_epoches=25, corrupt_lv=0.1)
    da.fit(sess, train_x, test_x)
    reconstructed_x = da.get_reconstructed_x(sess, test_sample)
    # Plot reconstructed mnist figures
    show_mnist_images(reconstructed_x)
    show_mnist_images(test_sample)
```
Stacked Denoising Autoencoder can be also used in a similar fashion, except for fitting process. In terms of fitting Stacked Denoising Autoencoder, you have the options to `pretrain`, or `finetune` the model, or do the both.
```python
with tf.Session() as sess:
    # Stacked Denoising Autoencoder
    sda = SdA(n_visible=n_visible, hidden_layers_sizes=[700, 600],
              keep_prob=0.05, pretrain_lr=1e-1, finetune_lr=1e-1,
              batch_size=1000, n_epoches=20, corrupt_lvs=[0.1, 0.1])
    sda.pretrain(sess, train_x, test_x, pretrained=False)
    sda.finetune(sess, train_x, train_y, test_x, test_y, pretrained=True)
    reconstructed_x = sda.get_reconstructed_x(sess, test_sample)
    # Plot reconstructed mnist figures
    show_mnist_images(reconstructed_x)
    show_mnist_images(test_sample)
```
Here console output has also been attached for your reference. The log information presents the process of the convergence in pretrain and finetune.
```bash
[2018-08-26T23:19:06.793786-04:00] --- Pre-train Phase ---
[2018-08-26T23:19:06.794408-04:00] *** Layer (stacked_layer_0_dA) ***
[2018-08-26T23:19:06.794665-04:00] visible size: 784, hidden size: 700, corruption level: 0.100000
[2018-08-26T23:19:24.002294-04:00] Epoch 1
[2018-08-26T23:19:24.002424-04:00] Training loss:	0.140033
[2018-08-26T23:19:24.002500-04:00] Testing loss:	0.138879
    ... ...
[2018-08-26T23:24:27.439909-04:00] Epoch 20
[2018-08-26T23:24:27.440036-04:00] Training loss:	0.079208
[2018-08-26T23:24:27.440108-04:00] Testing loss:	0.078656
[2018-08-26T23:24:27.441179-04:00] *** Layer (stacked_layer_1_dA) ***
[2018-08-26T23:24:27.441255-04:00] visible size: 700, hidden size: 600, corruption level: 0.100000
[2018-08-26T23:24:44.875772-04:00] Epoch 1
[2018-08-26T23:24:44.875898-04:00] Training loss:	0.239635
[2018-08-26T23:24:44.875973-04:00] Testing loss:	0.239351
    ... ...
[2018-08-26T23:30:19.810884-04:00] Epoch 20
[2018-08-26T23:30:19.811106-04:00] Training loss:	0.131480
[2018-08-26T23:30:19.811233-04:00] Testing loss:	0.131739
[2018-08-26T23:30:19.813579-04:00] --- Fine-tune Phase ---
[2018-08-26T23:30:34.394329-04:00] Epoch 1
[2018-08-26T23:30:34.394462-04:00] Training loss:	8.081374
[2018-08-26T23:30:34.394598-04:00] Testing loss:	8.032670
[2018-08-26T23:30:34.394721-04:00] Training accuracy:	0.301782
[2018-08-26T23:30:34.394865-04:00] Testing accuracy:	0.296291
    ... ...
[2018-08-26T23:34:46.131152-04:00] Epoch 20
[2018-08-26T23:34:46.131281-04:00] Training loss:	1.418032
[2018-08-26T23:34:46.131354-04:00] Testing loss:	1.462582
[2018-08-26T23:34:46.131465-04:00] Training accuracy:	0.779636
[2018-08-26T23:34:46.131572-04:00] Testing accuracy:	0.778764
```
