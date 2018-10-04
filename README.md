# TensorFlow Tutorials


## Getting Started

### Prerequisites
* `TensorFlow` above 1.11
  * [github.com/tensorflow/models](https://github.com/tensorflow/models)
  * `inception_v3` and `vgg_19` pretrained models
* Python above 3.6
  * `numpy`, `matplotlib`, `PIL`
* Jupyter notebook
* OS X and Linux (Not validated on Windows but probably it would work)


## Contents

### TensorFlow Basic Syntax
* Overview and Operations
  - [01.hello.tf.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/01.hello.tf.ipynb)
  - [02.tf.eager.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/02.tf.eager.ipynb)
  - [03.tensorboard.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/03.tensorboard.ipynb)
  - [04.tf.dimension.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/04.tf.dimension.ipynb)
  - [05.tf.Variable.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/05.tf.Variable.ipynb)
  - [06.tf.placeholder.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/06.tf.placeholder.ipynb)
* Managing and `tf.data`
  - [07.tf.train.Saver.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/07.tf.train.Saver.ipynb)
  - [08.tf.cond.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/08.tf.cond.ipynb)
  - [09.tf.control_dependencies.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/09.tf.control_dependencies.ipynb)
  - [10.tf.data.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/01.tf.basic/10.tf.data.ipynb)


### Regression
* Linear Regression
  - [01.1.linear.regression.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/01.1.linear.regression.ipynb)
  - [01.2.linear.regression.with.minibatch.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/01.2.linear.regression.with.minibatch.ipynb)
  - [02.1.linear.regression.3rd.order.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/02.1.linear.regression.3rd.order.ipynb)
  - [02.2.linear.regression.3rd.order.with.minibatch.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/02.2.linear.regression.3rd.order.with.minibatch.ipynb)
* Logistic Regression
  - [03.1.mnist.softmax.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/03.1.mnist.softmax.ipynb)
  - [03.2.mnist.softmax.with.tf.data.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/03.2.mnist.softmax.with.tf.data.ipynb)
  - [04.mnist.summary.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/02.regression/04.mnist.summary.ipynb)


### Convolutional Neural Networks
* Simple CNN model (LeNet5) and regularization & batch norm
  - [01.1.mnist.deep.with.estimator.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/01.1.mnist.deep.with.estimator.ipynb)
  - [01.2.mnist.deep.with.tf.data.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/01.2.mnist.deep.with.tf.data.ipynb)
  - [02.mnist.deep.slim.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/02.mnist.deep.slim.ipynb)
  - [03.mnist.slim.options.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/03.mnist.slim.options.ipynb)
  - [04.mnist.slim.arg.scope.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/04.mnist.slim.arg.scope.ipynb)
* Advanced CNN model (Cifar10) and data augmentation
  - [05.cnn.cifar10.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/05.cnn.cifar10.ipynb)
  - [06.cnn.cifar10.regularization.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/06.cnn.cifar10.regularization.ipynb)
  - [07.cnn.cifar10.data.augmentation.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/07.cnn.cifar10.data.augmentation.ipynb)
* Pretrained CNN models
  - [08.vgg19.slim.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/08.vgg19.slim.ipynb)
  - [09.inception_v3.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/09.inception_v3.ipynb)
* Transfer learning and `tfrecords` format
  - [10.tfrecords.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/10.tfrecords.ipynb)
  - [11.transfer.learning.with.inception_v3.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/03.cnn/11.transfer.learning.with.inception_v3.ipynb)


### Recurrent Neural Networks
* Usage for sequence data
  - [01.ready.for.sequence.data.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/01.ready.for.sequence.data.ipynb)
  - [02.rnn.basic.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/02.rnn.basic.ipynb)
* Sequence classification (many to one classification)
  - [03.01.sequence.classification.RNN.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.01.sequence.classification.RNN.ipynb)
  - [03.02.sequence.classification.GRU.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.02.sequence.classification.GRU.ipynb)
  - [03.03.sequence.classification.LSTM.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.03.sequence.classification.LSTM.ipynb)
  - [03.04.sequence.classification.biRNN.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.04.sequence.classification.biRNN.ipynb)
  - [03.05.sequence.classification.biLSTM.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.05.sequence.classification.biLSTM.ipynb)
  - [03.06.sequence.classification.Multi.RNN.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.06.sequence.classification.Multi.RNN.ipynb)
  - [03.07.sequence.classification.Multi.LSTM.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.07.sequence.classification.Multi.LSTM.ipynb)
  - [03.08.sequence.classification.Multi.RNN.dropout.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.08.sequence.classification.Multi.RNN.dropout.ipynb)
  - [03.09.sequence.classification.Multi.LSTM.dropout.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.09.sequence.classification.Multi.LSTM.dropout.ipynb)
  - [03.10.sequence.classification.Multi.biRNN.dropout.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.10.sequence.classification.Multi.biRNN.dropout.ipynb)
  - [03.11.sequence.classification.Multi.biLSTM.dropout.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/03.11.sequence.classification.Multi.biLSTM.dropout.ipynb)
* Sequence to sequence classification (many to many classification)
  - [04.01.seq2seq.classification.RNN.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.01.seq2seq.classification.RNN.ipynb)
  - [04.02.seq2seq.classification.GRU.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.02.seq2seq.classification.GRU.ipynb)
  - [04.03.seq2seq.classification.LSTM.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.03.seq2seq.classification.LSTM.ipynb)
  - [04.04.seq2seq.classification.biRNN.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.04.seq2seq.classification.biRNN.ipynb)
  - [04.05.seq2seq.classification.biLSTM.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.05.seq2seq.classification.biLSTM.ipynb)
  - [04.06.seq2seq.classification.Multi.RNN.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.06.seq2seq.classification.Multi.RNN.ipynb)
  - [04.07.seq2seq.classification.Multi.LSTM.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.07.seq2seq.classification.Multi.LSTM.ipynb)
  - [04.08.seq2seq.classification.Multi.RNN.dropout.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.08.seq2seq.classification.Multi.RNN.dropout.ipynb)
  - [04.09.seq2seq.classification.Multi.LSTM.dropout.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/04.rnn/04.09.seq2seq.classification.Multi.LSTM.dropout.ipynb)


### Generative Models
* Generative Adversarial Networks
  - [01.dcgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/tensorflow.tutorials/tree/master/tf.version.1/05.generative_models/01.dcgan.ipynb)





## Author
Il Gu Yi
