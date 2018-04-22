class Mnist(object):
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('./data/mnist/',one_hot=True)
    def get_train(self, batch_size):
        data = self.mnist.train.next_batch(batch_size)
        return {'input_data': data[0].reshape((-1, 28, 28, 1)), \
                'label': data[1], 'is_training': True}
    def get_val(self, batch_size):
        data = self.mnist.validation.next_batch(batch_size)
        return {'input_data': data[0].reshape((-1, 28, 28, 1)), \
                'label': data[1], 'is_training': True}
    def get_test(self, batch_size):
        data = self.mnist.test.next_batch(batch_size)
        return {'input_data': data[0].reshape((-1, 28, 28, 1)), \
                'label': data[1], 'is_training': True}

