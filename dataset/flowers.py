from utils import Cache
import numpy as np

class Sample(object):
    def __init__(self):
        self.image = []
        self.label = []

    #   to numpy.ndarray type
    def to_np(self):
        self.image = np.array(self.image)
        self.label = np.array(self.label)

class Flowers(object):
    def __init__(self, shape, train=7, val=3, test=2):
        self.labels = 17   
        
        #   read data from cache or origin tgz
        cache = Cache('./cache/17flowers{}.pkl'.format(shape))
        self.data = cache.load()
        if not self.data:
            from IO import load_flowers17
            self.data = load_flowers17('./data/17flowers.tgz', shape)
            cache.save(self.data)
        
        self.train, self.val, self.test = Sample(), Sample(), Sample()
        self.split_data(train, val, test)

        self.train.to_np()
        self.val.to_np()
        self.test.to_np()

    def split_data(self, a, b, c):
        s = a + b + c
        a, b, c = a / s, (a + b) / s, 1
        for label, imgs in enumerate(self.data):
            #   random shuffle
            np.random.shuffle(imgs)
            _a = int(round(a * len(imgs)))
            _b = int(round(b * len(imgs)))
            _c = int(round(c * len(imgs)))
            self.train.image += imgs[:_a]
            self.val.image += imgs[_a:_b]
            self.test.image += imgs[_b:]
            
            #   labels = [0, 0, ..., 1, ..., 0, 0]
            labels = np.zeros(self.labels)
            labels[label] = 1

            self.train.label += [labels for i in range (_a)]
            self.val.label += [labels for i in range (_b - _a)]
            self.test.label += [labels for i in range (_c - _b)]

        self.train_list = range(len(self.train.image))
        self.val_list = range(len(self.val.image))
        self.test_list = range(len(self.test.image))

    def get_train(self, batch_size):
        indexes = np.random.choice(self.train_list, batch_size)
        return {
            'input_data': np.array(self.train.image[indexes]),
            'label': np.array(self.train.label[indexes]),
            'is_training': True
        }
    def get_val(self, batch_size):
        indexes = np.random.choice(self.val_list, batch_size)
        return {
            'input_data': np.array(self.val.image[indexes]),
            'label': np.array(self.val.label[indexes]),
            'is_training': False
        }
    def get_test(self, batch_size):
        indexes = np.random.choice(self.test_list, batch_size)
        return {
            'input_data': np.array(self.test.image[indexes]),
            'label': np.array(self.test.label[indexes]),
            'is_training': False
        }


