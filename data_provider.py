from dataset import *
import common

class Pipe(object):
    def __init__(self, name, port):
        pass

    def send(self):
        pass

    def receive(self):
        pass

# cfg = common.mnist_config()
# dp = Mnist()
    
cfg = common.flower_config()
dp = Flowers(cfg.input_shape, train=7, val=3, test=2)

