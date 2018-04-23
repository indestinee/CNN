from dataset import *
import common

class Pipe(object):
    def __init__(self, name, port):
        pass

class Input_pipe(Pipe):

    def send(self):
        pass


class Output_pipe(Pipe):

    def receive(self):
        pass


# cfg = common.mnist_config()
# dp = Mnist()
    
cfg = common.flower_config()
dp = Flowers(cfg.input_shape, train=7, val=3, test=2)

