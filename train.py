import argparse

from network import Network

def get_args():# {{{
    parser = argparse.ArgumentParser(description='Training of CNN')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
    parser.add_argument('-s', '--step', type=int, default=-1, \
            help='total training step, -1 means infinite')
    parser.add_argument('-b', '--batch-size', type=int, default=16, \
            help='batch size of each step')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, \
            help='path to load pretrained model, default for no loading')
    parser.add_argument('--val-step', type=int, default=10, \
            help='how many training steps before each validation')
    parser.add_argument('--save-step', type=int, default=1000,
            help='how many training steps before each model-saving')
    parser.add_argument('--logdir', type=str, \
            default='train_log', \
            help='path to save training logs and models')
    return parser.parse_args()
# }}}


if __name__ == '__main__':
    args = get_args()   
    print(args)

    network = Network(args)
    network.train()
