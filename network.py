import tensorflow as tf
import os
from time import time
from progressbar import ProgressBar

import model, data_provider
cfg = data_provider.cfg
eps = 1e-10
inf = 1000000000

def show_epoch(progress, step, train_loss, val_loss):# {{{
    print('[LOG] loss #%d: train %.3f val %.3f' % \
            (step, train_loss, val_loss))
    progress.update(step)
# }}}

class Network(object):
    def __init__(self, args):# {{{
        self.args = args
        self.model_path = os.path.join(args.logdir, 'models')
        self.logdir = os.path.join(args.logdir, 'train_events')
        self.placeholder = {}
        self.net()
        self.progress = ProgressBar(\
                maxval=args.step if args.step >= 0 else inf)
    # }}}
    def feed_dict(self, data):# {{{
        #   build feed_dict
        #   change dict{str->data} to dict{tensor->data}
        return {self.placeholder[key]: value \
                for key, value in data.items()}
    # }}}



    def func_result(self, x):# {{{
        '''
            input:
                x: output of net
            output:
                result (label, prediction or sth.) of final result
        '''
        with tf.name_scope('prediction'):
            _class = tf.argmax(x, axis=1)
            _score = tf.nn.softmax(x, name="softmax_tensor")
            return _class, _score
    # }}}
    def func_loss(self, x, y):# {{{
        '''
            input:
                x: output of net
                y: ground truth
            output:
                loss tensor (scalar)
        '''
        with tf.name_scope('loss'):
            x = tf.nn.softmax(x, dim=1, name='softmax')
            return -tf.reduce_mean(y*tf.log(x+eps))
    # }}}
    def hidden_layer(self, x, is_training):# {{{
        '''
            hidden layer of the net
        '''
        with tf.name_scope('hidden_layer') as name_scope:
            return model.alexnet_model(x, is_training)
    # }}}
    def net(self):# {{{
        #   input layer
        with tf.name_scope('input_layer') as name:
            x = tf.placeholder('float', shape=[None, *cfg.input_shape], \
                    name='input_data')  #   input data
            y = tf.placeholder('float', shape=[None, *cfg.output_shape], \
                    name='label')       #   ground truth
            x = x / 255 #   [0, 1]

            #   specify the status of net
            is_training = tf.placeholder(tf.bool, name='is_training')

            #   prepare for function feed_dict
            self.placeholder['input_data'] = x
            self.placeholder['label'] = y
            self.placeholder['is_training'] = is_training

        #   save input data in tensorboard
        tf.summary.image('input_data', x)   

        #   hidden layer
        x = self.hidden_layer(x, is_training)

        #   dense layer
        x = tf.layers.dense(inputs=x, units=cfg.output_shape[0], \
                name='logits')
        
        #   loss function
        cross_entropy_loss = self.func_loss(x, y)
        #   save loss for tensorboard
        tf.summary.scalar('cross entropy', cross_entropy_loss)

        variables = \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        l2_regularizer = tf.contrib.layers.l2_regularizer(\
                self.args.weight_decay)
        l2_regularization_loss = tf.reduce_sum(\
                [l2_regularizer(variable) for variable in variables])
        tf.summary.scalar('L2 regularization', l2_regularization_loss)

        self.loss_op = cross_entropy_loss + \
                l2_regularization_loss
        tf.summary.scalar('total loss', self.loss_op)
        
        #   optimizer & learning rate
        # self.optimizer = tf.train.AdamOptimizer(\
                # learning_rate=self.args.learning_rate)
        self.optimizer = tf.train.MomentumOptimizer(\
                self.args.learning_rate, 0.9)
        self.train_op = self.optimizer.minimize(self.loss_op)
        
        #   prediction result
        self.predict_op = self.func_result(x)


        #   merge all summary for tensorboard
        self.summary = tf.summary.merge_all()
    # }}}

    def train(self):# {{{
        #   initializer all variables
        init = tf.global_variables_initializer()

        #   saver to save/restore model to/from path
        saver = tf.train.Saver()

        #   tf session
        with tf.Session() as sess:

            #   run initalizer for all variables
            sess.run(init)

            #   load pre-trained model if needed
            if self.args.checkpoint:
                print('[LOG] loading model from', self.args.checkpoint)
                saver.restore(sess, self.args.checkpoint)

            #   train/val writer of tensorboard
            train_writer = tf.summary.FileWriter(\
                    os.path.join(self.logdir, 'train'), sess.graph)
            val_writer = tf.summary.FileWriter(\
                    os.path.join(self.logdir, 'val'))
            
            print('[LOG] training started ..')
            step, self.start_time = 0, time()
            while step < self.args.step or self.args.step == -1:
                step += 1

                #   get train data
                train_batch = data_provider.dp.get_train(\
                        self.args.batch_size)

                #   get training summary, loss value, 
                #   and parameters updating
                train_summary, loss, _ = sess.run(
                    [self.summary, self.loss_op, self.train_op], 
                    feed_dict=self.feed_dict(train_batch)
                )

                #   recorde summary of each step for tensorboard
                train_writer.add_summary(train_summary, step)

                #   save model
                if step % self.args.save_step == 0:
                    name = 'model_%d.pkl' % step
                    saver.save(sess, \
                            os.path.join(self.model_path, name))

                #   display train/val loss
                if step == 1 or step % self.args.val_step == 0:
                    val_batch = data_provider.dp.get_val(\
                            self.args.batch_size)
                    val_summary, val_loss = sess.run(\
                            [self.summary, self.loss_op], \
                            feed_dict=self.feed_dict(val_batch))
                    val_writer.add_summary(val_summary, step)
                    show_epoch(self.progress, step, loss, val_loss)
        print('[LOG] Training finished ..')
            
    # }}}
