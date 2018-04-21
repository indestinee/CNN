import tensorflow as tf
from IPython import embed


init = [tf.local_variables_initializer(), tf.global_variables_initializer()]
with tf.Session() as sess:
    counter = tf.Variable(0, name='global_step')
    counter = counter + 1
    sess.run(init)
    sess.run(counter)
    embed()

