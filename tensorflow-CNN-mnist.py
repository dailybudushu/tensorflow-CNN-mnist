import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('G:\\我的文档\\data_set\\mnist', one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, name='initial_W')
    return tf.Variable(initial, name='W')


def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape, name='initial_b')
    return tf.Variable(initial, name='b')


def convolve_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
with tf.name_scope('input_label'):
    y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='label')

with tf.name_scope('input_image'):
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='image')

with tf.name_scope('convolve_layer1'):
    w_convolve_1 = weight_variable(shape=[5, 5, 1, 32])
    tf.summary.histogram(name='w_convolve_1', values=w_convolve_1)
    b_convolve_1 = bias_variable(shape=[32])
    tf.summary.histogram(name='b_convolve_1', values=b_convolve_1)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_convolve_1 = tf.nn.relu(convolve_2d(x_image, w_convolve_1) + b_convolve_1)
    h_pool_1 = max_pool_2x2(h_convolve_1)

with tf.name_scope('convolve_layer2'):
    w_convolve_2 = weight_variable(shape=[5, 5, 32, 64])
    tf.summary.histogram(name='w_convolve_2', values=w_convolve_2)
    b_convolve_2 = bias_variable(shape=[64])
    tf.summary.histogram(name='b_convolve_2', values=b_convolve_2)
    h_convolve_2 = tf.nn.relu(convolve_2d(h_pool_1, w_convolve_2) + b_convolve_2)
    h_pool_2 = max_pool_2x2(h_convolve_2)

with tf.name_scope('fully_connected'):
    w_fc_1 = weight_variable(shape=[7 * 7 * 64, 1024])
    tf.summary.histogram(name='w_fc_1', values=w_fc_1)
    b_fc_1 = bias_variable([1024])
    tf.summary.histogram(name='b_fc_1', values=b_fc_1)
    h_pool2_flat = tf.reshape(h_pool_2, shape=[-1, 7 * 7 * 64])
    h_fc_1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc_1) + b_fc_1)
    h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob=keep_prob)

with tf.name_scope('prediction'):
    w_fc_2 = weight_variable([1024, 10])
    tf.summary.histogram(name='w_fc_2', values=w_fc_2)
    b_fc_2 = bias_variable([10])
    tf.summary.histogram(name='b_fc_2', values=b_fc_2)
    y_convolve = tf.nn.softmax(tf.matmul(h_fc_1_drop, w_fc_2) + b_fc_2, name='prediction')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_convolve))
    tf.summary.scalar(name='corss_entropy', tensor=cross_entropy)

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
sess = tf.Session()

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_convolve, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'), name='accuracy')
    tf.summary.scalar(name='corss_entropy', tensor=cross_entropy)
writer = tf.summary.FileWriter("C:\\Users\\Lenovo\\Desktop\\mnist")
merged_summary = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
writer.add_graph(sess.graph)
for i in range(1200):
    batch = mnist.train.next_batch(50)
    if i % 50 == 0:
        s = sess.run(merged_summary, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        writer.add_summary(s, i)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
