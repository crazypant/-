#classification 分类问题
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
def add_layer(inputs,in_size,out_size,activate_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activate_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activate_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={
        xs:v_xs
    })
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={
        xs:v_xs,ys:v_ys
    })
    return result
xs = tf.placeholder(tf.float32,[None,784]) # 28*28
ys = tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs,784,10,activate_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_mean(ys*tf.log(prediction),axis=1))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={
            xs:batch_xs,ys:batch_ys
        })
        if i % 50 ==0:
            print(compute_accuracy(mnist.test.images,mnist.test.labels))

