# 可视化
#添加层
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs,in_size,out_size,activate_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activate_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activate_function(Wx_plus_b)
    return outputs



x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


l1 = add_layer(xs,1,10,activate_function=tf.nn.relu)
predicition = add_layer(l1,10,1,activate_function=None)

loss = tf.reduce_mean(tf.reduce_sum(
    tf.square(ys-predicition),reduction_indices=[1]
))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
with tf.Session()as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={
            xs:x_data,ys:y_data
        })
        if i%50 == 0:
            # print(sess.run(loss,feed_dict={
            #     xs:x_data,ys:y_data
            # }))
            #
            predicition_value = sess.run(predicition,feed_dict={
                xs:x_data
            })
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x_data,predicition_value,'r-',lw=5)

            plt.pause(0.5)


