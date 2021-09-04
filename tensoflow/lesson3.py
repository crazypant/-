### variable
import tensorflow as tf

state = tf.Variable(0,name='counter')
# print(state.name)
new_value = tf.add(state,1)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer() # 必须存在 如果定义变量
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


