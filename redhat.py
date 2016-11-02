import numpy as np 
import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import random

def act_data_treatment(dsname):
    dataset = dsname
    
    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if dataset[col].dtype == 'object':
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool':
                dataset[col] = dataset[col].astype(np.int8)
    
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    dataset = dataset.drop('date', axis = 1)
    
    return dataset



act_train_data = pd.read_csv("/Users/wnowak/Desktop/act_train.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
act_test_data  = pd.read_csv("/Users/wnowak/Desktop//act_test.csv", dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
people_data    = pd.read_csv("/Users/wnowak/Desktop/people.csv", dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])

act_train_data=act_train_data.drop('char_10',axis=1)
act_test_data=act_test_data.drop('char_10',axis=1)

print("Train data shape: " + format(act_train_data.shape))
print("Test data shape: " + format(act_test_data.shape))
print("People data shape: " + format(people_data.shape))

act_train_data  = act_data_treatment(act_train_data)
act_test_data   = act_data_treatment(act_test_data)
people_data = act_data_treatment(people_data)

train = act_train_data.merge(people_data, on='people_id', how='left', left_index=True)
test  = act_test_data.merge(people_data, on='people_id', how='left', left_index=True)

del act_train_data
del act_test_data
del people_data


train=train.sort_values(['people_id'], ascending=[1])
test=test.sort_values(['people_id'], ascending=[1])

train_columns = train.columns.values
test_columns = test.columns.values

y = train.outcome
train=train.drop('outcome',axis=1)

# drop people and activity
train=train.drop('people_id',axis=1)
train=train.drop('activity_id',axis=1)
train.head()


yy= pd.get_dummies(y)

train=train.values
yyy=yy.values

x_train,x_test,y_train,y_test = train_test_split(train,yyy,test_size=0.2)

x_train.shape

# there are 57 features
# place holder for inputs. feed in later
x = tf.placeholder(tf.float32, [None, x_train.shape[1]])

# # take 20 features  to 10 nodes in hidden layer
w1 = tf.Variable(tf.random_normal([x_train.shape[1], 1000],stddev=.5,name='w1'))
# # add biases for each node
b1 = tf.Variable(tf.zeros([1000]))
# calculate activations 
hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)

# bring from 10 nodes to 2 for my output
# w2 = tf.Variable(tf.zeros([10,1]))
w2 = tf.Variable(tf.random_normal([1000, 2],stddev=.5,name='w2'))

b2 = tf.Variable(tf.zeros([2]))



# # placeholder for correct values 
y_ = tf.placeholder("float", [None,2])
# # #implement model. these are predicted ys
y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)
# # clipped_output =  tf.clip_by_value(y, 1e-37, 1e+37)




# # CE loss
loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))


LR = .0005
opt = tf.train.AdamOptimizer(learning_rate=LR)

train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])


# # check accuracy
tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))




def get_mini_batch(x,y):
  rows=np.random.choice(x.shape[0], 100)
  return x[rows], y[rows]

# start session
sess = tf.Session()


# # init all vars
init = tf.initialize_all_variables()
sess.run(init)




ntrials = 20000
for i in range(ntrials):
    # get mini batch
    a,b=get_mini_batch(x_train,y_train)
    # run train step, feeding arrays of 100 rows each time
    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})
    if i%100 ==0:
      print("epoch is {0} and cost is {1}".format(i,cost))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))

print("train accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_train, y_: y_train})))

