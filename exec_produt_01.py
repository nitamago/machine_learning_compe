import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import re
import datetime
import math
 
train_data = pd.read_table('./train_train_mens_coat.tsv')
test_data  = pd.read_table('./test_train_mens_coat.tsv')
 
train_data['price'] = train_data['price'].fillna('0')
 
x_train = train_data.drop(['name','train_id','item_description','price'], axis=1)
y_train = pd.DataFrame({'01_high': (train_data['price']>=30).astype(int), '02_low': (train_data['price']<30).astype(int)})
 
x_test = test_data.drop(['name','train_id','item_description'], axis=1)
i_test = test_data['train_id']
 
def simplify_category_name(df):
    df['category_name'] = df['category_name'].fillna('N')
    df['category_name'] = pd.Categorical(df['category_name'])
    df['category_name'] = df['category_name'].cat.codes
    return df
 
def simplify_name(df):
    df['name'] = df['name'].fillna('N')
    df['name'] = pd.Categorical(df['name'])
    df['name'] = df['name'].cat.codes
    return df
 
def simplify_brand_name(df):
    df['brand_name'] = df['brand_name'].fillna('N')
    df['brand_name'] = pd.Categorical(df['brand_name'])
    df['brand_name'] = df['brand_name'].cat.codes
    return df
 
def transform_features(df):
    df = simplify_category_name(df)
    #df = simplify_name(df)
    df = simplify_brand_name(df)
    return df
 
x_train = transform_features(x_train)
x_test  = transform_features(x_test)
 
DIM = len(x_train.columns)
 
x = tf.placeholder("float", [None, DIM])
W = tf.Variable(tf.zeros([DIM,2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 2])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_))*100
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
 
print(x_train.head(10))
print(y_train.head(10))
print(x_test.head(10))
print(sess.run(W))
print(sess.run(b))
batch_size = 20
length = len(x_train)
 
#for i in range(2000):
for i in range(100):
    for j in range(int(length/batch_size)):
        batch_xs = x_train[j*batch_size:(j+1)*batch_size-1]
        batch_ys = y_train[j*batch_size:(j+1)*batch_size-1]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%10 == 0:
        #print("#----------")
        #print("loop -> ", i ,", " , end="")
        c=0
        t=0
        for index,value in x_train.iterrows():
            batch_xs = [value]
            dy = sess.run(y, feed_dict={x: batch_xs})
            t = t +1
            #print(y_train.loc[index]['high'] , ' <--> ' , int(np.round(dy[0][0])))
            if (y_train.loc[index]['01_high'] == int(np.round(dy[0][0]))):
                c = c + 1
        print(datetime.datetime.today())
        #print(sess.run(W))
        #print(sess.run(b))
        print("try = ", i , ": result = ", c/t)
        #print("----------#")
 
fn = './output.csv'
f = open(fn,'w')
f.write('train_id,high\n')
for index,value in x_test.iterrows():
    batch_xs = [value]
    dy = sess.run(y, feed_dict={x: batch_xs})
    train_id = i_test[index]
    text = str(train_id) +  ',' + str(int(np.round(dy[0][0]))) + '\n'
    f.write(text)
f.close()
