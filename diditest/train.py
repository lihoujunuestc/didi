import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
RAW_PATH = '/home/lijian/Downloads'


input_file = os.path.join(RAW_PATH, '1.csv')
test_file = os.path.join(RAW_PATH, 'test.csv')


def count_y(df_line):
    return df_line[2]-df_line[3]


def get_df_date(df_line):
    df_value = df_line[1].split('-')
    return df_value[:-1]


def get_df_time(df_line):
    df_value = df_line[1]
    return df_value[-1]


df_train = pd.read_csv(input_file, engine='python', encoding='utf-8', header=None, sep=' |\t')
df_test = pd.read_csv(test_file, engine='python', encoding='utf-8', header=None, sep=' |\t')

print df_train
x_train_second = df_train.iloc[:, 4:]
y_train = df_train.apply(count_y, axis=1)
x_time = df_train.apply(get_df_time, axis=1)
x_date = df_train.apply(get_df_date, axis=1)
date_to_number_dict = dict((date_didi, i) for date_didi, i in enumerate(x_date.unique(), 1))
x_date = x_date.map(lambda x: date_to_number_dict.get(x, x))
first_value = df_train.iloc[:, 0]
x_first_part = [[i, j, k] for i, j, k in zip(first_value, x_date, x_time)]

enc = preprocessing.OneHotEncoder()
x_enc_first_part = enc.fit_transform(x_first_part)
df_x_first_part = pd.DataFrame(x_first_part, index=x_train_second.index)
x_train = pd.concat(x_train_second, df_x_first_part, axis=1)


lr = LinearRegression()
lr.fit(x_train, y_train)

x_test = df_test.iloc[:, 4:]
y_pred = lr.predict(x_test)

