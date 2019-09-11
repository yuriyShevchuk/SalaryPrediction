import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


def extractData(data_path, **kwargs):
    tf_vec = kwargs.get('tf_vec', None)
    dict_vec = kwargs.get('dict_vec', None)

    data = pd.read_csv(data_path)
    data['FullDescription'] = data['FullDescription'].str.lower()\
        .replace('[^a-zA-Z0-9]', ' ', regex=True)
    if not tf_vec:
        tf_vec = TfidfVectorizer(min_df=5)
        X_text = tf_vec.fit_transform(data['FullDescription'])
    else:
        X_text = tf_vec.transform(data['FullDescription'])
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    if not dict_vec:
        dict_vec = DictVectorizer()
        X_categ = dict_vec.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    else:
        X_categ = dict_vec.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_train = hstack([X_text, X_categ])
    y_train = data['SalaryNormalized'].to_numpy()
    return X_train, y_train, tf_vec, dict_vec


X_train, y_train, tf_vec, dict_vec = extractData('./data/salary-train.csv')
X_test, y_test, tf, dic = extractData('./data/salary-test-mini.csv', tf_vec=tf_vec, dict_vec=dict_vec)
clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(pred)