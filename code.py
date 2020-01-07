import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk.stem
from nltk.corpus import stopwords
import string
##路径读取
txtPath = r'C:\Users\admin\Desktop\wenben\train'
txtPath2 = r'C:\Users\admin\Desktop\wenben\test\5'
txtLists = os.listdir(txtPath)
txtLists2 = os.listdir(txtPath2)
##---------------预处理-----------------##

X_TRAIN = []
X_TRIAL = []
label = []
label2 = []
name = []


#############训练集文本预处理
for txtlist in txtLists:

## 提取标签   

    label.append(txtlist[0])
    path = os.path.join(txtPath,txtlist)
    f = open(path)
    s = f.read()

##去掉'ABSTRACT'
    s = s.lstrip('ABSTRACT')
##去标点
    remove = str.maketrans('','',string.punctuation)
    without_punctuation = s.translate(remove)
##换行符
    s = without_punctuation.replace('\n',' ',10000)
##全部转换为小写
    s = s.lower()
##以空格为间隔划分为列表
    s = s.split(' ')
##去掉''    
    s = [w for w in s if w is not '']
##去掉英文停词    
    without_stopwords = [w for w in s if not w in stopwords.words('english')]
##词干提取
    s = nltk.stem.SnowballStemmer('english')
##列表转化为字符串
    cleaned_text = ' '.join([s.stem(ws) for ws in without_stopwords])
##存入X
    X_TRAIN.append(cleaned_text)


############测试集文本预处理  
for txtlist2 in txtLists2:
    name.append(txtlist2)
    label2.append(txtlist2[0])
    path2 = os.path.join(txtPath2,txtlist2)
    f2 = open(path2)
    s2 = f2.read()
    s2 = s2.lstrip('ABSTRACT')
    remove = str.maketrans('','',string.punctuation)
    without_punctuation2 = s2.translate(remove)
    s2 = without_punctuation2.replace('\n',' ',10000)
    s2 = s2.lower() 
    s2 = s2.split(' ')
    s2 = [w for w in s2 if w is not '']
    without_stopwords2 = [w for w in s2 if not w in stopwords.words('english')]
    s2 = nltk.stem.SnowballStemmer('english')
    cleaned_text2 = ' '.join([s2.stem(ws) for ws in without_stopwords2])
    X_TRIAL.append(cleaned_text2)


## -------------预测 ------------##

#TF-IDF
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}'\
                             , max_features=5000)
##词向量转化
X_TRAIN = tfidf_vect.fit_transform(X_TRAIN)
X_TRIAL = tfidf_vect.transform(X_TRIAL)

##朴素贝叶斯模型训练
rfc = MultinomialNB().fit(X_TRAIN,label)

##预测
label = rfc.predict(X_TRIAL)

##保存
data2 = {'name':name,'label': label}
frame2 = pd.DataFrame(data2)
frame2.to_csv(r'C:\Users\admin\Desktop\wenben\result.csv',\
              index = False,header = False)



