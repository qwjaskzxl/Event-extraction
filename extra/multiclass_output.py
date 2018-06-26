from pyltp import Segmentor  
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer #计算tfidf
from sklearn.feature_extraction.text import CountVectorizer  #计算df 
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
segmentor = Segmentor() # 加载模型
stopwords = []
X_train = []
y_train = []
X_text = []
y_text = []

def get_text_and_label(file_in):
    text_list,label_list = [],[]
    with open(file_in,encoding='utf-8') as f:   
        for T in f.readlines():
            sentence = str(T[2:]).encode('utf-8').decode('utf-8-sig').rstrip()
            words = segmentor.segment(sentence)  #分词，类型为 pyltp.VectorOfString 
            words = ' '.join(list(words))
            text_list.append(words)      
            label_list.append(str(T[0:1]).encode('utf-8').decode('utf-8-sig').rstrip())
    return text_list,label_list

def model(train_set_in,test_set_in):
    X_train,y_train = get_text_and_label(train_set_in)
    X_test,y_test = get_text_and_label(test_set_in)
#     for i in range(10):
#         print(y_test[i],X_test[i])
    step = [('vect',CountVectorizer(stop_words=stopwords)),
            ('tfidf',TfidfTransformer()),
#             ('pca',PCA(n_components=3000)),
#             ('clf',RandomForestClassifier()),
            ('clf',tree.DecisionTreeClassifier())
#             ('clf',svm.SVC(C=2,kernel='linear',gamma=0.4,probability=True))
#             ('clf',svm.LinearSVC())
#             ('clf',svm.SVC(C=1,kernel='poly',degree=2,gamma=0.4,coef0=0,probability=True))
#             ('clf',svm.SVC(C=1,kernel='poly',degree=2, gamma=1, coef0=0,cache_size=200, class_weight=None,
#                            max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False))
           ]
    
    ppl_clf = Pipeline(step)
    ppl_clf.fit(X_train,y_train)    
    joblib.dump(ppl_clf,'train_model.m') #保存模型
    
# 评估+预测 
    
    proba = ppl_clf.predict_proba(X_test) # 分类概率
    print(proba)
    [rows, cols] = proba.shape  
    print(np.shape(proba))
    correct_num = 0
    for i in range(rows): # 即样本数，注意不用-1，range(2)表示0到1
        pre_label,flag = [],False
        for j in range(cols): # 此处即文本类别         
            if proba[i][j] > 0.2 :
                pre_label.append(str(j)) #因为后面list中有数字不能直接join，但设置index时还要转化回int
                if str(j) == y_test[i]:#去掉0的情况记得j+1，因为索引从0开始
                    flag = True                
        pre_label = sorted(pre_label,key=lambda x:proba[i][int(x)],reverse = True) #注意不改变存储位置，所以赋值
        
        print(flag,'\t',y_test[i],'\t',' '.join(pre_label),'\t',X_test[i]) #类别和句子
        correct_num += flag
    print(correct_num/rows)
    
#     ppl_clf = joblib.load('train_model.m') #加载模型   
#     precision = np.mean(prediction == y_test)  # 准确率
#     print(f'by np.mean precision is:',precision)
    prediction = ppl_clf.predict(X_test)
    print(metrics.classification_report(y_test,prediction))
    print(metrics.confusion_matrix(y_test,prediction)) #混淆矩阵
    
#     svc=svm.SVC(C=1,kernel='poly',degree=3,gamma=10,coef0=0) #选择模型 & 参数
#     lr = LogisticRegression()
#     rf = RandomForestClassifier() 
#     knn = KNeighborsClassifier() 
#     dt = tree.DecisionTreeClassifier()
#     ? MultinomialNB()

def main(): 
    segmentor.load('cws.model')
    stopwords = [line.rstrip() for line in open('stopwords',encoding='utf-8')] #rstrip() 删除 str末尾的指定字符（默认为空格）
#     model('train_set0.txt','new_trainningSet_wl.txt')
    model('C:/Users/Administrator/Desktop/demo/train_set0.txt','test_set0.txt')
    segmentor.release()

if __name__ == "__main__":  
    main()