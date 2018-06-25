from pyltp import Segmentor  
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
import numpy as np
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
            label_list.append(int(T[0:1]))
    return text_list,label_list

def model(train_set_in,test_set_in):
    X_train,y_train = get_text_and_label(train_set_in)
    X_test,y_test = get_text_and_label(test_set_in)
#     for i in range(10):
#         print(y_test[i],X_test[i])
    step = [('vect',CountVectorizer(stop_words=stopwords)),
            ('tfidf',TfidfTransformer()),
#             ('pca',PCA(n_components=3000)),
            ('clf',RandomForestClassifier())
#             ('clf',svm.SVC(kernel='poly', degree=2, gamma=1, coef0=0))]
    ppl_clf = Pipeline(step)
    ppl_clf.fit(X_train,y_train)
    
    joblib.dump(ppl_clf,'train_model.m') #保存模型
#     ppl_clf = joblib.load('train_model.m') #加载模型   
    prediction = ppl_clf.predict(X_test)
    precision = np.mean(prediction == y_test)  # 准确率
    print(metrics.classification_report(y_test,prediction))
    print(metrics.confusion_matrix(y_test,prediction))
    
#     svc=svm.SVC(C=1,kernel='poly',degree=3,gamma=10,coef0=0) #选择模型 & 参数
#     lr = LogisticRegression()
#     rf = RandomForestClassifier() 
#     knn = KNeighborsClassifier()  
#     dt = tree.DecisionTreeClassifier()
#     ? MultinomialNB()
def main():
    segmentor = Segmentor() # 加载模型
    segmentor.load('cws.model')
    stopwords = [ line.rstrip() for line in open('stopwords',encoding='utf-8') ] #rstrip() 删除 str末尾的指定字符（默认为空格）
    model('train_set0.txt','test_set0.txt')
    segmentor.release()
    
if __name__ == "__main__":  
    main()
