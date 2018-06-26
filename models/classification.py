from pyltp import Segmentor  

def cut_sentence(file_in):
    cutwords_list = [] #����
    file_original_txt = open(file_in,'r',encoding='utf-8')
    stopwords = [ line.rstrip() for line in open('stopwords',encoding='utf-8') ] #rstrip() ɾ�� strĩβ��ָ���ַ���Ĭ��Ϊ�ո�

    segmentor = Segmentor()
    segmentor.load('cws.model')  #����ģ��
    sentences = file_original_txt.readlines()
    
    for sente in sentences:
        temp='' #������ű��зֺ�ġ����ӡ�
        sente = str(sente).encode('utf-8').decode('utf-8-sig') #���롢���롣����label����Ϊ�� '\ufeff' �Ƿ��ַ�
        label = str(sente[0:1]) #ȥlabel
        temp += label+'\t'
        sente = sente[2:] #ȥlabel
        words = segmentor.segment(sente)  #�ִʣ�����Ϊ pyltp.VectorOfString
        word_list = list(words) #������list��
        for word in word_list[1:]:
            if word not in stopwords:
                temp += word+' '
        cutwords_list.append(temp)
        
    segmentor.release()  #�ͷ�ģ��
    file_original_txt.close()
    return cutwords_list

# cut_sentence('train_set0.txt')

from sklearn.feature_extraction.text import TfidfTransformer #����tfidf
from sklearn.feature_extraction.text import CountVectorizer  #����df
vectorizer=CountVectorizer()#����Ὣ�ı��еĴ���ת��Ϊ��Ƶ���󣬾���Ԫ��a[i][j] ��ʾj����i���ı��µĴ�Ƶ  
transformer=TfidfTransformer()#�����ͳ��ÿ�������tf-idfȨֵ     

def get_tfidf(file_in):
    corpus = []
    for sente in cut_sentence(file_in):
        corpus.append(sente[2:])        #ȥlabel
#     print(corpus)
    tf=vectorizer.fit_transform(corpus) #��һ��fit_transform�ǽ��ı�תΪ��Ƶ����                    
    tfidf=transformer.fit_transform(tf) #�ڶ���fit_transform�Ǽ���tf-idf
    return tfidf      
get_tfidf('train_set0.txt')

from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import tree
from sklearn.externals import joblib
from sklearn import metrics

def training(train_set_in):
    X = get_tfidf(train_set_in)
    Y = []
    for sente in cut_sentence(train_set_in):
        Y.append(sente[0:1])
    svc=svm.SVC(C=1,kernel='poly',degree=3,gamma=10,coef0=0) #ѡ��ģ�� & ����
    lr = LogisticRegression()
    rf = RandomForestClassifier() 
    knn = KNeighborsClassifier()  
    dt = tree.DecisionTreeClassifier()
    clf = rf.fit(X,Y) #ѵ��ģ��
    joblib.dump(clf,'train_model.m') #����ģ��
    
def testing(test_set_in):
    file_original_txt = open(test_set_in,'r',encoding='utf-8')
    original_txt = file_original_txt.readlines() #ԭʼ�ı�
    i,j = 0,0
    clf = joblib.load('train_model.m') #����ģ��   
    
    for sente in cut_sentence(test_set_in):
        label = int(sente[0:1])
        T = []
        T.append(str(sente[2:]))      
        tf=vectorizer.transform(T) #�����ݽ���ת�����������ݵĹ�һ���ͱ�׼�������������ݰ���ѵ������ͬ����ģ�ͽ���ת�����õ�����������
        tfidf=transformer.transform(tf)
        print(label,'\t',int(clf.predict(tfidf)),'\t',str(original_txt[j][2:])) 
        if label == int(clf.predict(tfidf)):
            i += 1 
        j += 1
    
    print('\nprecision :',i/j)
# training('train_set0.txt')
# testing('test_set0.txt')

def main():
    training('train_set0.txt')
    testing('test_set0.txt')
    
if __name__ == "__main__":  
    main()