# test_classification
## 1.Introduction
   **Function** : This project constructs a model to implement test classification through different machine learning algorithms.<br>
   **Development environment** : python 3
   
## 1.type_dict
|Notanyclass|	Life|Movement|Transaction|Business|Conflict|Contact	|Personnel|Justice|
| - |-| -|-|-|-|-|-|-|
|0|1|2|3|4|5|6|7|8|

## 2.corpus
<div align=center><img width="554.8" height="200" src="https://github.com/qwjaskzxl/event_classification/blob/master/image/ace%20corpus.png" alt="ace corpus"/></div>

[out.txt](:storage\3cb00c28-f19b-4703-bfdb-baa843b33176\ec4b2bcc.txt) 一个txt的文本文件，是文本分类中的原始数据 ，它来自ace语料库。
   其中中文633篇，包括bn，nw，wl。其中取66篇文章作为测试集，剩下的567篇作为训练集，从训练集中随机选取33篇文章作为验证集。
   
## 3.方法
    方法是 预测时得到svm预测该样本 每一类的概率，然后设定一个阈值，当此类预测概率大于该阈值时，则输出该类，现在阈值设的是0.1
    
## 4.预处理
  预处理包括1.把文本处理为模型需要的格式。2.分词。3.去停用词。
  预处理的代码：[etree.py](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\43150200.py)
  预处理的结果：[build_set.txt](:storage\3cb00c28-f19b-4703-bfdb-baa843b33176\cad4251d.txt)

## 5.特征工程
### 文本变向量、特征处理、特征选择、特征降维
代码：[predict.py](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\f95c4f76.py)


## 6.训练模型

### 1.文本分类使用的测试集
 [本次分类使用的数据集]（[text-classification/test_set.txt at master · qwjaskzxl/text-classification · GitHub](https://github.com/qwjaskzxl/text-classification/blob/master/samples/test_set.txt)）
 ### 2.文本分类使用的训练集
 [文本分类使用的训练集]（[text-classification/train_set.txt at master · qwjaskzxl/text-classification · GitHub](https://github.com/qwjaskzxl/text-classification/blob/master/samples/train_set.txt)）
### 3.文本分类使用的验证集
[文本分类使用的验证集]（[text-classification/ver_set.txt at master · qwjaskzxl/text-classification · GitHub](https://github.com/qwjaskzxl/text-classification/blob/master/samples/ver_set.txt)）
## 7.预测

## 8.评估
![TIM图片20180626175829.png](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\93573a8f.png)

