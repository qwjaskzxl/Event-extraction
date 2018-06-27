# test_classification
## 1.Introduction
   **Function** : This project constructs a model to implement text classification through different machine learning algorithms.<br><br>
   **Requirement** : python 3

## 2.corpus
  We use the----ACE corpus, which is commonly used in event extraction, as our original dataset.
<div align=center><img width="554.8" height="200" src="https://github.com/qwjaskzxl/event_classification/blob/master/image/ace%20corpus.png" alt="ace corpus"/></div>

The 9 types of corresponding labels are as follows:

|Notanyclass|	Life|Movement|Transaction|Business|Conflict|Contact	|Personnel|Justice|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|1|2|3|4|5|6|7|8|
<p align="center">type dictionary</p>
   
## 3.Extract samples
[out.txt](:storage\3cb00c28-f19b-4703-bfdb-baa843b33176\ec4b2bcc.txt) 一个txt的文本文件，是文本分类中的原始数据 ，它来自ace语料库。
   其中中文633篇，包括bn，nw，wl。其中取66篇文章作为测试集，剩下的567篇作为训练集，从训练集中随机选取33篇文章作为验证集。
   
   ### dataset
  [文本分类使用的训练集]（[text-classification/train_set.txt at master · qwjaskzxl/text-classification · GitHub](https://github.com/qwjaskzxl/text-classification/blob/master/samples/train_set.txt)）
  
   [test_set]([text-classification/test_set.txt at master · qwjaskzxl/text-classification · GitHub])
   [train_set]([text-classification/train_set.txt at master · qwjaskzxl/text-classification · GitHub])
   [validation set]([text-classification/ver_set.txt at master · qwjaskzxl/text-classification · GitHub])
   
    方法是 xxx
## 4.Pre Processing
  预处理包括1.把文本处理为模型需要的格式。2.分词。3.去停用词。
  预处理的代码：[etree.py](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\43150200.py)
  预处理的结果：[build_set.txt](:storage\3cb00c28-f19b-4703-bfdb-baa843b33176\cad4251d.txt)

## 5.Feature Engineering
### 文本变向量、特征处理、特征选择、特征降维
代码：[predict.py](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\f95c4f76.py)

## 6.Models
### SVM
### RandowForest

## 7.Evaluation

## 8.Optimization

