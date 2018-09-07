# Neural Chinese Word Segmentation with Dictionary Knowledge
Source codes, paper pdf and talk slides for the Chinese word segmentation algorithm proposed in the following paper.

Junxin Liu, Fangzhao Wu, Chuhan Wu, Yongfeng Huang, and Xing Xie. Neural Chinese Word Segmentation with Dictionary Knowledge. NLPCC 2018

## Dependencies
* [Python 3.4](https://www.python.org/)
* [Tensorflow 1.2](https://www.tensorflow.org/)

## Illustration
    CWS_Dict_Pseudo.py The script for the pseudo labeled data method
    CWS_Dict_multi_task.py The script for the multi-task method
    ./data/ Corpus
    paper.pdf The paper
    NLPCC2018.pdf Meeting talk slides
        
## Abstract
Chinese word segmentation (CWS) is an important task for Chinese NLP. Recently, many neural network based methods have been proposed for CWS. However, these methods require a large number of labeled sentences for model training, and usually cannot utilize the useful information in Chinese dictionary. In this paper, we propose two methods to exploit the dictionary information for CWS. The first one is based on pseudo labeled data generation, and the second one is based on multi-task learning. The experimental results on two benchmark datasets validate that our approach can effectively improve the performance of Chinese word segmentation, especially when training data is insufficient
