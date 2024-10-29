# Word2Vector

使用torch复现word2vec，采用的数据集是《平凡的世界》txt文本。

目前实现了CBOW的分层softmax训练，一般softmax训练和负采样训练。

由于是torch训练模型，使用的是自动微分，训练速度没有gensim包中的word2vec模型速度快，普通softmax和分层softmax以及负采样训练三种训练方式之间没有明显的差距。

在训练效果上，普通softmax效果最好，其次是负采样，分层softmax效果较差。

采用余弦相似度度量和`孙少平`之间距离最近的词，三个模型的输出结果如下：

Softmax:

```
['母亲', '王满银', '润叶', '金俊武', '少安', '过去', '向前', '孙玉亭', '田福军', '孙少安']
```

Hierarchical Softmax:

```
['别人', '劳动', '郝红梅', '红梅', '晓霞', '他', '顾养民', '一切', '一种', '她']
```

Negative Sampling

```
['润叶', '向前', '红梅', '晓霞', '王满银', '金富', '孙玉亭', '少安', '孙少安', '田福军']
```