### 相关链接

- **Bag of Words  Meets Bags of Popcorn**: [Bag of Words Meets Bags of Popcorn | Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-2-word-vectors)

### 各模型的训练结果
- **imdb_bert_native**
```
Epoch 0: train loss=0.2872, train acc=0.88, val loss=0.3236, val acc=0.86
Epoch 1: train loss=0.1600, train acc=0.94, val loss=0.2055, val acc=0.92
Epoch 2: train loss=0.1003, train acc=0.97, val loss=0.2910, val acc=0.90
```

- **imdb_roberta**
```
Epoch 0: train loss=0.3009, train acc=0.87, val loss=0.2062, val acc=0.92
Epoch 1: train loss=0.1807, train acc=0.93, val loss=0.1991, val acc=0.93
Epoch 2: train loss=0.1250, train acc=0.96, val loss=0.2127, val acc=0.92
```

- **imdb_deberta**
```
Epoch 0: train loss=0.1784, train acc=0.93, val loss=0.1304, val acc=0.95
Epoch 1: train loss=0.1023, train acc=0.97, val loss=0.1361, val acc=0.95
Epoch 2: train loss=0.0635, train acc=0.98, val loss=0.1630, val acc=0.95
```

- **imdb_deberta_lora**
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.234300</td>
      <td>0.236316</td>
      <td>0.962250</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.131400</td>
      <td>0.210252</td>
      <td>0.964000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.321800</td>
      <td>0.209702</td>
      <td>0.964750</td>
    </tr>
  </tbody>
</table><p>

![alt text](image-2.png)

在kaggle上的评分
![alt text](image-1.png)