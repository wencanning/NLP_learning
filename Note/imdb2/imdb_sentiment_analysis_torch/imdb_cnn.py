import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm


from sklearn.metrics import accuracy_score


test = pd.read_csv("D:/Project/NLP_learning/dataset/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_filter = 128
filter_size = 3
bidirectional = True
batch_size = 64
labels = 2
lr = 0.8
# device = torch.device('cuda:0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = True

""" 
    嵌入层 ---> 1维卷积层 (Relu作为激活函数) ----> 池化层 --->
"""
class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_filter, filter_size, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.use_gpu = use_gpu

        """ 
            嵌入层的工作原理:负责将离散的单词索引（整数）转换为连续的词向量
            - 输入：单词的整数索引（形状为 (batch_size, sequence_length)) (64, 512)
            - 输出: 词向量序列（形状为 (batch_size, sequence_length, embed_size)) (64, 512, 300)
        """
        self.embedding = nn.Embedding.from_pretrained(weight)
        """ 
            因为采用Glove预处理, 所以不参与学习
        """
        self.embedding.weight.requires_grad = False

        """ 
            沿着seq方向进行卷积
            2D卷积是对矩阵进行卷积, 类比到1D就是对词向量进行卷积.
            - 输入: (batch_size, embed_size, seq_length) (64, 300, 512)
            - 输出: (batch_size, num_filter, output_length) (64, 128, 512)
            - 核的大小: filter_size * embed_size
            output_length = seq_len - filter_size + 1 + 2*padding
        """
        self.conv1d = nn.Conv1d(embed_size, num_filter, filter_size, padding=1)

        # 激活函数
        self.activate = F.relu
        
        # 全连接层输出为(batch_size=64, lables=2)
        # 相当于分类器, 感觉可以使用逻辑回归
        self.decoder = nn.Linear(num_filter, labels)


    def forward(self, inputs):
        embeddings = self.embedding(inputs)

        convolution = self.activate(self.conv1d(embeddings.permute([0, 2, 1])))


        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])

        # poolin的形状为(64, 128, 1), squeeze是将最后的1维去掉
        outputs = self.decoder(pooling.squeeze(dim=2))
        # print(outputs)
        return outputs


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 加载数据
    logging.info('loading data...')
    pickle_file = r"D:\Project\NLP_learning\imdb_sentiment_analysis_torch\pickle\imdb_glove.pickle3"
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
            vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    net = SentimentNet(embed_size=embed_size, num_filter=num_filter, filter_size=filter_size,
                       weight=weight, labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # ​ ​随机梯度下降（SGD）​​每次迭代​​随机采样一个样本或一个小批量（mini-batch）​计算梯度，
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # 将多个张量（如特征和标签）组合成一个数据集对象
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )
    """ 
        DataLoader是可迭代对象​, ​可以直接用于循环如:for batch in train_iter）
        关键的参数: 
        - batch_size,决定批量的大小
        - shuffle是否乱序
    """
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                # 清除模型参数的梯度，因为torch的梯度是默认累加的, 所以我们要避免梯度累积（每次反向传播前必须调用）。
                net.zero_grad()
                # feature = Variable(feature.cuda())
                # label = Variable(label.cuda())

                # 2. 数据迁移到设备（无需Variable封装）
                feature = feature.to(device)  # 自动保留梯度计算能力（若原Tensor需要梯度）
                label = label.to(device)      # 标签通常不需要梯度

                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(
                        torch.argmax(score.cpu(), dim=1), 
                        label.cpu()
                )
                train_loss += loss

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)
            # 在验证集进行计算时要禁用梯度计算
            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.to(device)
                    val_label = val_label.to(device)
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(
                        torch.argmax(val_score.cpu(), dim=1), 
                        val_label.cpu()
                    )
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)})

            # tqdm.write('{epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f}' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                # test_pred.extent
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())

                pbar.update(1)

    file_path = "./result/cnn.csv"
    # 自动创建路径中所有不存在的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv(file_path, index=False, quoting=3)
    logging.info('result saved!')

