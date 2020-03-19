from model import BILSTM
import utils
import os
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn


from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy


def train(data, params):
    # load word2vec
    print("loading word2vec...")
    ###
    word_vectors = KeyedVectors.load_word2vec_format("sgns.weibo.bigram-char", binary=False)
    wv_matrix = []
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]
        #生成输入数据的矩阵，wv中没有的词要随机初始化
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            # numpy.random.uniform(low,high,size)
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

    # one for UNK and one for zero padding
    # unk means unknown word
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix)
    params["wv_matrix"] = wv_matrix

    #############################
    model = BILSTM(**params)#.cuda(params["GPU"])

    #filter 函数将模型中属性 requires_grad = True 的参数帅选出来，传到优化器中，只有这些参数会被求导数和更新。
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(model.parameters(), params["learning_rate"])
    #交叉熵函数
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    #开始训练
    for e in range(params["epoch"]):
        #打乱顺序
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["batch_size"]):

            #data不足batchsize时的处理
            batch_range = min(params["batch_size"], len(data["train_x"]) - i)
            #list里是每一个句子中每一个词对应vocabulary的索引，句子长度不足的设置为9777，为全0
            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["vocab_size"] + 1] * (params["max_sent_len"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            # 这里把class数字化，用下标作为数据
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
            #################
            batch_x = Variable(torch.LongTensor(batch_x))#.cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y))#.cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)

            loss = criterion(pred, batch_y)
            loss.backward()
            #梯            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        #达到预设的准确率直接结束
        if params["early_stopping"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model

def test(data, model, params, mode="test"):
    #不启用 BatchNormalization 和 Dropout
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    #若词存在则加入index，不存在加入预设的随机值，剩余位置加入预设的空值补充到最大句子长度
    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["vocab_size"] for w in sent] +
         [params["vocab_size"] + 1] * (params["max_sent_len"] - len(sent))
         for sent in x]
    ############################
    x = Variable(torch.LongTensor(x))#.cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc

def main():
    parser = argparse.ArgumentParser(description="-----[BILSTM-classifier]-----")
    print("params begin")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")
    options = parser.parse_args()

    data=utils.Read_Hotel()
    # data = getattr(utils, "Read_Hotel")()
    # data的vacab包含所有的不重复word，class包含所有class，dic，string：list
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    # vocab=[]
    # for x in set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent]):
    #     vocab.append(x)
    # data["vocab"] = sorted(vocab)
    data["classes"] = sorted(list(set(data["train_y"])))
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    params = {
        "save_model": False,
        "early_stopping": False,
        "epoch": 10,
        "learning_rate": 0.001,
        "max_sent_len": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "batch_size": 50,
        "word_dim": 300,
        "lstm_hidden_dim": 100,
        "lstm_num_layers": 1,
        "vocab_size": len(data["vocab"]),
        "class_size": len(data["classes"]),
        "word_embedding": True,
        "dropout": 0.5,
        # "gpu": options.gpu
        }
    print("params finish")
    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    model=train(data, params)
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20 + '\n')


if __name__ == "__main__":
    main()
