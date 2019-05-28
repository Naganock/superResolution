# coding: utf-8

import argparse
import numpy as np
import cv2
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable, cuda
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import os
import skimage.transform
class CNNAE(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 3,pad=1,stride=1)
            self.conv2 = L.Convolution2D(32, 32, 3,pad=1,stride=1)
            self.conv3 = L.Convolution2D(32, 3, 3,pad=1,stride=1)

    def forward(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h
        
    def __call__(self, x, t):
        h = self.forward(x)
        loss = F.mean_squared_error(h, t)
        report({"loss": loss}, self)
        return loss
        
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=3, help="epoch")
parser.add_argument("-g", type=int, default=-1, help="GPU ID (negative value indicates CPU mode)")
args = parser.parse_args()


def getImgData():
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0,1):
        path = "E:/superresolution/300-food/"

        cutNum = 900
        cutNum2 = 10
        imgList = os.listdir(path+str(i))
        imgNum = len(imgList)
        for j in range(cutNum):
            imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])

            if imgSrc is None:continue
            if j < cutNum2:
                X_test.append(imgSrc)
                y_test.append(i)
            elif j < cutNum:
                X_train.append(imgSrc)
                y_train.append(i)

    return X_train,y_train,X_test,y_test

def getDataSet(X_train,y_train,X_test,y_test):
    train_blur=[]
    test_blur=[]
    for i in range(len(X_train)):
        half_size_img = skimage.transform.rescale(X_train[i],0.5)  
        train_blur.append(skimage.transform.rescale(half_size_img,2.))

    for i in range(len(X_test)):
        half_size_img = skimage.transform.rescale(X_test[i],0.5)  
        test_blur.append(skimage.transform.rescale(half_size_img,2.))

    X_train = np.array(X_train).astype(np.float32).reshape((len(X_train),3, 300, 300)) / 255
    y_train = np.array(y_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.float32).reshape((len(X_test),3, 300, 300)) / 255
    y_test = np.array(y_test).astype(np.int32)

    train_blur = np.array(train_blur).astype(np.float32).reshape((len(train_blur),3, 300, 300)) / 255
    test_blur = np.array(test_blur).astype(np.float32).reshape((len(test_blur),3, 300, 300)) / 255

    t_train = y_train.astype(np.int32)
    t_test = y_test.astype(np.int32)

    return X_train,t_train,X_test,t_test,train_blur,test_blur




# 低画質画像とオリジナル画像のペアにする
def getPearData(train,test,train_blur,test_blur):
    train = datasets.TupleDataset(train_blur, train)
    train_iter = iterators.SerialIterator(train, 10, shuffle=True, repeat=True)
    test = datasets.TupleDataset(test_blur, test)
    test_iter = iterators.SerialIterator(test, 10, shuffle=True, repeat=False)
    return train,test,train_iter,test_iter


def makeModel():
    chainer.config.user_gpu = (args.g >= 0)
    if chainer.config.user_gpu:
        cuda.get_device_from_id(args.g).use()
        print("GPU mode")
    else:
        print("CPU mode")

    model = CNNAE()
    if chainer.config.user_gpu:
        model.to_gpu()
    opt = optimizers.Adam()
    opt.setup(model)
    return model,opt


def startTraining(train_iter,test_iter,model,opt):
    # 学習の準備
    updater = training.updaters.StandardUpdater(train_iter, opt, device=args.g)
    trainer = training.Trainer(updater, (args.e,"epoch"))
    # テストの設定
    evaluator = extensions.Evaluator(test_iter, model, device=args.g)
    trainer.extend(evaluator)
    # 学習経過の表示設定
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss"]))
    trainer.extend(extensions.ProgressBar())
    # 学習開始
    trainer.run()


def startTest(test,model):
    chainer.config.train = False
    if chainer.config.user_gpu:
        model.to_cpu()

    test_samples=[]
    for i in range(len(test)):
        test_samples.append(test[i][0])


    for i,img in enumerate(test_samples):
        img = np.expand_dims(img, axis=0)

        output = model.forward(img)

        # 画像として保存
        img = np.squeeze(img*255).astype(np.uint8)
        output = np.squeeze(output.array*255).astype(np.uint8)

        img=img.reshape((300,300,3))
        output=output.reshape((300,300,3))

        cv2.imwrite("E:/superresolution/result/input_{:02d}.png".format(i), img)
        cv2.imwrite("E:/superresolution/result/output_{:02d}.png".format(i), output)


#メイン処理
X_train,y_train,X_test,y_test=getImgData()
train,t_train,test,t_test,train_blur,test_blur = getDataSet(X_train,y_train,X_test,y_test)
train,test,train_iter,test_iter = getPearData(train,test,train_blur,test_blur)
model,opt = makeModel()
startTraining(train_iter,test_iter,model,opt)
startTest(test,model)




