c-bpnn
======

BP神经网络的C语言实现

## 1. 简介
此项目使用C语言（C99）实现BP神经网络，其分为两个部分：
1. 训练器；
2. 适配器
训练器对原始数据进行训练、验证，并且生成BP神经网络参数文件，适配器使用训练器训练好的参数对输入给出相应的输出。

## 2. 使用
本项目实现了BP神经网络，并且产生了两个可执行文件对实现进行验证。

测试输入为3个[0,1]的浮点数据a,b,c，其输出结果为(a+b+c)/3.

1. git clone https://github.com/ThreeClassMrWang/c-bpnn.git
2. cd c-bpnn
3. mkdir build
4. cd build
5. cmake ..
6. make
7. ./test_bpnn_train
8. ./test_bpnn

## 3. 说明
1. 此BP神经网络要求输入必须归一化处理；
2. bpnn.h及bpnn.c为训练器的实现代码，test_bpnn_train.c对其进行测试；
3. bpnn_fit.h及bpnn_fit.c为适配器代码，利用训练器出来的网络参数文件对输入给出相应的输出；
4. 在bpnn_config.h中可以对数据输入输出的维度、迭代次数、学习率及临界累积误差进行配置。
5. 实际使用时，需要先使用数据训练BP神经网络，然后得到网络参数文件，再利用参数文件调用适配器。

## 4. 补充
理论部分请参阅：https://zhuanlan.zhihu.com/p/27110594
