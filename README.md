# convit

This repo is to implement convit using MindSpore

# Finish

train.py正在训练

# TO DO

1、model.train 方式训练

2、pynative+混合编程训练

3、差异文档分析编写

# 更新日志

## 2022/10/16

为加快训练速度，对代码进行部分修改

## 2022/10/9:

使用pthtockpt.py脚本，将pth模型转换为ckpt模型，并进行了精度推理测试
```
top1 acc: 0.72576
top5 acc: 0.9146
```
略低于论文中的精度，但认为可以通过调整训练参数

## 2022/10/8：

MLP层中，将
```
self.act = nn.GELU()
```
修改为
```
self.act = nn.GELU(approximate=False)
```
减小了与pytorch的误差
