# convit

This repo is to implement convit using MindSpore

# Finish

train.py正在训练

# TO DO

1、model.train 方式训练

2、pynative+混合编程训练

3、差异文档分析编写

# 更新日志

## 2022/12/13

因时间关系，剩余三个规格将以1%以内误差进行验收

各规格日志已上传，ckpt文件较大，通过[阿里云盘链接](https://www.aliyundrive.com/s/whVSbKocNKq)分享。

## 2022/11/30

convit_small_ascend，top1精度高出论文0.3%，top5精度低于论文0.1%

```
convit_small_ascend：top1:0.816, top5: 0.956
```

## 2022/11/20

在验证时修改了crop_pct，convit_tiny_plus_ascend精度达标
```
convit_tiny_plus_ascend：top1:0.770, top5: 0.936
```

## 2022/11/10

v4.6: 使用了cutmix，convit_tiny_ascend精度达标
```
convit_tiny_ascend：top1:0.737, top5: 0.917
```

## 2022/10/27

v4.4：主要就是修改了drop_path，使其适配ascend

## 2022/10/25

```
convit_tiny_gpu：top1:0.734, top5: 0.915
```

## 2022/10/20

v4.2：增加local_init

## 2022/10/17

删除GPSA中的一个tile，运行速度加快8-10倍

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
