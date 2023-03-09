# NLC_MIA

## Introduction

**Yiteng Peng** 

2019-2023 USTC B.E. Thesis Project

Chinese Title: 基于覆盖率指标的成员推理防御研究

English Title: Research of Member Inference Defense Based on Coverage

## ./

#### analysis.py

使用图表，统计学等方式对MIA攻击判断in/out集和训练测试集的覆盖率结果进行分析，分析的结果保留到mia_data中

#### config.py

预计将模型，数据集，MIA攻击方法，等内容直接写入config文件，从config直接读取，但是目前暂时保留为每个文件里面单独配置。

#### model_cover.py

计算模型训练，测试，mia攻击，防御的覆盖率

#### model_mia.py

对模型进行mia攻击

#### model_test.py

对模型进行测试

#### model_train.py

对模型进行训练

## models

模型源代码

## pretrained

预训练模型，考虑到模型体积，不传到github上面

## datasets

数据集载入部分，输出两种形式的数据，一种是Dataloader，一种是输出普通的数组对。

本目录内部结构组织方式：

```
- datasets
- - data
- - - 数据集名称
- - - - 常规数据
- - - - 用于MIA攻击后分割的数据
- 调用数据的脚本文件
```

**TODO**

- [ ] 可裁剪的数据输出，即，自定义输出输入的规模

## coverage

覆盖率指标，该部分参考：https://github.com/Yuanyuan-Yuan/NeuraL-Coverage

## mia

成员推理攻击部分，该部分参考：https://github.com/BielStela/membership_inference

本目录内部结构组织方式

```
- mia
- - attack_result：mia攻击的结果
- - shadow_data  ：存储生成的影子数据
- - synthesis	 ：生成影子数据的脚本
- shadow_model	 ：影子模型类
- attack_model	 ：攻击模型类
```

## cover_data

统计数据，用来存储model_cover.py输出的模型覆盖率的数据