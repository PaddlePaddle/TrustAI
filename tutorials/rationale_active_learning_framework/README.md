# 基于证据抽取的二段式模型增强方案
## 背景介绍
机器阅读理解(MRC)是一项通过让机器回答基于给定上下文的问题来测试机器理解自然语言的程度的任务，它有可能彻底改变人类和机器之间的互动方式。具有MRC技术的搜索引擎可以直接以自然语言返回用户提出的问题的正确答案，而不是返回一系列相关的web页面。

而答案抽取式MRC任务是一个典型的MRC任务。首先给定一个文章，一个问题，要求机器根据问题从文章中找出一个连续的片段作为答案。示例如下：

```
文章：目前中信银行信用卡额度一般从3000元到五万元不等。中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。中信信用卡的取现额度为实际额度的50%。如果信用卡批卡之后，持卡者便就可以查询您的信用额度。
问题：中信信用卡白金卡额度是多少？
答案：一万到五万。
```

在长文本的理解问题上，机器阅读理解(MRC)模型往往存在着严重的理解困难问题，同时也存在严重的虚假相关性，即会根据与答案无关的文本信息以抽取答案。具体来说，模型可能会出现下述2种情况：

```
文章：目前中信银行信用卡额度一般从3000元到五万元不等。中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。中信信用卡的取现额度为实际额度的50%。如果信用卡批卡之后，持卡者便就可以查询您的信用额度。
问题：中信信用卡白金卡额度是多少？
预测答案：3000元到五万元
```

```
文章：中信银行信用卡额度一般从3000元到五万元不等。一万到五万之间。
问题：中信信用卡白金卡额度是多少？
预测答案：一万到五万。
```

## 方法介绍
TrustAI提供了“长文本MRC数据证据抽取->抽取式阅读理解预测”流程。在长文本阅读任务中，即，我们先通过Selector抽取一个与问题相关的关键句子，然后再在关键句子上通过Predictor进行预测。

比如说，我们给定一个样本：
```
文章：目前中信银行信用卡额度一般从3000元到五万元不等。中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。中信信用卡的取现额度为实际额度的50%。如果信用卡批卡之后，持卡者便就可以查询您的信用额度。
问题：中信信用卡白金卡额度是多少？
答案：一万到五万。
```
Selector会抽取与问题相关的句子，抽取后的文章部分如下所示：
```
文章：中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。
问题：中信信用卡白金卡额度是多少？
```
Predictor会根据Selector抽取后的文章部分，进行预测：
```
文章：中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。
问题：中信信用卡白金卡额度是多少？
预测答案：一万到五万。
```
我们的方案实现了端到端的可解释和预测，即主动学习证据，基于证据做预测，避免后验分析的弊端。通过在长文本任务上的效果验证，我们的方案在体改性能的同时也有效地提升了模型鲁棒性。

## 实验设置
实验基于`roberta-wwm-ext`微调，评估指标为EM 和 F1 分数，数据集为Dureader<sub>robust</sub> 和 Dureader<sub>checklist</sub>。


## 下载数据

在运行基线系统之前，请下载[Dureader<sub>robust</sub>](https://arxiv.org/abs/2004.11142) 数据集和[Dureader<sub>checklist</sub>](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Checklist-BASELINE) 数据集。
你可以前往[千言](https://aistudio.baidu.com/aistudio/competition/detail/49/0/task-definition) 进行数据集的手动下载。

当然，你可以运行如下命令以快速获得这两个数据集：

```
sh download.sh
```

数据集将保存到`data/`中。

## 训练基线模型

```
sh train.sh -d ./data/robust -o ./output
```
这将启动数据集的训练过程。在训练结束时，模型参数、中间筛选证据和最终dev集预测结果将被保存到`output/`。

## 预测其他数据
由于我们的模型训练过程仅预测dev集结果，我们需要运行以下命令以预测其他数据集：

```
sh test.sh -d [test数据集所在文件夹路径] -o ./output -s [selector模型文件夹路径] -p [predictor模型文件夹路径]
```

## 项目文件结构
```
```

## 方法效果

由下表可知，通过我们的方案，在EM指标上在Dev集上效果提升1.13%，在Challenge Test集上提升了4.94%，说明我们的方案在提升模型性能的同时较好地提高了模型的鲁棒性。

同时我们的方案在Zero Shot设置上也取得了较高的收益。在Robust上训练的该方案在Checklist数据集（移除no answer设置）上将效果提升7.48%。
实验结果如下表所示：

<escape>
<table>
    <tr>
        <th rowspan="2" style="text-align: center;">模型</th>
        <th colspan="2" style="text-align: center;">DuReader-robust dev</th>
        <th colspan="2" style="text-align: center;">DuReader-robust Test</th>
        <th colspan="2" style="text-align: center;">【Zero shot】<br>DuReader-checklist dev<br>(Remove no answer)</th>
    </tr>
    <tr>
        <td style="text-align: center;">EM</td>
        <td style="text-align: center;">F1</td>
        <td style="text-align: center;">EM</td>
        <td style="text-align: center;">F1</td>
        <td style="text-align: center;">EM</td>
        <td style="text-align: center;">F1</td>
    </tr>
    <tr>
        <td>bert-base[官方数据]</td>
        <td>71.20</td>
        <td>82.87</td>
        <td>37.57</td>
        <td>53.86</td>
        <td style="text-align: center;">-</td>
        <td style="text-align: center;">-</td>
    </tr>
    <tr>
        <td>roberta-base[复现]</td>
        <td>73.18</td>
        <td>84.98</td>
        <td>45.97</td>
        <td>69.43</td>
        <td>23.56</td>
        <td>42.47</td>
    </tr>
    <tr>
        <td>Selector-Predictor</td>
        <td>74.31<font color="green">(+1.13)</font></td>
        <td>86.89<font color="green">(+1.91)</font></td>
        <td>50.91<font color="green">(+4.94)</font></td>
        <td>72.22<font color="green">(+2.79)</font></td>
        <td>31.04<font color="green">(+7.48)</font></td>
        <td>53.29<font color="green">(+10.82)</font></td>
    </tr>
</table>
</escape>
