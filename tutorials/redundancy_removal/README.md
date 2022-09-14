# 解决文本冗余导致精度下降的问题
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

> 注：开发者可访问[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/4525331)快速体验本案例。

## 初始文件结构介绍
```python
root
├──predictor  
│   ├── dataloader_factory.py   #数据加载
│   ├── model_manager.py        #模型训练、验证流程管理
│   └── model.py                #核心模型
├──selector  
│   ├── dataloader_factory.py   #数据加载
│   ├── model_manager.py        #模型训练、验证流程管理
│   └── model.py                #核心模型
├──utils
│   ├── checklist_process.py    #checklist移除no answer脚本
│   ├── dureader_robust.py      #robust数据读取脚本
│   ├── logger.py               #日志模块
│   ├── predict.py              #预测结果统计脚本
│   └── tools.py                #分句等处理工具脚本
├──args.py                       #参数管理
├──run_predict.py                #predictor运行启动脚本
├──run_selector.py               #selector运行启动脚本
├──requirements.txt              #环境配置文件
├──README.md                     #帮助文档
├──download.sh                   #数据加载脚本
├──test.sh                       #测试脚本
├──train.sh                      #一键训练脚本
├──train_selector.sh             #selector训练脚本
├──train_select_data.sh          #selector数据筛选脚本
└──train_predictor.sh            #predictor训练脚本
```
## 数据集准备
实验基于`roberta-wwm-ext`微调，评估指标为EM 和 F1 分数，数据集为Dureader<sub>robust</sub> 和 Dureader<sub>checklist</sub>。

在运行基线系统之前，请下载[Dureader<sub>robust</sub>](https://arxiv.org/abs/2004.11142) 数据集和[Dureader<sub>checklist</sub>](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Checklist-BASELINE) 数据集。
你可以前往[千言](https://aistudio.baidu.com/aistudio/competition/detail/49/0/task-definition) 进行数据集的手动下载。

可以运行如下命令以快速获得这两个数据集(数据集将保存到`data/`中)：

```shell
sh download.sh
```
最终，在项目根目录下会出现一个`data`文件夹，内部结构如下：

```
data  
├── robust  
│   ├── train.json  
│   ├── dev.json
├── checklist 
│   ├── train.json  
│   ├── dev.json
└── checklist_wo_no_answer
     ├── train.json  
     └── dev.json
```

## 训练我们的模型

我们支持运行以下命令进行一键训练：

```
sh train.sh -d ./data/robust -o ./output
```

这里我们分布介绍运行步骤以帮助大家更好的了解模型流程。

首先我们运行如下指令训练Selector模型：

```
sh train_selector.sh -d ./data/robust -o ./output
```
其次，在训练完selector模型后，我们会在`output`文件夹下获得类似下面的目录：
```python
output
└── selector
     ├── model_xxx 
     │   └── prediction.json #预测结果
     ├── model_xxx...
     ├── best_model #模型的最好效果的存档
     │   ├── added_tokens.json 
     │   ├── model_state.pdparams
     │   ├── special_tokens_map.json
     │   ├── tokenizer_config.json
     │   └── vocab.txt
     └── logging.json #模型的输出日志
```
下一步地，我们需要根据训练好的selector筛选相应dev集数据：

```
sh train_select_data.sh -d ./data/robust -o ./output
```

在select完dev集合数据后，我们会在`output`文件夹下获得类似下面的目录：

```
output
└── selected-data
     ├── dev.json
     └── train.json
```
然后，基于此，我们训练模型的Predictor:
```
sh train_predictor.sh -o ./output
```
类似地，训练完predictor模型后，我们会在`output`文件夹下获得类似下面的目录：

```python
output
└── predictor
     ├── model_xxx 
     │   └── prediction.json #预测结果
     ├── model_xxx...
     ├── best_model #模型的最好效果的存档
     │   ├── added_tokens.json 
     │   ├── model_state.pdparams
     │   ├── special_tokens_map.json
     │   ├── tokenizer_config.json
     │   └── vocab.txt
     └── logging.json #模型的输出日志
```
## 预测其他数据
由于我们的模型训练过程仅预测dev集结果，我们需要运行以下命令以预测其他数据集：



```shell
sh test.sh -d [test数据集所在文件夹路径] -o ./output -s [selector模型文件夹路径] -p [predictor模型文件夹路径]
```

具体示例详见[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/4525331)


## 方法效果

由下表可知，通过我们的方案，在EM指标上在Dev集上效果提升1.13%，在Challenge Test集上提升了4.94%，说明我们的方案在提升模型性能的同时较好地提高了模型的鲁棒性。

同时我们的方案在Zero Shot设置上也取得了较高的收益。在Robust上训练的该方案在Checklist数据集（移除no answer设置）上将效果提升3.48%。
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
        <td>27.56</td>
        <td>49.47</td>
    </tr>
    <tr>
        <td>Selector-Predictor</td>
        <td>74.31<font color="green">(+1.13)</font></td>
        <td>86.89<font color="green">(+1.91)</font></td>
        <td>50.91<font color="green">(+4.94)</font></td>
        <td>72.22<font color="green">(+2.79)</font></td>
        <td>31.04<font color="green">(+3.48)</font></td>
        <td>53.29<font color="green">(+3.82)</font></td>
    </tr>
</table>
</escape>