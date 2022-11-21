# 证据识别及基于证据的预测
## 背景介绍
在长文本理解任务中，输入中的冗余信息往往会干扰模型预测，导致模型鲁棒性差。如在机器阅读理解(MRC)任务中，模型容易受到输入中其他信息干扰而生成错误答案，如下面示例所示。
```
文章：目前中信银行信用卡额度一般从3000元到五万元不等。中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。中信信用卡的取现额度为实际额度的50%。如果信用卡批卡之后，持卡者便就可以查询您的信用额度。
问题：中信信用卡白金卡额度是多少？
模型预测答案：3000元到五万元
正确答案：一万到五万
```

为了缓解长文本任务中冗余信息的干扰，TrustAI提供了“证据识别-基于证据的预测”的二阶段流程。


## 方法介绍
在TrustAI提供的“证据识别-基于证据预测”的二阶段流程中，首先通过证据抽取模块（**Selector**）识别输入中有效信息，将该信息作为后续模块（**Predictor**）的输入，该模块基于有效信息进行最终答案生成。

在机器阅读理解任务中，Selector完成从文章中抽取与问题相关的一些关键句子，Predictor基于这些关键句子生成最终的答案。

比如说，我们给定一个样本：
```
文章：目前中信银行信用卡额度一般从3000元到五万元不等。中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。中信信用卡的取现额度为实际额度的50%。如果信用卡批卡之后，持卡者便就可以查询您的信用额度。
问题：中信信用卡白金卡额度是多少？
答案：一万到五万。
```
**Selector**从文章中抽取与问题相关的句子，抽取结果如下所示：
```
关键句子：中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。
问题：中信信用卡白金卡额度是多少？
```
**Predictor**根据Selector抽取结果进行预测：
```
关键句子（Predictor输入）：中信普卡的额度一般为3000元到10000元之间，中信白金卡额度在一万到五万之间。
问题：中信信用卡白金卡额度是多少？
预测答案：一万到五万。
```

该方案实现了端到端的可解释和预测，即主动学习证据，基于证据做预测，避免后验分析的弊端。通过在长文本任务上的效果验证，该方案在提高效果的同时也有效地提升了模型鲁棒性。

> 注：开发者可访问[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/4525331)快速体验本案例。

## 使用介绍

我们以阅读理解任务上的实验为例介绍该方案的使用。

实验基于`roberta-wwm-ext`在Dureader<sub>robust</sub>训练数据上微调得到阅读理解模型，在Dureader<sub>robust</sub> 测试集和Dureader<sub>checklist</sub>测试集合上评估模型效果，评估指标为答案的EM（exact match）和F1分数。


### 基线模型-文件结构介绍
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
### 数据集准备

在运行基线系统之前，请下载[Dureader<sub>robust</sub>](https://arxiv.org/abs/2004.11142) 数据集和[Dureader<sub>checklist</sub>](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Checklist-BASELINE) 数据集。
你可以前往[千言](https://aistudio.baidu.com/aistudio/competition/detail/49/0/task-definition) 进行数据集的手动下载。

也可以运行如下命令快速获得这两个数据集：

```shell
sh download.sh
```
下载的数据集将保存到项目根目录下的`data/`目录，内部结构如下：

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

### 模型训练

运行以下命令进行一键训练：

```
sh train.sh -d ./data/robust -o ./output
```

同时，我们分布介绍运行步骤以帮助大家更好的了解模型流程。

**Step 1**：我们运行如下指令训练Selector模型：

```
sh train_selector.sh -d ./data/robust -o ./output
```

在训练完Selector模型后，我们会在`output`文件夹下获得类似下面的目录：
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

**Step 2**：在验证集上（数据格式为[文章, 问题]），运行训练好的Selector来抽取关键句子，得到新的验证集（数据格式为[关键句子, 问题]）：

```
sh train_select_data.sh -d ./data/robust -o ./output
```

我们会在`output`文件夹下获得类似下面的目录：

```
output
└── selected-data
     ├── dev.json
     └── train.json
```

**Step 3**：基于新的验证集，我们基于原始训练数据训练Predictor模型（新的验证集合用来选择最优模型）:
```
sh train_predictor.sh -o ./output
```

训练完Predictor模型，我们会在`output`文件夹下获得类似下面的目录：

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
### 模型预测

针对待测试数据，我们需要通过Selector模块提取证据，然后使用Predictor模块进行最终答案预测。
运行以下命令可以完成数据预测：

```shell
sh test.sh -d [test数据集所在文件夹路径] -o ./output -s [selector模型文件夹路径] -p [predictor模型文件夹路径]
```

具体示例详见[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/4525331)


## 方法效果

由下表可知，该方案使模型在原始Dev上答案EM提升1.13%，在Challenge Test集上答案EM提升了4.94%，说明该方案在提升模型效果的同时提高了模型的鲁棒性。

同时该方案在Zero Shot设置上也取得了较高的表现。在DuReader-Robust数据集上训练的模型，在Checklist数据集（移除no answer设置）上将效果提升3.48%。

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