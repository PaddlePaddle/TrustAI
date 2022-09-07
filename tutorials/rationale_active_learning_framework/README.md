# 基于证据抽取的QA增强方案
## 背景介绍
在长文本的理解问题上，机器阅读理解(MRC)模型往往存在着严重的理解困难问题，同时也存在严重的虚假相关性，即会根据与答案无关的文本信息以抽取答案。

## 方法介绍
TrustAI提供了“长文本MRC数据证据抽取->抽取式阅读理解预测”流程。在长文本阅读任务中，我们采用了Selector-Predictor两阶段模型，首先通过Selector抽取与答案相关的证据，然后，Predictor进一步利用抽取后的证据以预测答案。

<p align="center">
<img align="center" src="../../../imgs/rationale_active_learning_framework_model.png", width=400><br>
</p>

我们的方案实现了端到端的可解释和预测，即主动学习证据，基于证据做预测，避免后验分析的弊端。通过在长文本任务上的效果验证，我们的方案在体改性能的同时也有效地提升了模型鲁棒性。

## 实验设置
实验基于`roberta-wwm-ext`微调，评估指标为EM 和 F1 分数，数据集为Dureader<sub>robust</sub> 和 Dureader<sub>checklist</sub>。


# 下载数据

在运行基线系统之前，请下载Dureader<sub>robust</sub>数据集和Dureader<sub>checklist</sub> 数据集。运行命令如下所示：

```
sh download.sh
```

数据集将保存到`data/`中。

# 训练基线模型

```
sh train.sh -d ./data/robust -o ./output
```
这将启动数据集的训练过程。在训练结束时，模型参数、中间筛选证据和最终dev集预测结果将被保存到`output/`。

# 预测其他数据
由于我们的模型训练过程仅预测dev集结果，我们需要运行以下命令以预测其他数据集：

```
sh test.sh -d [test数据集所在文件夹路径] -o ./output -s [selector模型文件夹路径] -p [predictor模型文件夹路径]
```

# 方法效果

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
