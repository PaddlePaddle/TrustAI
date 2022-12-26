# 基于训练信号的数据地图绘制 Dataset Cartography with Training Dynamics

## 方法介绍
现有工作表明，可以使用训练数据在训练过程中的信号绘制数据地图；根据信号特征划分数据，不同数据区域具有不同特点，如难学、标注错误等。通过绘制数据地图，可以帮助开发者更好地了解训练数据。

TrustAI提供了"训练信号收集 -> 数据地图绘制"方案。首先，收集每条训练数据在训练过程中不同step下的训练信号；然后，根据得到的统计信号，基于指定的信号维度绘制数据地图。

注：开发者可访问[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/5307701)快速体验本案例。

## 实验步骤
我们以相似度计算任务LCQMC数据集上的模拟实验为例，介绍该方案实现步骤和效果。


**Step 1**：从LCQMC训练集中随机抽取100条数据作为训练数据样例。训练集文件为.tsv格式，内容如下

```shell
# tsv格式数据，一行一条训练数据，不同字段用tab分隔，各字段如下介绍:
[
    "text_a" : xxx,              // 训练数据文本a
    "text_b" : xxx,              // 训练数据文本b
    "label" : xxx,               // 训练数据对应的label
    "s_label" : xxx,             // 训练数据的构造label （可省略）
]
```
注: ``s_label``可省，为用户构造的label，如构造脏数据。

基于ERNIE-3.0-base-zh在新训练集`sample_100.tsv`上微调得到基线模型，运行命令如下所示：

```shell
# 训练模型并收集训练信号
sh run_train_pointwise.sh
```
所有训练数据的训练信号按训练step保存在`outputs`路径下。

注: 训练信号的收集代码可参考代码`train_pointwise.py`中Lines 199-218，用户可根据自己模型代码进行修改。收集的训练信号如下（用户可自行设计更多信号进行收集）：

```shell
# .jsonl 格式数据
[
    {'id' : xxx,                // 训练数据的id
     'label' : xxx,             // 训练数据对应的label
     'pred_label' : xxx,        // 训练数据的预测label
     'correct' : xxx,           // 训练数据是否被预测正确
     'loss' : xxx,              // 训练数据当前的loss
     'probs' : [xxx, xxx],      // 训练数据在当前每个类下的预测概率(one-hot形式)
     'label_probs' : xxx        // 训练数据在label类别下的预测概率
     }  
    ... ...
]
```

**Step 2**：训练信号处理，即基于不同训练steps收集到的信号计算整体信号，如基于不同steps得到的预测概率计算整体平均预测概率。

```shell
# 训练信号处理
python -u sample_stat_summary.py
```
产出数据保存在`outputs`路径下。

<details>
    <summary> 训练信号详细信息 </summary>

```shell
# tsv 格式数据，一行保存一条训练数据的所有训练信号，信号之间用tab进行分隔，各信号如下表示:
[
    "id" : xxx,                  // 训练数据的id
    "label" : xxx,               // 训练数据对应的label
    "s_label" : xxx,             // 训练数据的构造label，数据地图绘制允许标记困难数据(s_label = 1)和构造脏数据(s_label = 2)
    "correct_times" : xxx,       // 总共预测正确的次数  
    "correct_ratio" : xxx,       // 预测正确次数占比
    "avg_probs" : xxx,           // 多次预测的置信度的平均数
    "label_var" : xxx,           // 多次预测的置信度的方差
    "max_label_probs" : xxx,     // 多次预测的置信度的最大值
    "min_label_probs" : xxx,     // 多次预测的置信度的最小值
    "forgetting_times" : xxx,    // 多次预测反映出的，模型对本数据的遗忘次数（之前预测对了，后来又错了）
    "learnt_times" : xxx,        // 多次预测反映出的，模型对本数据的学会次数（之前预测错了，后来又对了）
    "first_forget" : xxx,        // 多次预测中，第一次遗忘本数据  
    "first_learn" : xxx,         // 多次预测中，第一次学会本数据  
]
```
</details>

**Step 3**：基于产出的训练信号，选择两个信号作为数据地图的主要维度（默认为平均置信度与置信方差），并选择其他信号（如正确比例、正确次数、遗忘次数、学习次数等）以颜色、形状等进行区别绘制数据地图。

```shell
# 数据地图绘制
python -u plot_map.py

# 参数选择
attr1: str类型， 默认值为"avg_probs"，选择一个信号作为数据地图的纵轴
attr2: str类型， 默认值为"label_var"，选择一个信号作为数据地图的横轴
criterion: str类型，默认值为空，选择一个训练信号作为数据筛选依据，仅满足条件的数据会被绘制在地图上
threshold: float类型，默认值0，与criterion一同使用，为选择的训练信号设置阈值，筛选数据
use_f_times: float类型，默认值-1，使用forgotten_times并选择所有遗忘次数不小于use_f_times的样本
use_l_times: float类型，默认值-1，使用learnt_times并选择所有遗忘次数不小于use_l_times的样本

# 数据地图样例
python -u plot_map.py                                               # 图1左
python -u plot_map.py --criterion forgetting_times --threshold 1    # 图1中
python -u plot_map.py --use_l_times 0                               # 图1右
```

<p align="center">
<img align="center" src="../../imgs/data_map_normal.png", width=300>
<img align="center" src="../../imgs/data_map_criterion.png", width=300>
<img align="center" src="../../imgs/data_map_lt.png", width=300><br>
图1 数据地图样例。左: 默认参数设置的数据地图；中：指定criterion为forgetting_times，threshold为1的数据绘制地图；右：使用learnt_times(use_l_times=0)区分数据的颜色，其中左上和左下角数据分别使用correct_times做进一步区分。此数据地图根据全量LCQMC数据的训练信号绘制，而非提供的100条样例。
</p>
