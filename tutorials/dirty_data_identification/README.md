# 解决训练数据存在脏数据的问题

### 方法介绍
训练数据标注质量对模型效果有较大影响，但受限于标注人员水平、标注任务难易程度等影响，训练数据中都存在一定比例的标注较差的数据。当标注数据规模较大时，数据标注检查就成为一个难题。

TrustAI提供了"脏数据识别 -> 清洗"闭环方案，基于实例级证据分析方法给出的训练数据对模型的影响，识别出候选脏数据。开发者对少部分候选脏数据进行人工修正，可显著提升模型效果。


## 实验步骤
由于标注数据成本高昂，本方案基于相似度计算任务开源数据集LCQMC的部分数据上进行模拟实验，在LCQMC的测试集和DuQM鲁棒性数据集上评估效果。实验基于ERNIE-3.0-base-zh微调，评估指标为准确率。

首先，从LCQMC的训练数据中随机抽取5000条作为训练集。基于抽取的训练集`train_5000.tsv`训练一个基线模型，用于在后续步骤中做可信分析。运行命令如下所示：

```shell
# 下载数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/dirty_data.tar && tar xf dirty_data.tar && rm dirty_data.tar
# 训练基线模型
python -u train.py --dataset_dir ./data --train_file train_5000.tsv --dev_file dev.tsv --test_files test.tsv --num_classes 2 --save_dir ./checkpoint
```

基于训练的基线模型`checkpoint`从训练集中选择候选脏数据。
脏数据选择方法为：使用TrustAI提供的实例级可信分析方法`RepresenterPointModel`，计算训练集中样本对模型loss的影响分数，分数越大表明样本为脏数据的可能性越大，模型在这些样本上表现也相对较差。

```shell
# 从训练集中选取候选脏数据
python -u find_dirty_data.py --dataset_dir ./data --train_file train_5000.tsv  --num_classes 2  --rest_path ./data/rest_train.tsv --init_from_ckpt ./checkpoint/model_state.pdparams  --dirty_path ./data/dirty_train.tsv --dirty_num 500
# dirty_num表示选取候选脏数据的数量
# dirty_path表示候选脏数据的存储路径
```

对候选脏数据`dirty_train.tsv`进行人工检查和修正（占全部训练集10%），修正后的数据为`correction_data.tsv`。数据修正的比例为**38.4%**，而在随机选取的数据集中需要修正的数据比例仅为**5%**。

基于修正后的新训练集`train_5000_correction.tsv`训练模型，即可提升模型效果。
```shell
# 下载的数据中包含train_5000_correction.tsv文件
python -u train.py --dataset_dir ./data --train_file train_5000_correction.tsv --dev_file dev.tsv --test_files test.tsv DuQM --num_classes 2 --save_dir ./new_checkpoint
```

开发者可基于修正后的训练集和修正前的训练集进行对比实验，由下表可知，对候选脏数据进行人工检查及修正（规模为原始训练集的10%），模型在LCQMC测试集上提升2.13%，在相似度匹配鲁棒性数据集（DuQM）上提升4.01。


|   数据集  |   LCQMC<sub>dev</sub>  | LCQMC<sub>test</sub>  |   DuQM  |
| :-------:  | :-----: | :-----: |:-----: |
| 基线   |  86.42%  | 84.87% | 69.51%  |  
| 数据修正   | 87.76%  | 86.62% | 73.18%  |  

<font size=3 color=gray>注：以上结果均为10次实验的平均值。</font>
