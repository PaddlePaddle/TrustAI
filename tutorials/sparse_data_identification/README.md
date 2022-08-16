# 基于稀疏数据识别及丰富的增强方案

## 方法介绍
训练数据扩充是提升模型效果的重要手段，然而数据标注是一个费时费力的工作，如何标注更少的数据提升更大的效果是大多数NLP开发者面临的难题。

TrustAI提供了“稀疏数据识别->有效数据选择->训练数据丰富”流程，用尽量少的标注数据有效提升模型效果。在稀疏数据识别中，基于可信分析中的实例级证据分析方法，从测试数据中识别因训练证据不充足而导致的低置信数据，称作目标集。然后，在大量的未标注数据中，选择可以支持目标集中数据预测的证据进行标注。最后，将新标注的数据加入到训练数据中重训模型。

## 实验步骤

由于标注数据成本高昂，本方案基于相似度计算任务开源数据集LCQMC进行模拟实验，在LCQMC的测试集和DuQM鲁棒性数据集上评估效果。实验基于ERNIE-3.0-base-zh微调，评估指标为准确率。

首先，从LCQMC的训练数据中随机抽取5000条作为训练集，剩余数据作为未标注数据集。基于抽取的训练集`train_5000.tsv`训练一个基线模型，用于在后续步骤中做可信分析。运行命令如下所示：

```shell
# 下载数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/sparse_data.tar && tar xf sparse_data.tar && rm sparse_data.tar
# 训练基线模型
python -u train.py --dataset_dir ./data --train_file train_5000.tsv --dev_file dev.tsv --test_files test.tsv DuQM --num_classes 2 --save_dir ./checkpoint
```

基于训练的基线模型`checkpoint`从验证集中选择稀疏数据，即为**目标集**。

目标集选择方法为：使用TrustAI提供的实例级可信分析`FeatureSimilarityModel`方法，计算验证集中样本的正影响证据的平均分数。分数较低的样本表明其训练证据不足，在训练集中较为稀疏，模型在这些样本上表现也相对较差。
```shell
# 选取稀疏数据
python -u find_sparse_data.py --dataset_dir ./data --train_file train_5000.tsv --dev_file dev.tsv --num_classes 2  --init_from_ckpt ./checkpoint/model_state.pdparams --sparse_num 50 --sparse_path ./data/sparse_data.tsv
# sparse_num表示选择的稀疏数据的数量
# sparse_path表示目标集存储的路径
```

在稀疏数据选择好后，只需要再次利用`FeatureSimilarityModel`方法从未标注的数据集`rest_train.tsv`中选择支持目标集的有效数据进行人工标注即可。

<font size=3 color=gray>注：此处为模拟实验，`rest_train.tsv`的数据已被标注</font>

```shell
# 选取有效数据
python -u find_valid_data.py --dataset_dir ./data --unlabeled_file rest_train.tsv --target_file sparse_data.tsv --num_classes 2  --init_from_ckpt ./checkpoint/model_state.pdparams --valid_threshold 0.7 --valid_num 1000 --valid_path ./data/valid_data.tsv
# valid_threshold表示目标集证据的分数阈值，开发者可根据自己数据自主调整，默认为0.7
# valid_num表示抽取有效数据的数量
# valid_path表示有效数据的存储路径
```

在完成有效数据的标注后，将其与原始数据拼接后训练模型，即可提升模型效果。

```shell
# 将标注过的有效集和原始训练集拼接
cat ./data/train_5000.tsv ./data/valid_data.tsv > ./data/merge_valid.tsv
# 基于增强后的数据训练模型
python -u train.py --dataset_dir ./data --train_file merge_valid.tsv --dev_file dev.tsv --test_files test.tsv DuQM sparse_data.tsv --num_classes 2 --save_dir ./valid_checkpoint
```
同时，开发者也可以随机选择相同数量的随机数据进行对比实验。实验结果如下表所示：

|   数据集  | 数据量 |  LCQMC<sub>dev</sub>  | LCQMC<sub>test</sub>  |   DuQM  | 目标集 |
| :-------:  | :-------:  | :-----: | :-----: |:-----: |:-----: |
| 基线   | 5000 | 86.31%  | 84.49% | 69.17%  | 55.20% |  
| 基线 + 随机1000条 | 6000 | 86.76% | 85.05% | 69.23% | 55.20% |
| 基线 + 策略1000条 | 6000 | 87.04% | 85.58% | 70.20% | 69.60% |

<font size=3 color=gray>注：以上结果均为10次实验的平均值。</font>
