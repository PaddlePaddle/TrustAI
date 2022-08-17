# 解决训练数据存在脏数据的问题 - 数据分布修正方案

## 方法介绍
受限于数据集收集方法、标注人员经验等影响，构建的训练数据集存在分布偏置问题。模型会利用数据集中的偏置作为预测的捷径，如在情感分析任务中，遇到否定词或描述直接给出“负向”情感预测。这种偏置会导致模型没有学会真正的理解和推理能力，在与训练数据分布一致的测试数据上表现非常好，但在与训练数据分布不一致的测试数据上表现很差，也就是说模型的泛化性和鲁棒性很差。

TrustAI提供了基于数据集统计方法偏置识别方法，并提供了数据分布修正和权重修正两种优化策略。

基于数据集统计方法偏置识别方法即统计训练数据中词与标注标签的分布，基于此进行偏置词和数据的识别。

数据分布修正通过对非偏置数据多次重复采样，使训练数据分布尽量均衡。该方案通过可信分析方法识别训练数据中对模型预测其重要贡献的证据，然后通过分析训练中标签和证据的分布识别偏置样本，对偏置样本重复采样来达到数据均衡的目的。

## 实验步骤
本方案在情感分析数据集ChnsentiCorp上进行实验，在情感分析鲁棒性数据集上评估效果。实验基于ERNIE-3.0-base-zh微调，评估指标为准确率。

首先，通过可信分析识别训练数据中对模型预测其重要贡献的证据。
重要证据统计方法为：基于特征级可信分析方法`IntGradInterpreter`识别训练数据中起重要贡献的证据和频次。
```shell
# 下载数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/distribution_data.tar && tar xf distribution_data.tar && rm distribution_data.tar
# 训练基线模型
python -u train.py --dataset_dir ./data --train_file train.tsv --dev_file robust.tsv --num_classes 2 --save_dir ./checkpoint

# 统计重要证据和频次
python -u get_rationale_importance.py --dataset_dir ./data --input_file train.tsv --num_classes 2  --rationale_path ./data/rationale_importance.txt  --init_from_ckpt ./checkpoint/model_state.pdparams
# rationale_path为证据及其频次保存的地址
```

基于统计的证据及其频次分析偏置样本，在偏置样本的不均衡类别上重复采样，达到数据均衡的目的。

```shell
# 生成均衡训练数据
python -u balance_train_data.py  --input_path ./data/train.tsv  --rationale_path ./data/rationale_importance.txt --output_path ./data/balanced_train.tsv
```

基于生成的均衡数据`balanced_train.tsv`训练模型，即可提升模型效果。

```shell
python -u train.py --dataset_dir ./data --train_file balanced_train.tsv --dev_file robust.tsv --num_classes 2 --save_dir ./checkpoint
```
实验结果如下表所示：
|   数据集  |   鲁棒性数据集  |  
| :-------:  |  :-------:  |
| 基线   |   69.97 |  
| 分布修正   |   71.38 |  

<font size=3 color=gray>注：以上结果均为10次实验的平均值。</font>
