# 训练数据偏置识别及偏置消除 - 数据分布修正

## 方法介绍
受限于数据集收集方法、标注人员经验等影响，构建的训练数据集中往往存在偏置现象。模型会利用数据集偏置作为预测捷径，如在情感分析任务中，遇到否定词或描述直接给出“负向”情感预测。这种偏置会导致模型没有学会真正的理解和推理能力，在与训练数据分布一致的测试数据上表现很好，但在与训练数据分布不一致的测试数据上往往会表现较差。

TrustAI提供了数据集偏置识别及基于分布修正的偏置缓解策略。
* 偏置识别：统计训练数据中词与标注标签的分布，在分布上不均衡的词可能是偏置词，这里需要使用任务相关词典对候选偏置词过滤，得到真正的偏置词。包含偏置词的样本为偏置样本。
* 分布修正：对非偏置样本进行重复采样。


注：开发者可访问[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/4434652)快速体验本案例。

## 实验步骤
实验基于ERNIE-3.0-base-zh在情感分析任务ChnsentiCorp数据集上微调得到基线模型，在情感分析鲁棒性数据集上评估效果，评估指标为准确率。


**Step 1**：识别偏置词。基于特征级证据可信分析方法（`IntGradInterpreter`）获取训练数据预测依赖的证据，然后统计各证据频次信息。
```shell
# 下载数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/distribution_data.tar && tar xf distribution_data.tar && rm distribution_data.tar
# 训练基线模型
python -u train.py --dataset_dir ./data --train_file train.tsv --dev_file robust.tsv --num_classes 2 --save_dir ./checkpoint

# 统计重要证据和频次
python -u get_rationale_importance.py --dataset_dir ./data --input_file train.tsv --num_classes 2  --rationale_path ./data/rationale_importance.txt  --init_from_ckpt ./checkpoint/model_state.pdparams
# rationale_path为证据及其频次保存的地址
```

**Step 2**：识别偏置样本，及对偏置样本重复采样以达到均衡。

```shell
# 生成均衡训练数据
python -u balance_train_data.py  --input_path ./data/train.tsv  --rationale_path ./data/rationale_importance.txt --output_path ./data/balanced_train.tsv
```

基于生成的均衡数据`balanced_train.tsv`训练模型。

```shell
python -u train.py --dataset_dir ./data --train_file balanced_train.tsv --dev_file robust.tsv --num_classes 2 --save_dir ./checkpoint
```
实验效果如下表所示：
|   数据集  |   鲁棒性数据集  |  
| :-------:  |  :-------:  |
| 基线   |   69.97 |  
| 分布修正   |   71.38 |  

<font size=3 color=gray>注：以上结果均为10次实验的平均值。</font>
