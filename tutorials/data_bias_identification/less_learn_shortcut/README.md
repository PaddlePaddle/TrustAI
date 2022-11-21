# 解决训练数据分布偏置的问题 - 数据权重修正方案
## 方法介绍
受限于数据集收集方法、标注人员经验等影响，构建的训练数据集中往往存在偏置现象。模型会利用数据集偏置作为预测捷径，如在情感分析任务中，遇到否定词或描述直接给出“负向”情感预测。这种偏置会导致模型没有学会真正的理解和推理能力，在与训练数据分布一致的测试数据上表现很好，但在与训练数据分布不一致的测试数据上往往会表现较差。

TrustAI提供了数据集偏置识别及基于权重修正的偏置缓解策略。
* 偏置识别：统计训练数据中词与标注标签的分布，在分布上不均衡的词可能是偏置词，包含偏置词的样本为偏置样本。
* 权重修正：降低偏置样本对训练loss的影响，即针对每一条样本计算一个偏置度，在训练loss计算时通过偏置度降低偏置样本影响，具体见[Du, Yanrui, et al. 2022](https://arxiv.org/abs/2205.12593)。

注：开发者可访问[ AI Studio示例 ](https://aistudio.baidu.com/aistudio/projectdetail/4434616)快速体验本案例。

## 实验步骤
实验基于ERNIE-3.0-base-zh在情感分析任务ChnsentiCorp数据集上微调得到基线模型，在情感分析鲁棒性数据集上评估效果，评估指标为准确率。


**Step 1**：识别训练数据中的偏置词。在训练数据中，统计每个词在不同类别上的分布，对于频次大于`cnt_threshold`、且最少在一个类别上出现比例大于`p_threshold`的词视为偏置词。

```shell
# 下载数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/lls_data.tar && tar xf lls_data.tar && rm lls_data.tar
# 统计偏置词
python -u find_bias_word.py --output_dir output --input_path ./data/train.tsv --num_classes 2 --cnt_threshold 3 --p_threshold 0.90 --output_dir output
# cnt_threshold表示为偏置词最少需要出现的频次
# p_threshold表示偏置比例的阈值，偏置词至少需要在一个类别上大于此阈值
# output_dir表示统计结果的存储路径
```

**Step 2**：基于偏置词的统计结果，针对每一训练样本，计算偏置度，作为样本对训练loss的影响权重。

当前方案提供了`lls_d`和`lls_d_f`两种计算样本偏置度的策略，前者考虑词的有偏性，后者同时考虑词的有偏性和频次。

```shell
# 基于`lls_d`策略计算样本偏置度
python -u lls.py --input_path ./data/train.tsv --bias_dir ./output --stopwords_path ./data/stop_words.txt --num_classes 2 --mode lls_d --output_path ./data/train_lls_d.tsv
# 基于`lls_d_f`策略计算样本偏置度
python -u lls.py --input_path ./data/train.tsv --bias_dir ./output --stopwords_path ./data/stop_words.txt --num_classes 2 --mode lls_d_f --output_path ./data/train_lls_d_f.tsv
# mode表示计算样本偏置度的策略，当前有`lls_d`和`lls_d_f`两种策略
# output_path表示为生成带偏置度训练集的存储路径
```

**Step 3**：用带偏置度的训练数据训练模型，偏置度作用于loss计算。
```shell
# 基于`lls_d`策略产生的数据训练模型
python -u train.py --dataset_dir ./data --train_file train_lls_d.tsv --dev_file dev.tsv --test_files test.tsv DuQM --num_classes 2 --save_dir ./lls_d_checkpoint
# 基于`lls_d_f`策略产生的数据训练模型
python -u train.py --dataset_dir ./data --train_file train_lls_d_f.tsv --dev_file dev.tsv --test_files test.tsv DuQM --num_classes 2 --save_dir ./lls_d_f_checkpoint
```

实验结果如下表所示：相比于基线，权重修正后，模型在鲁棒性数据集DuQM上准确率提升0.94%。

|   数据集  |  LCQMC<sub>dev</sub>  | LCQMC<sub>test</sub>  |   DuQM  |  
| :-------:  | :-------:  | :-------:  | :-------:  |
| 基线   | 90.93 | 87.06 | 73.82 |  
| lls_d  | 90.76 | 87.58 | 74.76 |  
| lls_d_f  |  90.80 | 87.22 | 74.44 |  

<font size=3 color=gray>注：以上结果均为3次实验的平均值。</font>
