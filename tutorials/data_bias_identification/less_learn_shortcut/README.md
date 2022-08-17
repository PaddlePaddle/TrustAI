# 解决训练数据存在脏数据的问题 - 数据分布修正方案
## 方法介绍
受限于数据集收集方法、标注人员经验等影响，构建的训练数据集存在分布偏置问题。模型会利用数据集中的偏置作为预测的捷径，如在情感分析任务中，遇到否定词或描述直接给出“负向”情感预测。这种偏置会导致模型没有学会真正的理解和推理能力，在与训练数据分布一致的测试数据上表现非常好，但在与训练数据分布不一致的测试数据上表现很差，也就是说模型的泛化性和鲁棒性很差。

TrustAI提供了基于数据集统计方法偏置识别方法，并提供了数据分布修正和权重修正两种优化策略。

基于数据集统计方法偏置识别方法即统计训练数据中词与标注标签的分布，基于此进行偏置词和数据的识别。

数据权重修正通过降低偏置样本对训练loss的影响来减少模型从偏置样本中学习，即在训练loss计算时引入样本的偏置度(详见[Du, Yanrui, et al. 2022](https://arxiv.org/abs/2205.12593))。

## 实验步骤
本方案基于相似度计算任务开源数据集LCQMC训练模型，在LCQMC的测试集和DuQM鲁棒性数据集上评估效果。实验基于ERNIE-3.0-base-zh微调，评估指标为准确率。

首先，统计训练数据中偏置词。

偏置词的统计方法为：统计词在不同类别上的分布，若词出现的频次大于`cnt_threshold`，且最少在一个类别上出现的比例大于`p_threshold`，则将该词视为偏置词。

```shell
# 下载数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/lls_data.tar && tar xf lls_data.tar && rm lls_data.tar
# 统计偏置词
python -u find_bias_word.py --output_dir output --input_path ./data/train.tsv --num_classes 2 --cnt_threshold 3 --p_threshold 0.90 --output_dir output
# cnt_threshold表示为偏置词最少需要出现的频次
# p_threshold表示偏置比例的阈值，偏置词至少需要在一个类别上大于此阈值
# output_dir表示统计结果的存储路径
```

基于偏置词的统计结果，计算训练集中样本偏置度的大小，生成包含样本权重的训练数据。

当前方案提供了`lls_d`和`lls_d_f`两种计算样本偏置度的策略，前者考虑了词的有偏性，而后者同时考虑词的有偏性和频次。

```shell
# 基于`lls_d`策略计算样本偏置度
python -u lls.py --input_path ./data/train.tsv --bias_dir ./output --stopwords_path ./data/stop_words.txt --num_classes 2 --mode lls_d --output_path ./data/train_lls_d.tsv
# 基于`lls_d_f`策略计算样本偏置度
python -u lls.py --input_path ./data/train.tsv --bias_dir ./output --stopwords_path ./data/stop_words.txt --num_classes 2 --mode lls_d_f --output_path ./data/train_lls_d_f.tsv
# mode表示计算样本偏置度的策略，当前有`lls_d`和`lls_d_f`两种策略
# output_path表示为生成带偏置度训练集的存储路径
```

基于带偏置度的训练数据训练模型，即可提升模型效果。
```shell
# 基于`lls_d`策略产生的数据训练模型
python -u train.py --dataset_dir ./data --train_file train_lls_d.tsv --dev_file dev.tsv --test_files test.tsv DuQM --num_classes 2 --save_dir ./lls_d_checkpoint
# 基于`lls_d_f`策略产生的数据训练模型
python -u train.py --dataset_dir ./data --train_file train_lls_d_f.tsv --dev_file dev.tsv --test_files test.tsv DuQM --num_classes 2 --save_dir ./lls_d_f_checkpoint
```

开发者可以基于原始训练数据进行对比实验，由下表可知，相比基线效果，在鲁棒性数据集DuQM上准确率最高提升0.94%。

|   数据集  |  LCQMC<sub>dev</sub>  | LCQMC<sub>test</sub>  |   DuQM  |  
| :-------:  | :-------:  | :-------:  | :-------:  |
| 基线   | 90.93 | 87.06 | 73.82 |  
| lls_d  | 90.76 | 87.58 | 74.76 |  
| lls_d_f  |  90.80 | 87.22 | 74.44 |  

<font size=3 color=gray>注：以上结果均为3次实验的平均值。</font>
