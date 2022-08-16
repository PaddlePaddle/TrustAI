# 基于证据指导的模型训练增强方案
## 方法介绍
通过当前可信分析结果，即模型预测依赖证据的分析，发现现有NN模型提供的证据合理性偏弱。为进一步提高证据的合理性，TrustAI提供了基于证据指导的模型训练增强的方法，即标注少量证据数据，通过联合学习原始任务和证据学习任务，提升模型的可解释性。

参照MAW(Mean attention weights, [Jayaram etc. 2021](https://aclanthology.org/2021.emnlp-main.450/))方法，我们利用专家标注的证据指导模型attention的优化。

## 实验步骤

我们使用标注了证据的英文情感分析数据集验证方案效果(训练集1000条，验验证集500条)。

实验基于ERNIE-2.0-EN-Base微调，效果评估指标为准确率，可解释性评估指标为合理性、充分性和完备性。

```shell
# 下载样例数据，每个文件仅包含两条样例数据，开发者可根据样例数据的格式自行标注证据数据
wget --no-check-certificate https://trustai.bj.bcebos.com/application_data/rationale_data.tar && tar xf rationale_data.ta && rm rationale_data.ta
python -u train.py --dataset_dir ./data --train_file train.tsv --dev_file dev.tsv --num_classes 2 --save_dir ./maw --use_maw
# user_maw表示是否使用证据增强模型效果
```

由下表可知，在引入MAW loss后，模型效果和可解释性都有明显提升，准确率提升0.5%，同时模型的合理性（+5.0%）、充分性（-0.185）和完备性（+0.044）获得一致改善。

|   数据集   | 准确率   | 合理性 | 充分性 | 完备性 |
| :-------:  | :-----: | :-----: | :-----: | :-----: |
| base   | 93.5% | 26.1% | 0.367 | 0.118 |
| base + maw loss | 94.0% | 31.1% | 0.182 | 0.162 |

<font size=3 color=gray>注：以上结果均为3次实验的平均值。</font>
