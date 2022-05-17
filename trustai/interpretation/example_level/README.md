# 实例级证据

## 背景介绍
实例级证据分析旨在从训练数据中找出对当前预测起重要作用的若干条实例数据。<br>
## 方法介绍
### 表示点方法
表示点方法([representer point](https://proceedings.neurips.cc/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf))将训练数据对当前预测数据的重要度影响（即表征值），分解为训练数据对模型的影响和训练数据与预测数据的语义相关度。对于一条给定的测试数据和测试结果，表征值为正的训练数据表示支持该预测结果，相反，表征值为负的训练数据表示不支持该预测结果。同时，表征值的大小表示了训练数据对测试数据的影响程度。
#### 输入
- 给定的训练好的模型
- 该模型的训练数据
- 测试数据
#### 输出
- 每一条测试数据对应的支持数据和不支持数据
#### 示例
传入训练好的模型、模型对应的训练数据来初始化表示点模型。基于该表示点模型对预测数据进行证据抽取，需用户指定返回的支持数据和不支持数据条数，由sample_num指定。

```python
from trustai.interpretation import RepresenterPointModel
# initialization
# 开发者需要传入模型及对应的训练数据，以及模型输出层中最后一层的layer name
representer_model = RepresenterPointModel(model, train_data_loader, classifier_layer_name="classifier")
# predict_labels为测试数据预测标签
# pos_examples为训练数据中支持该预测的实例id
# neg_examples为训练数据中不支持该预测的实例id
predict_labels, pos_examples, neg_examples = representer_model(test_dataloader, sample_num=3)
```

结果示例：
```txt
text: 本来不想评价了，但为了携程的携粉们，还是说一下，这称不上是九点，细说就真没必要了，就一个字：差    predict label: 0
pos examples
text: 感觉非常奇怪,这套书我明明都写了两次评论了,可我的当当始终提醒我对这套书写评论!晕啊!这是套很好的书,也不用我写几次评论吧!    gold label: 1
text: 1）背面少个螺丝钉,说是thinkpad都少，靠 2）键盘周围的壳不平整，按下去发现有：“滋啦滋啦”声音，我才意识到，那是个双面胶，按下去就不上来了，过会儿还是回弹上来，很明显仅靠双面胶是 粘不住的，你还不如拿502呢，起码这样粘得严实还能让我心里舒服（但是这样只是弥补质量问题），何必还弄个滋啦兹啦的声音，多闹心啊，（还有一地方用了双面胶，我换内存的时候发现键盘下部盖子左侧打不开，一直不敢用力    gold label: 1
text: 用了6年的THINKPAD,一直认为是笔记本中最好的! 现在这台新的让我......哎!!    gold label: 0
neg examples
text: 是LINUX系统 相当及其恶心 不知道这狗 日 的是什么想法 要强行逼我们使用啊 买了两台电脑 一个事VISTA系统 一个 是 LINUX 就没见一个XP的 网上销售这东西 最重要的是打架尽量不要涉及到售后服务这块 尽量是都搞好了相安无事 其实网上的售后服务比没有售后服务还差劲 我的THINKPAD SL400就是因为换货期间以为是键盘小问题就懒得换了    gold label: 1
text: 盼了2周终于拿到本了，一开机就屏不亮，本人自己跑回总部退机，现在还在等着检测，说要等上15个工作日，呵呵，买个电脑容易吗？时间浪费的起吗？请问？    gold label: 0
text: 价格确实比较高，而且还没有早餐提供。 携程拿到的价格不好？还是自己保留起来不愿意让利给我们这些客户呢？ 到前台搞价格，430就可以了。    gold label: 1
```


表示点方法召回了对测试数据影响较大实例数据。

*注：在真实情况下，众包标注的语料通常掺杂噪音（标注错误），易干扰模型预测。表示点方法倾向于召回梯度较大的训练数据，因此开发者不仅可以使用实例级证据分析方法了解模型行为，也可以通过人工检测标注数据错误，提升模型效果。*

详细示例见[tutorials](../../../tutorials/interpretation/example_level/)。
### 基于梯度的相似度方法
基于梯度的相似度方法([GC, GD](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html))通过模型的梯度挑选与当前测试数据最相似或不相似的数据。

#### 输入
- 给定的训练好的模型
- 该模型的训练数据
- 测试数据
#### 输出
- 每一条测试数据对应的梯度意义上最相似的数据和最不相似的数据
#### 示例
传入训练好的模型、模型对应的训练数据来初始化梯度相似度模型。基于该梯度相似度模型对预测数据进行证据抽取，需用户指定返回的相似数据和不相似数据条数，由sample_num指定。

```python
from trustai.interpretation import GradientSimilarityModel
# initialization
# 开发者需要传入模型及对应的训练数据，以及模型输出层中最后一层的layer name
grad_sim_model = GradientSimilarityModel(model, train_data_loader, classifier_layer_name="classifier")
# 开发者可以通过sim_fn参数指定相似度计算方式，目前支持cos、dot、euc（分别是余弦距离，点积距离和欧式距离）
# predict_labels为测试数据预测标签
# most_similar_examples为训练数据中在梯度意义上预测数据最相似的实例id
# most_dissimilar_examples为训练数据中在梯度意义上预测数据最相似的实例id
predict_labels, most_similar_examples, most_dissimilar_examples = grad_sim_model(test_dataloader, sample_num=3, sim_fn='cos')
```

结果示例：
```txt
text: 本来不想评价了，但为了携程的携粉们，还是说一下，这称不上是九点，细说就真没必要了，就一个字：差    predict label: 0
most similar examples
text: 我选分期付款，在上海扣的款，在上海开的发票，东西却从北京用快递发出，真是舍近求远，害的我等了一星期才收到货，真是脑残！    gold label: 0
text: 看到评价那么高，就买了，但女儿不喜欢，我看了一下，也不喜欢，不知所云，汽车什么的，都是过时的，或生活中没有的。不明白，为什么有这么高的评价。    gold label: 0
text: 看到评价那么高，就买了，但女儿不喜欢，我看了一下，也不喜欢，不知所云，汽车什么的，都是过时的，或生活中没有的。不明白，为什么有这么高的评价。    gold label: 0
most dissimilar examples
text: 单位用户千万别买，支付太不方便。 一定要支票到帐才发货，结果给了支票，居然两天不到财务那里，找了一天才找到支票在那里。 18号下的订单，28号才送到，而且还是支付的现金，支票依然没有到帐。不知道那个单位可以忍受这样的服务。    gold label: 1
text: 这款机子，我没发现任何值得说好的地方！等待我不停的投诉吧！。。。（下面的评价不让多写字，写了很多发不了）    gold label: 1
text: 已经评过了，可是还要求我来评价，这是什么玩意啊？当当的一些功能特别的差劲！难道我买基本就得评价几次么？希望有人处理下这个功能，一点都不人性和智能！！！    gold label: 1
```


基于梯度的相似度方法召回了在梯度意义上与测试数据最相似和最不相似的实例数据。

详细示例见[tutorials](../../../tutorials/interpretation/example_level/)。
### 基于特征的相似度方法
基于特征的相似度方法([FC, FD, FU](https://arxiv.org/abs/2104.04128))通过模型的特征挑选与当前测试数据最相似或不相似的数据。

#### 输入
- 给定的训练好的模型
- 该模型的训练数据
- 测试数据
#### 输出
- 每一条测试数据对应的特征意义上最相似的数据和最不相似的数据
#### 示例
传入训练好的模型、模型对应的训练数据来初始化特征相似度模型。基于该特征相似度模型对预测数据进行证据抽取，需用户指定返回的相似数据和不相似数据条数，由sample_num指定。

```python
from trustai.interpretation import FeatureSimilarityModel
# initialization
# 开发者需要传入模型及对应的训练数据，以及模型输出层中最后一层的layer name
# 注意因为需要计算每一条数据对于模型参数的梯度，所以train_dataloader的batch_size需要设置为1，且需要提供label
feature_sim_model = FeatureSimilarityModel(model, train_data_loader, classifier_layer_name="classifier")
# 开发者可以通过sim_fn参数指定相似度计算方式，目前支持cos、dot、euc（分别是余弦距离，点积距离和欧式距离）
# test_dataloader的标签应是模型的预测标签
# predict_labels为测试数据预测标签
# most_similar_examples为训练数据中在特征意义上预测数据最相似的实例id
# most_dissimilar_examples为训练数据中在特征意义上预测数据最相似的实例id
predict_labels, most_similar_examples, most_dissimilar_examples = feature_sim_model(test_dataloader, sample_num=3, sim_fn='cos')
```

结果示例：
```txt
text: 本来不想评价了，但为了携程的携粉们，还是说一下，这称不上是九点，细说就真没必要了，就一个字：差    predict label: 0
most similar examples
text: 我选分期付款，在上海扣的款，在上海开的发票，东西却从北京用快递发出，真是舍近求远，害的我等了一星期才收到货，真是脑残！    gold label: 0
text: 看到评价那么高，就买了，但女儿不喜欢，我看了一下，也不喜欢，不知所云，汽车什么的，都是过时的，或生活中没有的。不明白，为什么有这么高的评价。    gold label: 0
text: 看到评价那么高，就买了，但女儿不喜欢，我看了一下，也不喜欢，不知所云，汽车什么的，都是过时的，或生活中没有的。不明白，为什么有这么高的评价。    gold label: 0
most dissimilar examples
text: 昨晚看着看着就睡着了，今天早晨醒来就立马抓起继续啃，正逢小说结尾部分，也正如作者的期望，我被吓了一跳。这是我读东野圭吾小说最意料不到的事实。爱有很多种，可是当它变成某种负担时，还算爱吗？爱有多种表现形式，当某种行为让对方感觉负罪时，这还能叫做爱吗？爱可以很长久，当这种永恒变成一种恶梦时，还能归于最初的爱吗？爱可以深入骨髓，无法割除，当它接受无法预知的变数时，还能始终如一吗？    gold label: 1
text: 她们是一群神秘、美丽的女人，从上古的嫘祖，到末代的婉如，每一个都吸引了无数人。我们翻开历史，她们似乎是男人的陪衬，可是当你细细研读，你会发现她们是不可或缺的人物，是每一个朝代最美丽的花朵，也是每一位皇帝的明珠。她们有的与丈夫情投意合，助男人打下一片天下，有的共享荣华，帮皇帝创造盛世，有的颠沛流离，与失意的他同甘共苦，有的……，这一切不仅仅是故事，更是历史，神秘、华美的历史。    gold label: 1
text: 受重建轻管思想的影响，造成水利工程管理研究十分薄弱，涉及工程管理，特别是河道管理类的书籍就更少了。郑教授长期从事河道管理工作，有十分丰富的管理经验和理论修养。该书系统阐述了河道的基本理论，国内外先进管理模式和发展方向，对我国河道管理、法规现状进行了梳理，提出问题和措施问题，并对涉河项目行政许可、河道具体管理提出具体方法和指导，理念新，实用性强，非常有价值！江苏水利 刘    gold label: 1
```


基于特征的相似度方法召回了在特征意义上与测试数据最相似和最不相似的实例数据。


详细示例见[tutorials](../../../tutorials/interpretation/example_level/)。
