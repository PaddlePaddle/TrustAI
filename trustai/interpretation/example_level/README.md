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
