# ��������
���ǴӺ����Ժ��ҳ�������ά�ȶ�ģ�ͽ��пɽ���������

<br>

## ������
����ģ��Ԥ��������֤�����˹���ע֤�ݵ���϶ȣ�����ʹ��token-F1(macro-F1)��set-F1(IoU-F1, Intersection over Union F1)��Ϊ����ָ�ꡣ

����������ʽһ�͹�ʽ����<br>
��ʽһ��
<p align="center">
<img align="center" src="../..//imgs/equation1.png", width=600><br>
</p>

��ʽ����
<p align="center">
<img align="center" src="../..//imgs/equation2.png", width=600><br>
</p>

���� S<sub>i</sub><sup>p</sup>��S<sub>i</sub><sup>g</sup>�ֱ������Ե�i������ģ��Ԥ��֤�ݺ��˹���ע֤�ݣ�N�������ݼ������ݵ�������<br>

<br><br>

## �ҳ���
����ģ���ṩ֤�ݶ��̶��Ϸ�Ӧ��ģ��Ԥ��ʱ��ʵ��������̡����Ǵ�һ���ԡ�����Ժ��걸������ά������ģ�͵��ҳ��ԡ�

<br>

### һ����
һ��������(ԭʼ���룬��Ӧ�Ŷ�����)���д���Ҫ�������һ���ԡ�֤�ݷ���������������ÿ���ʸ���һ����Ҫ�ȣ����ڸ���Ҫ�ȶ����������дʽ�����������ʹ�����������е�**MAP**��mean average precision��ָ�����������������һ���ԡ������¹�ʽ��
<br>

��ʽ����
<p align="center">
<img align="center" src="../..//imgs/equation3.png", width=600><br>
</p>

����X<sup>o</sup>��X<sup>p</sup>�ֱ����ԭʼ������Ŷ�����Ĵ���Ҫ���������С�|X<sup>p</sup>|����X<sup>p</sup>�дʵĸ�����X<sup>o</sup><sub>1:j</sub>��ʾX<sup>o</sup>��ǰj����Ҫ�Ĵʡ�����G(x, Y)����x�Ƿ�������б�Y�У����������G(x, Y)=1��MAPԽ�߱�ʾ������������һ����Խ��

<br>

### ����Ժ��걸��
**�����**����ģ�͸�����֤���Ƿ������Ԥ����Ҫ��ȫ����Ϣ��<br>
**�걸��**����ģ�Ͷ�����x��Ԥ������<br>
������㷽ʽ���£�<br>
��ʽ��(�����)��
<p align="center">
<img align="center" src="../..//imgs/equation4.png", width=600><br>
</p>
��ʽ��(�걸��)��

<p align="center">
<img align="center" src="../..//imgs/equation5.png", width=600><br>
</p>

F(x<sub>i</sub>)<sub>j</sub>��ʾģ��F��������x<sub>i</sub>Ԥ��Ϊlabel j�ĸ��ʣ�r<sub>i</sub>��ʾ����x<sub>i</sub>��֤�ݣ���Ӧ�أ�x<sub>i</sub>\r<sub>i</sub>��ʾx<sub>i</sub>�ķ�֤�ݲ��֣�����r<sub>i</sub>�����x<sub>i</sub>��ȥ��������Ե÷�Խ�͡��걸�Ե÷�Խ�ߣ���ʾ֤�ݵ��ҳ���Խ�ߡ�
<br>


## ����ʾ��
```python
from trustai.evaluation import Evaluator

evaluator = Evaluator()

# goldens�ǻ��ڲ������ݱ�ע�ı�׼֤��
# predicts�ǻ��ڷ���������õ�Ԥ��֤��
# ���������⣬����ģ��Ԥ���֤�����˹���ע֤�ݼ����ϳ̶ȣ�ѡ��token-F1(macro-F1)��set-F1(IoU-F1, Intersection over Union F1)��Ϊ����ָ�ꡣ
macro_f1 = evaluator.cal_f1(goldens, predicts)
iou_f1 = evaluator.calc_iou_f1(goldens, predicts)

# �ҳ������⣬����ģ�͸�����֤�ݶ��̶���Ӱ��Ԥ�⡣���Ǵ�����ά����������Ӧ3������ָ��
# ά��һ����֤�ݵĳ���Ժ��걸���������ҳ��ԣ�����Ա���֤�ݰ���������Ԥ����㹻��Ϣ���걸�Ա���֤�ݰ����˶�Ԥ����Ӱ���������Ϣ������֤�ݲ����޷�֧��Ԥ�⡣
sufficient, comprehensive = evaluator.cal_suf_com(goldens, predicts)
# ά�ȶ����Ŷ���֤�ݵ�һ���ԣ������Ƶ�����+���Ƶ����=���Ƶ�֤�ݣ����������һ����ģ�����ҳϵ�
map_score = evaluator.cal_map(goldens, predicts)
```
����ʹ��ʾ�����[tutorials](../../tutorials/evaluation/)��
