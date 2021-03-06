# 第四周作业评价标准
本标准来自 Udacity：https://review.udacity.com/#!/rubrics/271/view
（右边的说明是符合要求的条件）

## 数据研究
- 问题 1 选取样本：已选取三个数据样本，提出建立表达式并给出合理解释。
	- 选取样本具代表性 
- 问题 2 属性相关性：准确报告被删除属性的预测分数，合理解释被删除属性是否具有相关性。
	- 成功将 Grocery 属性删除
- 问题 3 属性分布模式：学生找出具有关联的属性并将其与预测属性相比较，随后深入讨论这些属性的数据分布模式。
	- 成功找出 Grocery 和 Detergents_Paper 之间的关系

## 数据处理
- 特征缩放：数据和样本的特征缩放已在代码中正确实施。
- 问题 4 异常检测：学生找出极端的异常值，讨论是否删除这些异常值，并说明删除各数据点的理由。
	- 成功找出所有异常点，并给出理由：移除多个特征值异常的点会使轮廓系数更高 

## 属性转换
- 问题 5：准确报告主要成分分析数据的二个维度与四个维度的总方差。将前四个维度合理解释为对消费者支出的表达。  
	- 因生成 PCA 的结果图与「参考答案」不同，因此判据不同。  
	- 根据所生成的结果图，对消费支出的表达较为令人信服：  
	  第一、三主成分表示的是餐饮相关的商户，第四主成分表示的是批发商
- 降维：对二维缩放数据及样本数据的主要成分分析已在代码中正确实施。
	- 正确实施降维，可视化的 Biplot 和「参考答案」反向近似

## 聚类
- 问题 6 聚类算法：高斯混合模型和K-均值算法已进行详细比较。学生选择的算法符合算法和数据的特点。 
	-  K-Means 聚类算法在数据类别本身区分较大的情况下效果好，发现双标图中原始特征有两个比较明显的方向，因此选用 K-Means 聚类算法
- 问题 7 创建集群：准确报告多个轮廓分数，根据报告的最佳分数选择最佳集群数量。已给出的集群视觉化将根据已选的聚类算法生成最佳的集群数量。
	- 找到最佳轮廓系数：0.426281015469 
- 问题 8 数据恢复：根据数据集的统计描述提出每个客户细分所代表的类型。对集群中心的逆变换和反比例级联已在代码中正确实施。
- 问题 9 样本预测：客户细分正确识别样本数据点，讨论各样本数据点的预测集群。

## 结论
- 问题 10 A/B 测试：提出了某些功能改进方法，可以改进从 A/B 测试获取结果的功能。
- 问题 11 预测额外属性：学生讨论了聚类数据如何可以通过监督学习预测新的属性。
- 问题 12 比较客户数据：客户细分与客户通道数据进行对比，对通道数据识别客户细分的问题进行讨论，包括该表达是否符合早期结果。



