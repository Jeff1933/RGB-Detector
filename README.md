Trouble:
我们的机器，比如小车，总是会碰到被杂色影响的情况
而我们一般通过颜色阈值、颜色面积来改善，但确保稳定性，只能见招拆招
所以我们都在寻找一招解决各种问题的方法
#
Solution:
这个方法设置了三个条件来识别RGB物体，前两个就是颜色通道与面积阈值，第三个是计算目标物体的Hu矩阵，Hu矩阵能更好的反映物体的形状，通过Hu矩，就能让计算机确定我们想要的目标图形的形状
#
本来Hu矩应该是用在机器学习，然计算机通过Hu矩来预测形状的，但感觉识别简单的东西貌似浪费资源了
#
所以现在有两个步骤：先通过FindH.py来计算出你的目标物体Hu矩，用什么你最喜欢颜色就行，然后就把Hu矩按格式放到Get.py，再把print出来的颜色范围也应用到Get.py
#
这个方法具有较强的专一性，貌似旋转下角度都会有所不同，但面积不会影响，另外，这个方法对识别纯色物体才比较可以
