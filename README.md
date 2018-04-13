# U-net-segmentation
keras / object dection / image segmentation

U-net网络是一个非常强大的分割网络，（其实说它是一个二分类网络更为准确），这个程序中包含了图像数据变换，U-net模型以及结果可视化和图像数据转换保存等功能。我利用U-net实现了对鱼（可拓展到其他物体）边缘检测的结果，结果发现比传统的边缘检测方法的效果要好很多。

## U-net网络结构：
-------------

<img src="https://github.com/shuyucool/U-net-segmentation/blob/master/image/20170517192834805.png"  height="430" width="500">

## 程序介绍
   在`ata/train`路径下已经有了`image`,`label`两个文件，分别是训练原始图像和`ground truth`.在以'aug_'开头的文件夹下都有一个`test`文件,这个只是为了测试，可以直接删除。

`程序使用说明：`
-------------
1：运行`data.py`,这是为了进行数据增强，原始训练数据太少，所以需要进行仿射变换或者镜像操作等变换方法来生成更多的训练图像和`ground truth`。

2：运行`unet.py`,这可能会花上一段时间。运行结果会在根目录下出现几个`.npy`文件。

3：运行`test_predict.py`,可以将测试集的结果进行可视化。

4：运行`data_vision.py`,可以将测试集及其结果保存成指定的图片格式，并且保存到指定的路径下。（可以根据自己情况选择）

### 效果展示：
<div align="center">
<img src="https://github.com/shuyucool/U-net-segmentation/blob/master/0%20(2).tif"  height="230" width="200">
<img src="https://github.com/shuyucool/U-net-segmentation/blob/master/0.tif" height="230" width="200" >
</div>
