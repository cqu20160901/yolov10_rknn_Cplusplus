# yolov10_rknn_Cplusplus
yolov10 瑞芯微 rknn 板端 C++部署，使用平台 rk3588。


## 编译和运行

1）编译

```
cd examples/rknn_yolov10_demo_open

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_yolov10_demo_open

./rknn_yolov10_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/firefly/zhangqian/rknn/rknpu2_1.4.0/examples/rknn_yolov10_demo_open/model/RK3588/yolov8n_ZQ.rknn";
    char image_path[256] = "/home/firefly/zhangqian/rknn/rknpu2_1.4.0/examples/rknn_yolov10_demo_open/test.jpg";
    char save_image_path[256] = "/home/firefly/zhangqian/rknn/rknpu2_1.4.0/examples/rknn_yolov10_demo_open/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```


# 测试效果


冒号“:”前的数子是coco的80类对应的类别，后面的浮点数是目标得分。（类别:得分）

![images](https://github.com/cqu20160901/yolov10_rknn_Cplusplus/blob/main/examples/rknn_yolov10_demo_open/test_result.jpg)


把板端模型推理和后处理时耗也附上，供参考，使用的芯片rk3588，模型输入640x640，检测类别80类。

![image](https://github.com/cqu20160901/yolov10_rknn_Cplusplus/assets/22290931/3c843ff5-4746-4cb3-83da-6fb758a98195)


本示例用的是yolov10n，模型计算量6.7G，看到这个时耗觉得可能是有操作切换到CPU上进行计算的，查了rknn转换模型日志确实是有操作切换到CPU上进行的，对应的是模型中 PSA 模块计算 Attention 这部分操作。

![image](https://github.com/cqu20160901/yolov10_rknn_Cplusplus/assets/22290931/27f478b1-c99a-4e55-9d8e-228e1dc9fe66)


# 导出onnx 参考
[【yolov10 瑞芯微RKNN、地平线Horizon芯片部署、TensorRT部署，部署工程难度小、模型推理速度快】](https://blog.csdn.net/zhangqian_1/article/details/139239964)


