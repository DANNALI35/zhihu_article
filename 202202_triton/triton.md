



### triton安装

triton是NVIDIA 2018年开源的服务框架，和flask、tornado类似，但使用triton通常可以获得更高的QPS和更低的内存占用。

#### docker安装

```shell
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
 
 docker -v
```

#### [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)安装

在容器里使用gpu，需要安装 nvida container toolkit。

```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   
   # 验证是否安装成功
   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi  # 如报错，参考排坑
```

#### [triton-inference-server](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md)安装

选择合适[版本](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags),个人选择21.11。

```
sudo docker pull nvcr.io/nvidia/tritonserver:21.11-py3
```

安装完成后的部分log

```
7e9edccda8bc: Pull complete 
a77d121c6271: Pull complete 
074e6c40e814: Pull complete 
Digest: sha256:1ddc4632dda74e3307e0251d4d7b013a5a2567988865a9fd583008c0acac6ac7
Status: Downloaded newer image for nvcr.io/nvidia/tritonserver:21.11-py3
nvcr.io/nvidia/tritonserver:21.11-py3
```

#### 测试可用

```shell
 # 下载官方提供的模型
 git clone https://github.com/triton-inference-server/server.git
cd ./server/docs/examples
./fetch_models.sh
```

```shell
# 启动triton server
sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/triton/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:21.11-py3 tritonserver --model-repository=/models
    
 curl -v localhost:8000/v2/health/ready
```

```shell
# Use docker pull to get the client libraries and examples image from NGC.
sudo docker pull nvcr.io/nvidia/tritonserver:21.11-py3-sdk
# Run the client image
sudo docker run --gpus all -it --rm --net=host nvcr.io/nvidia/tritonserver:21.11-py3-sdk
# run the inference example
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```

### triton使用

#### torch 模型准备

运行pytorch_cifar10.ipynb，使用简单网络训练cifar10数据生成cifar.pt。

#### torch模型转onnx

运行triton.ipynb，读取cifar.pt并将其转换为onnx格式模型，转换完成后模型文件具有如下结构。

```
/home/sync/work/triton/model_repository  # model-repository-path 可自行修改
└── model_test  # your_model_name  可自行修改
    ├── 1  # 固定路径，不可修改
    │   └── model.onnx    # 默认onnx模型名
```

#### triton服务其启动及配置

1、triton服务启动

准备好模型文件的目录结构之后，启动triton服务，并使用`--strict-model-config=false`自动生成模型配置文件。

```shell
sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/sync/work/triton/model_repository:/models nvcr.io/nvidia/tritonserver:21.11-py3 tritonserver --model-repository=/models --strict-model-config=false
```

2、修改triton配置

```
curl localhost:8000/v2/models/model_test/config  # 生成默认config
```

参考默认config，新建config.pbtxt并修改如下，格式要求详见[model_configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)

```
name: "model_test"
platform: "onnxruntime_onnx"
max_batch_size : 256
input [
  {
    name: "images"
    data_type: TYPE_FP16
    format: FORMAT_NONE
    dims: [ 3, -1, -1 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP16
    dims: [ 10 ]
    label_filename: "cifar10_labels.txt"
  }
]
```

修改完成后，具有如下文件结构。

```
/home/sync/work/triton/model_repository
│   └── model_test
│       ├── 1
│       │   └── model.onnx
│       ├── cifar10_labels.txt
│       └── config.pbtxt
```

3、triton服务重启

配置`--strict-model-config=true`选项后重启服务，修改好的配置文件即可生效。

```
sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/sync/work/triton/model_repository:/models nvcr.io/nvidia/tritonserver:21.11-py3 tritonserver --model-repository=/models --strict-model-config=true
```

#### triton服务调用

首先，需安装triton client。

```
pip install tritonclient[all]
```

其次，需要准备triton inference代码，并调用triton inference服务，详见triton.ipynb，其中记录了triton服务的同步和异步调用。

### 排坑

[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)安装

报错：运行   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi  时报错docker: Error response from daemon: Head "https://registry-1.docker.io/v2/nvidia/cuda/manifests/11.0-base": dial tcp: lookup registry-1.docker.io: no such host.See 'docker run --help'.

原因：无法连接registry-1.docker.io

解决方法：

查询registry-1.docker.io对应iphttp://tool.chinaz.com/dns?type=1&host=registry-1.docker.io&ip=，并加入host文件。

```
sudo gedit /etc/hosts  # 加入查询到的ip  52.0.218.102 	registry-1.docker.io
sudo /etc/init.d/networking restart 
```

### REF

triton使用：https://maple.link/2021/06/10/Nvidia%20Triton%20Server%E7%9A%84%E4%BD%BF%E7%94%A8/

triton介绍：https://zhuanlan.zhihu.com/p/418962517