# MuseTalk

MuseTalk：实时高质量的唇同步与潜在空间修复
</br>
张悦 <sup>\*</sup>，
刘敏豪<sup>\*</sup>，
陈兆康，
吴斌<sup>†</sup>，
曾宇彬， 
詹超，
何英杰，
黄俊鑫，
周文江
(<sup>*</sup>平等贡献，<sup>†</sup>通讯作者，benbinwu@tencent.com)

Lyra Lab，腾讯音乐娱乐

**[github](https://github.com/TMElyralab/MuseTalk)**    **[huggingface](https://huggingface.co/TMElyralab/MuseTalk)**    **[space](https://huggingface.co/spaces/TMElyralab/MuseTalk)**    **[技术报告](https://arxiv.org/abs/2410.10122)**

我们介绍 `MuseTalk`，一个**实时高质量**的唇同步模型（在NVIDIA Tesla V100上达到30fps+）。MuseTalk可以与输入视频一起使用，例如由[MuseV](https://github.com/TMElyralab/MuseV)生成，作为完整的虚拟人解决方案。

:new: 更新：我们很高兴地宣布[MusePose](https://github.com/TMElyralab/MusePose/)已发布。MusePose是一个图像到视频生成框架，能够根据姿势等控制信号生成虚拟人。结合MuseV和MuseTalk，我们希望社区能够加入我们，共同朝着生成具有全身运动和互动能力的虚拟人这一愿景迈进。

# 概述
`MuseTalk`是一个实时高质量的音频驱动唇同步模型，训练于`ft-mse-vae`的潜在空间中，具有以下特点：

1. 根据输入音频修改未见过的面孔，面部区域大小为`256 x 256`。
2. 支持多种语言的音频，如中文、英文和日文。
3. 在NVIDIA Tesla V100上支持30fps+的实时推理。
4. 支持修改面部区域的中心点，这**显著**影响生成结果。
5. 提供在HDTF数据集上训练的检查点。
6. 训练代码（即将发布）。

# 新闻
- [04/02/2024] 发布MuseTalk项目和预训练模型。
- [04/16/2024] 在HuggingFace Spaces上发布Gradio [演示](https://huggingface.co/spaces/TMElyralab/MuseTalk)（感谢HF团队的社区资助）。
- [04/17/2024]：我们发布了一个利用MuseTalk进行实时推理的管道。
- [10/18/2024] :mega: 我们发布了[技术报告](https://arxiv.org/abs/2410.10122)。我们的报告详细介绍了一个优于开源L1损失版本的模型。它包括GAN和感知损失以提高清晰度，以及同步损失以增强性能。

## 模型
![模型结构](assets/figs/musetalk_arc.jpg)
MuseTalk在潜在空间中训练，图像由冻结的VAE编码，音频由冻结的`whisper-tiny`模型编码。生成网络的架构借鉴了`stable-diffusion-v1-4`的UNet，其中音频嵌入通过交叉注意力与图像嵌入融合。

请注意，尽管我们使用了与Stable Diffusion非常相似的架构，但MuseTalk与之不同，它**不是**一个扩散模型。相反，MuseTalk通过在潜在空间中进行单步修复来操作。

## 案例
### MuseV + MuseTalk让人类照片复活！
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="33%">图像</td>
        <td width="33%">MuseV</td>
        <td width="33%">+MuseTalk</td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/musk/musk.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/4a4bb2d1-9d14-4ca9-85c8-7f19c39f712e controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/b2a879c2-e23a-4d39-911d-51f0343218e4 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/yongen/yongen.jpeg width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/57ef9dee-a9fd-4dc8-839b-3fbbbf0ff3f4 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/94d8dcba-1bcd-4b54-9d1d-8b6fc53228f0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/sit/sit.jpeg width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/5fbab81b-d3f2-4c75-abb5-14c76e51769e controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/f8100f4a-3df8-4151-8de2-291b09269f66 controls preload></video>
    </td>
  </tr>
   <tr>
    <td>
      <img src=assets/demo/man/man.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/a6e7d431-5643-4745-9868-8b423a454153 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/6ccf7bc7-cb48-42de-85bd-076d5ee8a623 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/monalisa/monalisa.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/1568f604-a34f-4526-a13a-7d282aa2e773 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/a40784fc-a885-4c1f-9b7e-8f87b7caf4e0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/sun1/sun.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/37a3a666-7b90-4244-8d3a-058cb0e44107 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/172f4ff1-d432-45bd-a5a7-a07dec33a26b controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/sun2/sun.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/37a3a666-7b90-4244-8d3a-058cb0e44107 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/85a6873d-a028-4cce-af2b-6c59a1f2971d controls preload></video>
    </td>
  </tr>
</table >

* 最后两行的角色`Xinying Sun`是一位超级模特KOL。您可以在[douyin](https://www.douyin.com/user/MS4wLjABAAAAWDThbMPN_6Xmm_JgXexbOii1K-httbu2APdG8DvDyM8)上关注她。

## 视频配音
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="70%">MuseTalk</td>
        <td width="30%">原始视频</td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/4d7c5fa1-3550-4d52-8ed2-52f158150f24 controls preload></video>
    </td>
    <td>
      <a href="//www.bilibili.com/video/BV1wT411b7HU">链接</a>
      <href src=""></href>
    </td>
  </tr>
</table>

* 对于视频配音，我们应用了一种自开发的工具，可以识别说话者。

## 一些有趣的视频！
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%">图像</td>
        <td width="50%">MuseV + MuseTalk</td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/video1/video1.png width="95%">
    </td>
    <td>
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/1f02f9c6-8b98-475e-86b8-82ebee82fe0d controls preload></video>
    </td>
  </tr>
</table>

# 待办事项：
- [x] 训练模型和推理代码。
- [x] Huggingface Gradio [演示](https://huggingface.co/spaces/TMElyralab/MuseTalk)。
- [x] 实时推理的代码。
- [ ] 技术报告。
- [ ] 训练代码。
- [ ] 更好的模型（可能需要更长时间）。

# 开始使用
我们为新用户提供了关于MuseTalk安装和基本使用的详细教程：

## 第三方集成
感谢第三方集成，使得安装和使用对每个人来说更加方便。
我们也希望您注意，我们没有验证、维护或更新第三方。请参考该项目以获取具体结果。

### [ComfyUI](https://github.com/chaojie/ComfyUI-MuseTalk)

## 安装
要准备Python环境并安装额外的包，如opencv、diffusers、mmcv等，请按照以下步骤操作：
### 构建环境

我们推荐使用python版本>=3.10和cuda版本=11.7。然后按如下方式构建环境：

```shell
pip install -r requirements.txt
```

### mmlab包
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### 下载ffmpeg-static
下载ffmpeg-static并设置环境变量
```
export FFMPEG_PATH=/path/to/ffmpeg
```
例如：
```
export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
```
### 下载权重
您可以手动下载权重，如下所示：

1. 下载我们的训练[权重](https://huggingface.co/TMElyralab/MuseTalk)。

2. 下载其他组件的权重：
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
   - [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

最后，这些权重应按如下方式组织在`models`中：
```
./models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```
## 快速开始

### webui
终端运行 `python app.py`  
webui链接：[http://localhost:7860/](http://localhost:7860/)  

### API
终端运行 `python api.py`  
并发数、端口等在`api.py`中配置。  
API文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

### 推理
这里，我们提供推理脚本。 
```
python -m scripts.inference --inference_config configs/inference/test.yaml 
```
configs/inference/test.yaml是推理配置文件的路径，包括video_path和audio_path。
video_path可以是视频文件、图像文件或图像目录。

建议您输入`25fps`的视频，这是训练模型时使用的相同帧率。如果您的视频远低于25fps，建议您应用帧插值或直接使用ffmpeg将视频转换为25fps。

#### 使用bbox_shift以获得可调结果
:mag_right: 我们发现，掩码的上限对嘴部张开的影响很大。因此，为了控制掩码区域，我们建议使用`bbox_shift`参数。正值（向下半部分移动）会增加嘴部张开度，而负值（向上半部分移动）会减少嘴部张开度。

您可以先使用默认配置运行以获得可调值范围，然后在此范围内重新运行脚本。

例如，在`Xinying Sun`的情况下，运行默认配置后，显示可调值范围为[-9, 9]。然后，为了减少嘴部张开度，我们将值设置为`-7`。 
```
python -m scripts.inference --inference_config configs/inference/test.yaml --bbox_shift -7 
```
:pushpin: 更多技术细节可以在[bbox_shift](assets/BBOX_SHIFT.md)中找到。

#### 结合MuseV和MuseTalk

作为虚拟人生成的完整解决方案，建议您首先应用[MuseV](https://github.com/TMElyralab/MuseV)生成视频（文本到视频、图像到视频或姿势到视频），请参考[此处](https://github.com/TMElyralab/MuseV?tab=readme-ov-file#text2video)。建议使用帧插值来提高帧率。然后，您可以使用`MuseTalk`生成唇同步视频，参考[此处](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file#inference)。

#### :new: 实时推理

这里，我们提供推理脚本。该脚本首先应用必要的预处理，如人脸检测、人脸解析和VAE编码。在推理过程中，仅涉及UNet和VAE解码器，这使得MuseTalk能够实时运行。

```
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --batch_size 4
```
configs/inference/realtime.yaml是实时推理配置文件的路径，包括`preparation`、`video_path`、`bbox_shift`和`audio_clips`。

1. 在`realtime.yaml`中将`preparation`设置为`True`以准备新`avatar`的材料。（如果`bbox_shift`已更改，您还需要重新准备材料。）
2. 之后，`avatar`将使用从`audio_clips`中选择的音频片段生成视频。
    ```
    正在推理使用：data/audio/yongen.wav
    ```
3. 当MuseTalk正在推理时，子线程可以同时将结果流式传输给用户。生成过程可以在NVIDIA Tesla V100上达到30fps+。
4. 如果您想使用相同的avatar生成更多视频，请将`preparation`设置为`False`并运行此脚本。

##### 实时推理注意事项
1. 如果您想使用相同的avatar/video生成多个视频，您也可以使用此脚本来**显著**加快生成过程。
2. 在之前的脚本中，生成时间也受到I/O（例如保存图像）的限制。如果您只想测试生成速度而不保存图像，可以运行
```
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --skip_save_images
```

# 致谢
1. 我们感谢开源组件，如[whisper](https://github.com/openai/whisper)、[dwpose](https://github.com/IDEA-Research/DWPose)、[face-alignment](https://github.com/1adrianb/face-alignment)、[face-parsing](https://github.com/zllrunning/face-parsing.PyTorch)、[S3FD](https://github.com/yxlijun/S3FD.pytorch)。
2. MuseTalk在很大程度上参考了[diffusers](https://github.com/huggingface/diffusers)和[isaacOnline/whisper](https://github.com/isaacOnline/whisper/tree/extract-embeddings)。
3. MuseTalk是基于[HDTF](https://github.com/MRzzm/HDTF)数据集构建的。

感谢开源！

# 限制
- 分辨率：尽管MuseTalk使用的面部区域大小为256 x 256，使其优于其他开源方法，但尚未达到理论分辨率限制。我们将继续处理这个问题。
如果您需要更高的分辨率，可以结合使用超分辨率模型，如[GFPGAN](https://github.com/TencentARC/GFPGAN)。

- 身份保留：原始面孔的一些细节未能很好保留，如胡须、嘴唇形状和颜色。

- 抖动：由于当前管道采用单帧生成，存在一些抖动。

# 引用
```bib
@article{musetalk,
  title={MuseTalk: Real-Time High Quality Lip Synchorization with Latent Space Inpainting},
  author={Zhang, Yue and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2024}
}
```
# 免责声明/许可证
1. `代码`：MuseTalk的代码在MIT许可证下发布。对学术和商业使用没有限制。
2. `模型`：训练模型可用于任何目的，甚至是商业用途。
3. `其他开源模型`：其他开源模型的使用必须遵守其许可证，如`whisper`、`ft-mse-vae`、`dwpose`、`S3FD`等。
4. 测试数据来自互联网，仅可用于非商业研究目的。
5. `AIGC`：该项目努力积极影响AI驱动的视频生成领域。用户被授予使用该工具创建视频的自由，但他们应遵守当地法律并负责任地使用。开发者不对用户的潜在误用承担任何责任。
