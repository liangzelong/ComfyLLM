<div align="center">
    <b><font size="6">ComfyLLM</font></b>

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1073056342287323168" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>


## 简介

MMEngine 是一个基于 PyTorch 实现的，用于训练深度学习模型的基础库。它作为 OpenMMLab 所有代码库的训练引擎，其在不同研究领域支持了上百个算法。此外，MMEngine 也可以用于非 OpenMMLab 项目中。它的亮点如下：

**集成主流的大模型训练框架**

- [ColossalAI](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/large_model_training.html#colossalai)
- [DeepSpeed](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/large_model_training.html#deepspeed)
- [FSDP](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/large_model_training.html#fullyshardeddataparallel-fsdp)

**支持丰富的训练策略**

- [混合精度训练（Mixed Precision Training）](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/speed_up_training.html#id3)
- [梯度累积（Gradient Accumulation）](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/save_gpu_memory.html#id2)
- [梯度检查点（Gradient Checkpointing）](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/save_gpu_memory.html#id3)

**提供易用的配置系统**

- [纯 Python 风格的配置文件，易于跳转](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python-beta)
- [纯文本风格的配置文件，支持 JSON 和 YAML](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#id1)

**覆盖主流的训练监测平台**

- [TensorBoard](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#tensorboard) | [WandB](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#wandb) | [MLflow](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#mlflow-wip)
- [ClearML](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#clearml) | [Neptune](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#neptune) | [DVCLive](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#dvclive) | [Aim](https://mmengine.readthedocs.io/zh-cn/latest/common_usage/visualize_training_log.html#aim)

**兼容主流的训练芯片**

- 英伟达 CUDA | 苹果 MPS
- 华为 Ascend | 寒武纪 MLU | 摩尔线程 MUSA

