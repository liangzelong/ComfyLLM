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

ComfyLLM 是一个基于节点的可视化大模型开发学习框架，希望通进一步精简大模型操作从而使得每一个人都有能力打造个人数据流以及工作流，最终实现用自己的数据开发自己的大模型成为自己的数据分身为自己发展所用的目标。

个人数据包括笔记、Blog以及Vlog等形式，文字以Obsidian的markdown形式存储；视频在本地直接存储，可通过视频转文字工作流汇总到Obsidian中。Obsidian的数据可以在本地通过RAG方式检索，并训练自己的LoRA模型，LoRA与开源大模型最终成为自己的智能分身。随着自己的数据积累与自身的提高大模型越来越具有个人特色。

ComfyLLM是数据管理中枢可用于快速训练、评估与部署大语言模型。它内置InternLM大部分开源算法，并紧随中国的开源大语言模型发展予以同步更新，传播汉语语言大模型，让每一个人都能拥有属于自己的大模型。它的亮点如下：

**集成主流的大语言模型**

| Model                      | Transformers(HF)                           | ModelScope(HF)                           | OpenXLab(HF)                           | OpenXLab(Origin)                           | Release Date |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------ |
| **Meta-Llama-3-8B**          | [🤗Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | |  | | 2024-04-18 |
| **InternLM2-Chat-1.8B**          | [🤗internlm2-chat-1.8b](https://huggingface.co/internlm/internlm2-chat-1_8b) | [<img src="./.assets/modelscope_logo.png" width="20px" /> internlm2-chat-1.8b](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-1.8b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-1.8b-original) | 2024-02-19   |
| **InternLM2-Chat-7B**      | [🤗internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) | [<img src="./.assets/modelscope_logo.png" width="20px" /> internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b-original) | 2024-01-17   |



**支持丰富的输出接口**

- Obsidian
- Gradio
- Video


## TODO
- [x] 添加大模型节点
- [x] Obsidian连接与交互
- [x] Huixiangdou接入
- [ ] 基于配置流一键部署大模型