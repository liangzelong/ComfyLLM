# 使用说明

## 如果无法显示
1. 安装mmagic: https://github.com/open-mmlab/mmagic
2. 在任意路径打开terminal下载sd1.5权重文件: 
   ```
   huggingface-cli download --resume-download runwayml/stable-diffusion-v1-5 --local-dir ./sd1_5
   ```
3. 权重文件会下载到本路径sd1_5文件夹下
4. 将config中的`/home/lzl/temp/sd1_5/`替换成你的sd1_5文件，即可运行prompt