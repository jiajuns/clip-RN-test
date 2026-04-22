# RN50 最小推理运行说明

## 1. 激活环境

```bash
source .env/bin/activate
```

README.md 顶部已经汇总了本地新增文件、版本信息、CIFAR-10 benchmark 启动方式，以及后续 ResNet-20 最可能修改的位置。

## 2. 运行最小推理示例

这个环境里 `ALL_PROXY` 是 `socks://...`，会让 `huggingface_hub/httpx` 下载权重时报错。
所以运行时把代理变量临时清空即可。

```bash
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY= no_proxy= \
.env/bin/python infer_rn50_cn.py \
  --image docs/CLIP.png \
  --text "a diagram" "a dog" "a cat"
```

## 3. 首次运行说明

首次运行会自动下载 RN50 的 `openai` 预训练权重，后续会直接复用本地缓存。
