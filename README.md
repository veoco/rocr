# rocr

纯 Rust 实现的 OCR 库，基于 PaddlePaddle 的 **PP-OCRv6** 模型。推理引擎使用 [candle](https://github.com/huggingface/candle)，无 C++ 依赖。

> 状态：开发中。**检测、识别、文本行方向分类与文档方向分类均已实现，并通过与 PaddleOCR transformers 引擎 / ONNX oracle 的数值验证**（中英文混排文档端到端识别正确）。

**档位支持**：Tiny / Small / Medium 三档全部可用（检测 + 识别 + 管线 + CLI），均通过与 PaddleOCR transformers 引擎 / ONNX oracle 的数值验证。三档中英文混排文档端到端识别正确，倒置（180°）文本行自动校正，旋转文档页（0/90/180/270°）可自动摆正。

## 特性

- **纯 Rust 推理**：candle 张量库，无 ONNX Runtime / C++ 后端
- **PP-OCRv6**：检测（PPLCNetV4 + RepLKFPN + DB 头）+ 识别（PPLCNetV4 + LightSVTR + CTC）
- **文本行方向分类**：PP-LCNet_x0_25（0°/180°），倒置文本行自动旋转校正
- **文档方向分类**：PP-LCNet_x1_0_doc_ori（0/90/180/270°），旋转页面自动摆正（默认关闭）
- **三档模型**：Tiny / Small / Medium，同一份代码由 config 参数化
- **多语言**：单模型支持 50 种语言
- **多平台**：CPU、CUDA、Apple Metal（通过 feature flags）
- **模型不打包**：用户手动下载，库只读取本地模型目录

## 构建

```bash
cargo build --release
```

可选后端：

```bash
cargo build --release --features rocr/cuda        # NVIDIA GPU
cargo build --release --features rocr/metal       # Apple Silicon
cargo build --release --features rocr/accelerate  # macOS CPU 加速
```

## 模型下载

模型由用户手动下载（Apache-2.0，PaddlePaddle 发布），rocr 不内置、不自动下载。
把下载好的仓库目录传入 `--model-dir` 即可。

### 目录布局

`model-dir` 下应包含对应档位的模型仓库目录（safetensors 格式）：

```
model-dir/
├── PP-OCRv6_small_det_safetensors/           # 检测模型（tier 为 small 时）
├── PP-OCRv6_small_rec_safetensors/           # 识别模型
├── PP-LCNet_x0_25_textline_ori_safetensors/  # 文本行方向分类（跨档位共用）
└── PP-LCNet_x1_0_doc_ori_safetensors/        # 文档方向分类（可选，默认关闭）
```

不同档位（tiny / small / medium）对应不同仓库名，`--model` 选择档位。
方向分类模型固定仓库名：`PP-LCNet_x0_25_textline_ori_safetensors`（默认开启）与
`PP-LCNet_x1_0_doc_ori_safetensors`（默认关闭，CLI 加 `--doc-orientation`，库接口设
`OcrConfig.enable_doc_orientation = true`；关闭可用 `--no-textline-orientation`）。

### 下载命令

从 Hugging Face 下载：

```bash
huggingface-cli download PaddlePaddle/PP-OCRv6_small_det_safetensors --local-dir ./models/PP-OCRv6_small_det_safetensors
huggingface-cli download PaddlePaddle/PP-OCRv6_small_rec_safetensors --local-dir ./models/PP-OCRv6_small_rec_safetensors
```

各档位仓库：
- `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_det_safetensors`
- `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_rec_safetensors`
- `PaddlePaddle/PP-LCNet_x0_25_textline_ori_safetensors`（文本行方向分类，需一个）
- `PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors`（文档方向分类，可选）

开发/测试环境可用脚本下载到 gitignored 的 `dev-models/`：

```bash
# 需要 huggingface_hub CLI: pip install -U huggingface_hub
./scripts/download-models.sh
```

## 使用

```bash
rocr --image image.png --model small --device cpu --model-dir ./models --json
```

CLI 选项：

- `--image <path>`：输入图片，可多次传入实现批量处理
- `--model {tiny,small,medium}`：模型档位（默认 `small`）
- `--device {cpu,cuda,metal}`：推理设备（默认 `cpu`，需对应 feature 编译）
- `--model-dir <dir>`：模型仓库所在目录
- `--doc-orientation`：启用文档方向分类（0/90/180/270°），旋转页面自动摆正
- `--no-textline-orientation`：关闭文本行方向分类（默认开启）
- `--verbose`：向 stderr 输出模型加载与逐图耗时
- `--json`：JSON 输出（批量时按文件名分组）

```bash
rocr --image a.png --image b.png --model medium --model-dir ./models --doc-orientation --verbose
```

## 性能

以下为本机（单线程 CPU，release 构建）对一张 900×700 文档图的整页 OCR 耗时：

| 档位 | 耗时 |
|------|------|
| Tiny  | ~2.9 s |
| Small | ~5.6 s |
| Medium| ~14.6 s |

实际取决于图片尺寸、文本行数与硬件；CUDA / Metal 后端可显著加速。

## 许可证

- 本库代码：**MIT**（见 [LICENSE](LICENSE)）
- 模型权重：**Apache-2.0**（PaddlePaddle / PaddleOCR，见 [NOTICE](NOTICE)）
