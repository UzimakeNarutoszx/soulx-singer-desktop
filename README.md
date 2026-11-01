# SoulX-Singer 桌面集成实验

## 功能

在 SoulX-Singer 上游基础上保留歌声合成界面与 macOS 桌面封装实验。

## 技术栈

Python、PyTorch、Gradio、Swift、React、TypeScript。依赖版本以现有清单和上游兼容要求为准，早期代码可能需要历史版本。

## 目录

soulxsinger/ 上游模型实现；webui.py 界面；desktop/ macOS 包装；preprocess/ 预处理及 MIDI 编辑器。

## 启动

在本仓库目录执行；Python 示例需先激活自行创建的虚拟环境。

```sh
python3 -m venv .venv
# 激活虚拟环境后
python3 -m pip install -r requirements.txt
python3 webui.py
```

## 公开范围与许可

仅包含脱敏文本源码，不含原 Git 历史、私人绝对路径、凭据、用户数据库/聊天日志、未经逐张审阅的原始图片或二进制媒体。未发布资源不在许可证授权范围内。详见 [LICENSE](LICENSE) 与 [ATTRIBUTION.md](ATTRIBUTION.md)。

保留上游 Apache-2.0 许可证；本地界面与桌面集成不被宣称为模型原创。

## 验证与限制

不含权重、示例音频、图片、图标、生成音频或第三方压缩 JS。安装模型与运行条件见 README.upstream.md；未执行模型推理或桌面构建。

本次执行静态语法/JSON 检查和独立密钥扫描；结果记录在 `VALIDATION.md`。静态通过不代表界面、依赖安装、硬件或端到端运行成功。缺失资源须使用自行创建或获得授权的替代品，不能从原个人素材库自动恢复。
