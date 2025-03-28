# LangBotPluginDocument

## 安装

配置完成 [LangBot](https://github.com/RockChinQ/LangBot) 主程序后使用管理员账号向机器人发送命令即可安装：

```
!plugin get https://github.com/unanmed/LangBotPluginDocument
```

或查看详细的[插件安装说明](https://docs.langbot.app/plugin/plugin-intro.html#%E6%8F%92%E4%BB%B6%E7%94%A8%E6%B3%95)

## 使用

首先将你需要的文档放入 `docs` 文件夹，目前仅支持 `markdown` 格式的文件以及代码文件，可以添加文件夹。

打开 `config.json`，你将会看到一系列配置，它们分别是：

-   `reference_prompt` 和 `question_prompt`: 文档提示词和用户问题提示词，在向大模型提问时，会自动将这些内容添加至消息内容，最终消息会变成：

```python
f"{reference_prompt}\n{查询到的文档内容}\n{question_prompt}{用户提问}"
```

-   `text_model`: 本地运行 RAG 推理时默认使用的模型，会从 HuggingFace 自动下载，需要科学上网，如果没有办法科学上网，可以自行在镜像站下载后，放入本地，然后本项填写模型本地路径。

-   `code_model`: 本地运行 RAG 推理时对代码处理的默认模型，如果文档是 `text-code` 或 `code-only` 模式，那么会在代码部分使用该模型推理。其余同 `model_name`。

-   `mode`: 用于设置默认 RAG 方案，可以填写这些值：

    -   `text-only`: 纯文本文档，不包含代码、图片等内容。
    -   `text-code`: 文本+代码组合的文档，适用于一些项目的开发文档，会自动针对文本和代码使用不同的 RAG 方案。
    -   `code-only`: 纯代码。

-   `chunk_size`: 每个文档分块的大小，参考[分割与查询原则](#分割与查询原则)。

-   `chunk_overlap`: 每个文档分块的重叠大小，，参考[分割与查询原则](#分割与查询原则)。

-   `code_context_length`: 代码片段的上下文长度。在处理 `text-code` 模式时，会将代码和文本分开处理，此值表示了代码联系上下文的长度，设大点会使得上下文联系增强，但也会引起输入给大模型的文本长度变长。默认值是 1，表示联系一个上下文片段，约 `chunk_size` 字。参考[分割与查询原则](#分割与查询原则)。

-   `debug`: 是否开启调试模式。
-   `log_queryies`: 是否将所有用户提问存入本地文件，默认会存入 `user_queries.log`，可以方便后续分析。
-   `extensions`: 拓展功能，详见[拓展](#拓展)

-   `files`: 文档内容，是一个数组，每一项可以直接填写一个字符串，表示使用默认 RAG 方案，如果是文件夹中的，可以填写 `folder/doc.md`，这样就会自动读取 `docs/folder/doc.md` 文档。除了字符串，还可以填写对象，对象包含这些属性：

    -   `mode`: 该文档的 RAG 方案，不填时自动使用默认。注意，如果这个文件不是 md 文档，那么会使用 `code-only` 模式。
    -   `path`: 该文档的路径。

## 文档监听

该机器人拥有文档监听功能，当 `docs` 文件夹中的任意文件发生变化（包括新增、删除、移动等）时，会自动定向重建索引，并将新增文件自动添加至 `config.json` 中，避免频繁重启。

**注意！！！**新增文档会使用配置中的默认 RAG 方案，如果既有代码又有文本，记得提前在配置中设为 `text-code` 模式。

由于缓存格式更改，旧版缓存将失效，更新后的首次重启需要重建所有文档。

## 用户提问

在用户提问时，如果提问内容以 `*raw` 开头，那么本次提问将会不参考文档（不进行文档检索）。

## 添加代码种类

插件内置了 `python` `typescript` `tsx` `javascript` `jsx` `html` 等语言的代码解析功能，如果需要添加自己的代码语言，例如 `rust` `java` 等，可以手动安装对应的 `pip` 包，可以在[此处](https://github.com/orgs/tree-sitter/repositories)查看对应语言的包名，安装后在 `splitter.py` 中参考注释操作即可。

## 分割与查询原则

文本的分割原则大致如下：按照段落（双换行，即空行）、标题（\#开头的文本）来进行分割，每个分段长度约 `chunk_size` 字，两个分段之间会有约 `chunk_overlap` 字的重叠，来提高上下文能力。

代码分割的原则大致如下：如果代码长度较短（小于等于 `chunk_size / 2` 字符），那么不会执行分割；如果代码长度较长，会尝试寻找以注释开头的行，并以此行作为分割，此行算作后文的一部分；若寻找不到，会尝试寻找纯空行来分割；如果还是寻找不到，那么会以一个完整的行作为分割，此行算作后文的一部分。每个代码分段的最短长度为 `chunk_size / 2` 字符，最大长度为 `chunk_size * 2` 字符。分割时，会有约 `chunk_overlap * 2` 字符作为上下文。

## 优化

如果你的设备包含 nvidia GPU，可以手动安装 `faiss_gpu` 库，这样的话本地推理会使用 cuda 核心推理，启动速度更快，性能更好。

对于中文文档，合理使用标点，标点应该使用全角字符 （，。：？！等），句尾添加句号，这样可以使得语义分割更加准确。不建议频繁分段，最好保持每段在 200-500 字之间，让每段之间的联系减少，这会提高检索准确率。可以在有序和无序列表之间时不时插入一行纯空行（不包含空格），来让插件分割，提高检索效率和准确率。

对于代码块，应该显式标明代码语言，可以帮助插件准确判断语言类型。如果没有明确标注，会抛出警告，同时使用纯文本的方式解析。

代码中的注释可以使用中文。

## 一些推荐模型

中文文档模型：`BAAI/bge-m3`, `BAAI/bge-large-zh-v1.5`, `moka-ai/m3e-base`

代码模型：`microsoft/codebert-base`, `microsoft/graphcodebert-base`, `Salesforce/codegen-350M-mono`

## 注意事项

由于在启动时会从云端加载模型，并对文档索引，因此该插件会显著提高机器人启动时间，请耐心等待。不过加载完毕后，此插件将几乎不会消耗性能，可以放心使用。

不过，现在插件配备了索引缓存的功能，如果文档没有修改，那么再次启动时将会从缓存读取，很快就能启动，不需要重新索引。而且如果有文档变动，只会定向重新索引修改的文档，未修改的文档不会重新索引。不过，由于索引数据库的限制，不建议部署大型文档，推荐在 100 个文档以内，最好不要超过 500 个。

缓存会保存在 `indices.json` 及 `data` 文件夹中，请勿删除它们，除非你需要全部重新索引

注意，模型会从 HuggingFace 加载，需要科学上网。如果没有办法科学上网，可以在镜像站上下载到本地后，修改 `config.json` 中的 `model_name` 属性为你的模型本地路径。

全部设置完成后，重启机器人，等待文档索引完毕，即可使用。

## 调试

如果输出不符预期，可以在 `config.json` 中将 `debug` 属性改为 `true`，然后重启机器人，这时候输入给大模型的完整 RAG 文档内容及用户提问将会被打印在控制台。

## 拓展

本插件有拓展功能，目拓展的配置都在 `extensions` 属性中，目前包括这些拓展：

### 提问分类拓展

默认的检索排序方式是将文本和代码一个个取出，准确率较低，此拓展可以通过一个本地小模型推理出用户提问是更偏向于文本还是代码，从而使用动态权重来排序，提高检索准确率。

同时，有时候用户的提问是不需要参考文档的，这时候可以略去文档检索步骤，可以省输入 `token`，也可以减少推理压力。此拓展也提供了判断一个提问是否需要参考文档的功能。

本拓展的名称为 `classification`，它包含这些配置：

-   `enable`: 是否启用该拓展。
-   `model_path`: 分类推理模型的文件路径。
-   `need_doc_threshold`: 判断是否需要文档的阈值，模型推理得到的值大于此值就会检索文档，反之不会。如果要求准确度，可以设低一点，如果考虑成本，可以设高一点。

本拓展的模型需要自己训练，我们提供了一个开源项目可以让你更方便地训练模型。[项目地址](https://github.com/unanmed/rag-classification)
