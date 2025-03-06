# LangBotPluginDocument

## 安装

配置完成 [LangBot](https://github.com/RockChinQ/LangBot) 主程序后使用管理员账号向机器人发送命令即可安装：

```
!plugin get https://github.com/unanmed/LangBotPluginDocument
```
或查看详细的[插件安装说明](https://docs.langbot.app/plugin/plugin-intro.html#%E6%8F%92%E4%BB%B6%E7%94%A8%E6%B3%95)

## 使用

<!-- 插件开发者自行填写插件使用说明 -->

打开 `config.json`，填写文档提示词和用户问题提示词。发送给 AI 大模型的内容将会是：

> {reference_prompt}\n{查询到的文档内容}\n{question_prompt}用户提问

然后将你的文档放入 docs 文件夹，并在 `config.json` 里面填写要读取的文件，重启机器人即可，之后的聊天中，机器人将会自动引用文档中的内容，并参考文档给出回复。

由于在启动时会从云端加载模型，并对文档索引，因此该插件会显著提高机器人启动时间，请耐心等待。不过加载完毕后，此插件将几乎不会消耗性能，可以放心使用。

可以手动修改 `config.json` 中的 `model_name` 属性来修改 RAG 嵌入模型。

一些推荐模型：

- 中文文档：`BAAI/bge-m3`, `BAAI/bge-large-zh-v1.5`
- 含有代码的文档：`microsoft/codebert-base`, `Salesforce/codet5-base`

注意，模型会从 HuggingFace 加载，需要科学上网。如果没有办法科学上网，可以在镜像站上下载到本地后，修改 `config.json` 中的 `model_name` 属性为你的模型本地路径。

## 调试

如果输出不符预期，可以在 `config.json` 中将 `debug` 属性改为 `true`，然后重启机器人，这时候输入给机器人的完整 RAG 文档内容及用户提问将会被打印在控制台。
