import os
import typing
from pkg.command import operator, entities, errors
from pkg.core.app import Application
from .main import LangBotPluginDocument

def get_plugin(ap: Application) -> LangBotPluginDocument | None:
    plugins = ap.plugin_mgr.plugins(enabled=True)
    doc_plugin = None
    for plugin in plugins:
        if plugin.plugin_name == 'LangBotPluginDocument':
            doc_plugin = plugin.plugin_inst
            break
        
    return doc_plugin

@operator.operator_class(
    name="document",
    help="操作机器人的文档库",
    usage='!document <操作> [参数]\n操作包括：\nappend, list'
)
class DocumentOperator(operator.CommandOperator):    
    async def execute(self, context):
        yield entities.CommandReturn(
            text="用法参考：\n" +
            "!document append <文档路径> <新增内容>\n" +
            "!document list\n\n"+
            "使用 !cmd <append | list> 查看详细用法"
        )

@operator.operator_class(
    name="append",
    privilege=2,
    help="追加内容到已有文档",
    usage='!document append <文档路径> <新增内容>\n' +
        '示例：!document append example.md ## 如何使用此插件？\n\n参考README.md',
    parent_class=DocumentOperator
)
class DocumentAppendOperator(operator.CommandOperator):
    """追加内容到已有文档"""
    
    async def execute(
        self,
        context: entities.ExecuteContext
    ) -> typing.AsyncGenerator[entities.CommandReturn, None]:
        doc_plugin = get_plugin(self.ap)
        
        if doc_plugin is None:
            yield entities.CommandReturn(text="请先启用 LangBotPluginDocument 插件！")
            
        if len(context.crt_params) == 0:
            yield entities.CommandReturn(error=errors.CommandOperationError('缺少参数'))
            return
        
        path = context.crt_params[0]
        content = " ".join(context.crt_params[1:])
        current_dir = doc_plugin.current_dir
        files = doc_plugin.parser.config["files"]
        
        if files.count(path) == 0:
            yield entities.CommandReturn(text="未找到要追加的文件！")
        
        with open(os.path.join(current_dir, "docs", path), 'a', encoding='utf-8') as f:
            f.write(content + "\n")
        
        yield entities.CommandReturn(text="成功追加文档内容！")
        
@operator.operator_class(
    name="list",
    privilege=2,
    help="列出所有文档",
    usage='!document list',
    parent_class=DocumentOperator
)
class DocumentAppendOperator(operator.CommandOperator):
    """列出当前所有文档"""
    
    async def execute(
        self,
        context: entities.ExecuteContext
    ) -> typing.AsyncGenerator[entities.CommandReturn, None]:
        doc_plugin = get_plugin(self.ap)
        
        if doc_plugin is None:
            yield entities.CommandReturn(text="请先启用 LangBotPluginDocument 插件！")
            
        files = doc_plugin.parser.config["files"]
        
        yield entities.CommandReturn(text="文档列表：\n" + "\n".join(files))