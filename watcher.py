import os
import asyncio
import threading
import watchdog.events as ev
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.events import FileSystemEventHandler

DEBOUNCE_TIME = 5

def find_index(lst, condition):
    return next((i for i, x in enumerate(lst) if condition(x)), -1)

class Debounce:
    def __init__(self, func, wait):
        self.func = func
        self.wait = wait
        self.task = None

    def __call__(self, *args, **kwargs):
        if self.task:
            self.task.cancel()
        self.task = asyncio.create_task(self._debounce(*args, **kwargs))

    async def _debounce(self, *args, **kwargs):
        await asyncio.sleep(self.wait)
        self.func(*args, **kwargs)
        
class DocumentHandler(FileSystemEventHandler):
    root: str
    
    def __init__(self, root: str, watcher):
        super().__init__()
        self.root = root
        self.watcher = watcher
        
    def resolve_path(self, path: str):
        return os.path.normpath(path)
    
    def on_created(self, event):
        if not event.is_directory:
            path = self.resolve_path(event.src_path)
            self.watcher.update_store('add', path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            path = self.resolve_path(event.src_path)
            self.watcher.update_store('delete', path)
    
    def on_modified(self, event):
        if not event.is_directory:
            path = self.resolve_path(event.src_path)
            self.watcher.update_store('modify', path)
    
    def on_moved(self, event):
        if not event.is_directory:
            dest_path = self.resolve_path(event.dest_path)
            src_path = self.resolve_path(event.src_path)
            self.watcher.update_store('delete', src_path)
            self.watcher.update_store('add', dest_path)

class DocumentWatcher:
    observer: BaseObserver
    handler: DocumentHandler
    root: str
    
    reindex_task: asyncio.Future = None
    reindex_list: list[tuple[str, str]] = list()
    reindexing: bool = False
    
    def __init__(self, root: str, path: str, parser):
        self.root = root
        self.parser = parser
        self.observer = Observer()
        self.handler = DocumentHandler(root, self)
        self.observer.schedule(self.handler, path, recursive=True)
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # 在独立线程中运行事件循环
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.thread.start()
        
    def start_loop(self):
        """ 事件循环在独立线程中运行 """
        self.loop.run_forever()
        
    def update_store(self, mode: str, path: str):
        if self.reindex_task:
            self.reindex_task.cancel()
        else:
            print("Received document update, waiting for reindexing...")
            
        # 将文件修改操作添加至列表
        idx = find_index(self.reindex_list, lambda x: x[0] == path)
        if idx != -1:
            _, m = self.reindex_list[idx]
            if mode == 'modify':
                self.reindex_list[idx] = [path, 'modify']
            elif mode == 'delete':
                self.reindex_list[idx] = [path, 'delete']
            elif m == 'delete' and mode == 'add':
                self.reindex_list[idx] = [path, 'modify']
        else:
            self.reindex_list.append([path, mode])
        
        self.reindex_task = asyncio.run_coroutine_threadsafe(self.reindex(), self.loop)
        
    async def reindex(self):
        await asyncio.sleep(DEBOUNCE_TIME)
        
        # 不能同时重建俩索引，因此需要等待
        while self.reindexing:
            await asyncio.sleep(DEBOUNCE_TIME)
            
        self.reindex_task = None
        to_reindex = self.reindex_list
        self.reindex_list = list()
        self.reindexing = True
        self.parser.reindex(to_reindex)
        self.reindexing = False

    def start(self):
        self.observer.start()
        
    def end(self):
        self.observer.stop()
        self.observer.join()
