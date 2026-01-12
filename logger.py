# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for this UNetonly SR project. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
import logging
from contextlib import contextmanager
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = int(sec%60)
    return h,m,s

class TimeFilter(logging.Filter):

    def filter(self, record):
        try:
          start = self.start
        except AttributeError:
          start = self.start = time.time()

        time_elapsed = get_time(time.time() - start)

        record.relative = "{0}:{1:02d}:{2:02d}".format(*time_elapsed)

        # self.last = record.relativeCreated/1000.0
        return True

class Logger(object):
    def __init__(self, rank=0, log_dir=".log"):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        self.console = Console()
        
        if self.rank == 0:
            os.makedirs(log_dir, exist_ok=True)

            # 生成带时间戳的日志文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"{timestamp}.txt"
            log_file = open(os.path.join(log_dir, log_filename), "w")
            file_console = Console(file=log_file, width=150)
            
            logging.basicConfig(
                level=logging.INFO,
                format="(%(relative)s) %(message)s",
                datefmt="[%X]",
                force=True,
                handlers=[
                    RichHandler(show_path=False, console=self.console),
                    RichHandler(console=file_console, show_path=False)
                ],
            )
            # https://stackoverflow.com/questions/31521859/python-logging-module-time-since-last-log
            log = logging.getLogger()
            [hndl.addFilter(TimeFilter()) for hndl in log.handlers]

    def info(self, string, *args):
        if self.rank == 0:
            logging.info(string, *args)

    def warning(self, string, *args):
        if self.rank == 0:
            logging.warning(string, *args)

    def error(self, string, *args):
        if self.rank == 0:
            logging.error(string, *args)
    
    @contextmanager
    def progress_bar(self, iterable, desc="", total=None):
        """
        使用 Rich Progress 创建彩色进度条
        
        Args:
            iterable: 可迭代对象
            desc: 描述文字
            total: 总数（如果 iterable 没有 __len__）
        
        Yields:
            进度条包装的迭代器
            
        Example:
            with log.progress_bar(dataloader, desc="Training") as pbar:
                for batch in pbar:
                    # 使用 pbar.update_postfix(loss=0.123) 更新信息
                    ...
        """
        if self.rank != 0:
            yield iterable
            return
            
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        # 创建 Rich Progress
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(complete_style="red", finished_style="bright_red"),
            TextColumn("[magenta]{task.completed}/{task.total}"),  # 自定义进度数字颜色
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[postfix]}"),
            console=self.console,
            transient=False,  # 保留历史进度条
        )
        
        with progress:
            task_id = progress.add_task(desc, total=total, postfix="")
            
            class ProgressWrapper:
                def __init__(self, progress_obj, task_id, iterable):
                    self.progress = progress_obj
                    self.task_id = task_id
                    self.iterable = iterable
                    
                def __iter__(self):
                    for item in self.iterable:
                        yield item
                        self.progress.advance(self.task_id)
                
                def update_postfix(self, **kwargs):
                    """更新进度条后缀信息，如 loss 等"""
                    postfix_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                    self.progress.update(self.task_id, postfix=postfix_str)
            
            wrapper = ProgressWrapper(progress, task_id, iterable)
            yield wrapper
