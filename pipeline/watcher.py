#! /usr/bin/python
import os
import time
from threading import Thread

_DEFAULT = {
    "task_path": "../data/tasks",
    "idle_time": 60 * 10,  # 10 mins
}


class Watcher():

    path_to_watch = _DEFAULT['task_path']
    idle_time = _DEFAULT['idle_time']
    before = dict()

    @staticmethod
    def init(path=None, idle=None):
        if(path is not None):
            Watcher.path_to_watch = path
        if(idle is not None):
            Watcher.idle_time = idle

        os.makedirs(Watcher.path_to_watch, exist_ok=True)

        Watcher.before = dict()
        Watcher.before = dict([
            (f, None) for f in os.listdir(Watcher.path_to_watch)
        ])

    @staticmethod
    def watch():
        after = dict()
        new_files = []
        while True:
            after = dict([
                (f, None) for f in os.listdir(Watcher.path_to_watch)
            ])
            new_files = [f for f in after if not f in Watcher.before]

            for (upload, _) in new_files:
                Task(upload).start()

            Watcher.before = after
            time.sleep(Watcher.idle_time)


class Task(Thread):
    """Thread Task for an new upload file to process

    Arguments:
        Thread {string} -- The upload filename used also as name
    """
    _upload_id = None

    def __init__(self, id):
        Thread.__init__(self)
        self.name = "Task #"+id
        self._upload_id = id
        print(self.name)

    def run(self):
        try:
            print("New Task to do : Upload #%s" % self._upload_id)
            # TODO : 1 - Optimise the uploaded image
            # TODO : 2 - Send the optimised img to OCR
            # TODO : 3 - Parse the OCR result
            # TODO : 4 - Store the parsed result in NoSQL DB

        finally:
            # TODO: Notify the Watcher about the failed task


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--path", type=str, help="Path to watch")
    arg_parser.add_argument(
        "-i", "--idle", type=int, help="Idle time between each watch")
    args = vars(arg_parser.parse_args())

    Watcher.init(**args)
    Watcher.watch()
