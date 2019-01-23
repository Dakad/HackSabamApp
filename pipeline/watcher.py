#! /usr/bin/python
import os
import time
from threading import Thread

from config import Config


class Watcher(object):

    before = dict()

    @staticmethod
    def init(path=Config.TASK_DIR, idle=Config.IDLE_TIME):

        Watcher.path_to_watch = path
        Watcher.idle_time = idle

        os.makedirs(Watcher.path_to_watch, exist_ok=True)

        if Config.DEBUG:
            os.makedirs(Config.PROCESS_DIR, exist_ok=True)

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

            for upload in new_files:
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
            pass


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--path", type=str, help="Path to watch")
    arg_parser.add_argument(
        "-i", "--idle", type=int, default=3, help="Idle time between each watch")
    args = vars(arg_parser.parse_args())

    Watcher.init(**args)
    Watcher.watch()
