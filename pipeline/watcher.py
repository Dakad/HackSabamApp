#! /usr/bin/python
import os
import time

_DEFAULT = {
    "task_path": "../data/tasks",
    "idle_time": 60 * 10,
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
            print(new_files)
            Watcher.before = after
            time.sleep(10)


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--path", type=str, help="Path to watch")
    arg_parser.add_argument(
        "-i", "--idle", type=int, help="Idle time between each watch")
    args = vars(arg_parser.parse_args())

    Watcher.init(**args)
    Watcher.watch()
