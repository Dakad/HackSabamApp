import os
import logging
from logging.handlers import RotatingFileHandler

from config import Config


def config_log(log_instance="app"):
    if Config.LOG_TO_STDOUT:
        stream_handler = logging.StreamHandler()
        logging.root.addHandler(stream_handler)
    else:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        print('%s.log' % log_instance)
        my_file_handler = RotatingFileHandler(
            filename=os.path.join(Config.LOG_DIR, '%s.log' % log_instance),
            maxBytes=1024*10,
            backupCount=3,
            encoding='utf-8'
        )
        my_file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s:[in %(pathname)s:%(lineno)d] %(message)s '))
        logging.root.addHandler(my_file_handler)

    logging.root.setLevel(logging.INFO)
    logging.root.info('[Config] Logging : DONE ')


def run_watcher():
    from pipeline.watcher import Watcher
    Watcher.init()
    Watcher.watch()


def run_web_app():
    import webapp as webappMain
    webappMain.run(Config)


def main(run='all'):
    config_log(run)

    if run in ("pipeline", "all"):
        run_watcher()
    if run in ("web", "all"):
        run_web_app()


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-r", "--run",
        choices=['pipeline', 'web', 'all'],
        const='all', default='all', nargs='?',
        help="Which part of the app to run : pipeline, web, all (by default: %(default)s)")

    args = vars(arg_parser.parse_args())
    main(**args)
