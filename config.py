
import os

from dotenv import load_dotenv

base_dir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(base_dir, '.env'))


_DEF_VAL = {
    'LOCAL_DB': 'sqlite:///' + os.path.join(base_dir, 'data', 'app.db'),
    'OCR_LANGUAGES': 'eng, fra, spa',
    'LOG_DIR': os.path.join(base_dir, 'logs'),
    "UPLOAD_DIR": os.path.join(base_dir, 'data', 'uploads'),
    "TASK_DIR": os.path.join(base_dir, 'data', 'tasks'),
    "FAILED_DIR": os.path.join(base_dir, 'data', 'failed'),
    "PROCESS_DIR": os.path.join(base_dir, 'data', 'optmised'),
    "IDLE_TIME": 60 * 10,  # 10 mins
    'IMAGE_TYPES': "jpg,jpeg,png"
}


def _rand_string():
    from random import choice
    from string import ascii_letters, digits

    pattern = digits+ascii_letters + "?&~#-_@%$µ*!§€+"
    return ''.join(choice(pattern) for i in range(39))


class Config(object):
    DEFAULTS = _DEF_VAL

    DEBUG = os.environ.get('DEBUG', True)

    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT', False)

    LOG_DIR = os.environ.get('LOG_DIR', _DEF_VAL['LOG_DIR'])
    UPLOAD_DIR = os.environ.get('UPLOAD_DIR', _DEF_VAL['UPLOAD_DIR'])
    TASK_DIR = os.environ.get('TASK_DIR', _DEF_VAL['TASK_DIR'])
    FAILED_DIR = os.environ.get('FAILED_DIR', _DEF_VAL['FAILED_DIR'])
    PROCESS_DIR = os.environ.get('PROCESS_DIR', _DEF_VAL['PROCESS_DIR'])
    IDLE_TIME = int(os.environ.get('IDLE_TIME', _DEF_VAL['IDLE_TIME']))
    OCR_LANGS = os.environ.get(
        'OCR_LANGS', _DEF_VAL['OCR_LANGUAGES']).split(',')

    DB_SQL_URI = os.environ.get('DB_SQL_URI', _DEF_VAL['LOCAL_DB'])
    DB_SQL_HOST = os.environ.get('DB_SQL_HOST')
    DB_SQL_PORT = int(os.environ.get('DB_SQL_PORT'))
    DB_SQL_USER = os.environ.get('DB_SQL_USER')
    DB_SQL_PWD = os.environ.get('DB_SQL_PWD')
    DB_SQL_DB = os.environ.get('DB_SQL_DB')

    DB_NOSQL_URI = os.environ.get('DB_NOSQL_URI')
    DB_NOSQL_HOST = os.environ.get('DB_NOSQL_HOST')
    DB_NOSQL_PORT = int(os.environ.get('DB_NOSQL_PORT'))
    DB_NOSQL_USER = os.environ.get('DB_NOSQL_USER')
    DB_NOSQL_PWD = os.environ.get('DB_NOSQL_PWD')
    DB_NOSQL_DB = os.environ.get('DB_NOSQL_DB')

    ACCEPT_IMAGE_TYPES = os.environ.get(
        'ACCEPT_IMAGE_TYPES', _DEF_VAL['IMAGE_TYPES']).split(',')

    # SECRET_KEY = os.environ.get('SECRET_KEY') or _rand_string()
