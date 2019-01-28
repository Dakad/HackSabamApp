#! /usr/bin/env python

import os
from datetime import datetime
from random import choice
from string import ascii_lowercase, digits
from urllib.parse import unquote


import tornado.web


_UPLOAD_FILENAME_FORMAT = "%Y%m%d_%H%M%S_%f"


class UploadHandler(tornado.web.RequestHandler):

    def initialize(self, db, upload_dir, task_dir):
        self._db = db
        self._upload_dir = upload_dir
        self._task_dir = task_dir

    def _clean_filename(self, filename):
        """Remove any "evil" special characters

        Arguments:
            filename {string} -- The name of the file

        Returns:
            string -- Clean filename
        """

        idx = filename.rfind(".")
        f_name = filename[:idx].replace(".", "")
        f_name += str(datetime.now().time()).split('.')[1]
        f_name += filename[idx:]
        filename = f_name.replace("/", "")

        return filename

    @staticmethod
    def random_name():
        return ''.join(choice(ascii_lowercase + digits) for _ in range(10))

    def get(self):
        self.render("upload.html", title="Send me all pics !")

    def post(self):
        try:
            upload_id = datetime.now().strftime(_UPLOAD_FILENAME_FORMAT)
            uploads = self.request.files['uploads']

            # TODO: Write upload user data in SQL DB

            for idx, upload in enumerate(uploads):
                up_ext = os.path.splitext(upload['filename'])[1]
                # Just use the idx with 000,padded as name
                upload_name = str(idx).zfill(3) + up_ext
                upload_f_name = os.path.join(self._upload_dir, upload_name)
                with open(upload_f_name, 'wb') as upload_file:
                    upload_file.write(upload['body'])

            # Write a task file for the pipeline
            new_task = os.path.join(self._task_dir, upload_id)
            open(new_task, 'a').close()
        finally:
            self.redirect('/upload/')


@tornado.web.stream_request_body
class UploadStreamHandler(UploadHandler):
    def initialize(self):
        self.bytes_read = 0

    def data_received(self, chunk):
        pass
