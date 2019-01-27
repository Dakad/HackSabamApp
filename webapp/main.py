#! /usr/bin/env python
import os

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler

from upload import UploadHandler, UploadStreamHandler


class My404Handler(RequestHandler):
    def prepare(self):
        self.set_status(404)
        self.render(
            "404.html", msg="The page, you are looking for, does not exist.")


class YelloHandler(RequestHandler):
    def get(self):
        self.render("home.html", msg="Yello, World !")


def _make_app(options):
    handlers = [
        (r"/", YelloHandler),
        (r"/upload/stream", UploadStreamHandler),
        (r"/upload/", UploadHandler, dict(
            db=None,
            task_dir=options.TASK_DIR,
            upload_dir=options.UPLOAD_DIR,
        )),
        # (r"/(.*)", My404Handler)
    ]

    settings = dict(
        app_title=options.WEBAPP_TITLE,
        default_handler_class=My404Handler,
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        template_path=os.path.join(os.path.dirname(__file__), "template"),
        debug=True  # No cache for the compiled templates

    )
    return Application(handlers, **settings)


def run(options):
    # Tornado configures logging.
    # _config_logging()
    app = _make_app(options)
    # http_server = HTTPServer(app)
    app.listen(options.WEBAPP_PORT)
    print('Listening on http://localhost:%i' % options.WEBAPP_PORT)
    IOLoop.current().start()


if __name__ == "__main__":
    from tornado.options import define, options

    # Create attribute 'port' in options object
    define("WEBAPP_PORT", default=8081, help="Port to run the WebApp", type=int)

    define("WEBAPP_TITLE", default="Hack Sabam APP",
           help="WebApp Title", type=int)

    define("TASK_DIR",
           default=os.path.join(os.path.realpath('../'), 'data', 'tasks/'),
           help="Path to store the new task for the pipeline",
           type=str)

    define("UPLOAD_DIR",
           default=os.path.join(os.path.realpath('../'),
                                'data', 'uploads/'),
           help="Path to store the uploads folder to store the upload img",
           type=str)

    options.parse_command_line()

    run(options)
