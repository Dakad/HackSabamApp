#! /usr/bin/env python
import os

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application, RequestHandler

from upload import UploadHandler, UploadStreamHandler

# Create attribute 'port' in options object
define("port", default=8080, help="Port to run the WebApp", type=int)

define("task_folder", 
    default=os.path.join(os.path.realpath('../'), 'pipeline', 'tasks/'), 
    help="Path to store the new task for the pipeline", 
    type=str)

define("upload_folder", 
    default=os.path.join(os.path.realpath('../'), 'pipeline', 'uploads/'), 
    help="Path to store the uploads folder to store the upload img", 
    type=str)



class My404Handler(RequestHandler):
    def prepare(self):
        self.set_status(404)
        self.render(
            "404.html", msg="The page you are looking for does not exist.")


class YelloHandler(RequestHandler):
    def get(self):
        self.render("home.html", msg="Yello, World !")


def _make_app():
    handlers = [
        (r"/", YelloHandler),
        (r"/upload/stream", UploadStreamHandler),
        (r"/upload/", UploadHandler, dict(
            db=None, 
            task_folder = options.task_folder,
            upload_folder = options.upload_folder,
        )),
        # (r"/(.*)", My404Handler)
    ]

    settings = dict(
        app_title="App Hack Sabam",
        default_handler_class=My404Handler,
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        template_path=os.path.join(os.path.dirname(__file__), "template"),
        debug=True  # No cache for the compiled templates

    )
    return Application(handlers, **settings)


def run():
    # Tornado configures logging.

    options.parse_command_line()
    app = _make_app()
    # http_server = HTTPServer(app)
    app.listen(options.port)
    print('Listening on http://localhost:%i' % options.port)
    IOLoop.current().start()



if __name__ == "__main__":
    run()