#! /usr/bin/env python

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application

from yello import YelloHandler

# Create attribute 'PORT' in options object
define("PORT", default=8080, help="Port to run the WebApp", type=int)


def main():
    options.parse_command_line()
    app = Application([(r"/", YelloHandler)])
    http_server = HTTPServer(app)
    http_server.listen(options.PORT)
    print('Listening on http://localhost:%i' % options.PORT)
    IOLoop.current().start()


if __name__ == "__main__":
    main()
