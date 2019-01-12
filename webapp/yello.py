#! /usr/bin/env python

import tornado.web


class YelloHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Yello, World !')
