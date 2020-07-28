# -*- coding: utf-8 -*-

import time
import functools
import json

import logging

from tornado import gen
from tornado import httpclient
from tornado import httputil
from tornado import ioloop
from tornado import websocket

APPLICATION_JSON = 'application/json'

DEFAULT_CONNECT_TIMEOUT = 60
DEFAULT_REQUEST_TIMEOUT = 60


class WebSocketClient(object):
    """Base for web socket clients.
    """

    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2

    def __init__(self, connect_timeout=DEFAULT_CONNECT_TIMEOUT, request_timeout=DEFAULT_REQUEST_TIMEOUT):

        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._ws_connection = None
        self._connect_status = self.DISCONNECTED

    def connect(self, url):
        """Connect to the server.
        :param str url: server URL.
        """

        logging.info("this client attempt to connect to the server")

        self._connect_status = self.CONNECTING
        headers = httputil.HTTPHeaders({'Content-Type': APPLICATION_JSON})
        request = httpclient.HTTPRequest(url=url,
                                         connect_timeout=self.connect_timeout,
                                         request_timeout=self.request_timeout,
                                         headers=headers)
        # this set client sending and receiving max message size limit
        ws_conn = websocket.WebSocketClientConnection(request, max_message_size=1024 * 1024 * 1024)
        ws_conn.connect_future.add_done_callback(self._connect_callback)

    def send(self, data):
        """Send message to the server
        :param str data: message.
        """

        logging.info("this client sending message to the server")

        if not self._ws_connection:
            raise RuntimeError('Web socket connection is closed.')

        if self._ws_connection:
            self._ws_connection.write_message(json.dumps(data))

    def close(self, reason='unknown'):
        """Close connection.
        """

        logging.info("this client closed connection")

        if not self._ws_connection:
            raise RuntimeError('Web socket connection is already closed.')

        if self._connect_status != self.DISCONNECTED:
            self._connect_status = self.DISCONNECTED
            self._ws_connection and self._ws_connection.close()
            self._ws_connection = None
            self._on_connection_close(reason)

    def _connect_callback(self, future):
        if future.exception() is None:
            self._connect_status = self.CONNECTED
            self._ws_connection = future.result()
            self._on_connection_success()
            self._read_messages()
        else:
            self._on_connection_error(future.exception())
            self.close(future.exception())

    def is_connected(self):
        return self._ws_connection is not None

    @gen.coroutine
    def _read_messages(self):
        while True:
            msg = yield self._ws_connection.read_message()
            if msg is None:
                self._on_connection_close()
                self.close()
                break

            self._on_message(msg)

    def _on_message(self, msg):
        """This is called when new message is available from the server.
        :param str msg: server message.
        """

        pass

    def _on_connection_success(self):
        """This is called on successful connection ot the server.
        """

        pass

    def _on_connection_close(self, reason="unknown"):
        """This is called when server closed the connection.
        """

        pass

    def _on_connection_error(self, exception):
        """This is called in case if connection to the server could
        not established.
        """

        pass