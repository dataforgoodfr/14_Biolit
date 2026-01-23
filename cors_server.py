#!/usr/bin/env python3
"""Simple HTTP server with CORS support for serving images to Label Studio."""

import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


if __name__ == '__main__':
    display_host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    server = HTTPServer(('0.0.0.0', 8000), CORSRequestHandler)
    print(f'Serving on http://{display_host}:8000')
    server.serve_forever()
