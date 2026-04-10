from http.server import HTTPServer, SimpleHTTPRequestHandler

# Problem, when importing data to Label Studio, is CORS. Label Studio's frontend (running on port 8080) is trying to load images from your server on port 8081, and Python's built-in http.server doesn't send CORS headers, so the browser blocks it. This custom handler adds the necessary headers to allow cross-origin requests.

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass

HTTPServer(("localhost", 8081), CORSHandler).serve_forever()
