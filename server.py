from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from classifiers.pun_detection_with_features import PunDetectionWithFeaturesClassifier
from pun_data import DetectionData

class ServerMain:
    def __init__(self):
        self.featuresClassifier = PunDetectionWithFeaturesClassifier()

    # Initialize server so that it will be able to make predictions
    # This will involve training the needed classifiers
    def do_init (self):
        even = True
        graphic = 'homographic'
        detectionData = DetectionData(graphic, even)
        self.featuresClassifier.train(detectionData.x_train, detectionData.y_train)

    # Detect the probabilities of each class (pun or non pun) for the sentence passed in
    # TODO: Which classifier to use here? Or both?
    def do_detection (self, request):
        request_array = " ".split(request)
        return self.featuresClassifier.test_with_probabilities(request_array)[0]

class S(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(bytes("Hello World", "utf8"))
        return

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        request = self.rfile.read(content_length).decode("utf8")  # <--- Gets the data itself
        print(request)  # <-- Print post data

        if self.path == '/detection':
            probabilities = server_main.do_detection(request)
            response = {"non-pun": probabilities[0], "pun": probabilities[1]}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes(json.dumps(response), "utf8"))
        else :
            self.send_response(403)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

        return


def run():
    server_main.do_init()
    print('starting server...')

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, S)
    print('running server...')
    httpd.serve_forever()

server_main = ServerMain()
run()