from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pickle

from classifiers.baseline import BaselinePunClassifier
from classifiers.pun_detection_with_features import PunDetectionWithFeaturesClassifier
from classifiers.pun_rnn import PunRNNClassifier
from sklearn.model_selection import train_test_split
#from pun_data import DetectionData

from eval import Eval

SEED = 20171110

class DetectionDataType:
    def __init__(self):
        path = "./data/pickles/test-1.pkl.gz"

        with open(path, 'rb') as f:
            self.x_set, self.y_set = pickle.load(f)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_set, self.y_set, random_state=SEED)

class ServerMain:
    def __init__(self):
        self.baselineClassifier = BaselinePunClassifier()
        self.featuresClassifier = PunDetectionWithFeaturesClassifier()
        self.rnnClassifier = PunRNNClassifier()

    # Initialize server so that it will be able to make predictions
    # This will involve training the needed classifiers
    def do_init (self):
        #even = True
        #graphic = 'homographic'
        detectionData = DetectionDataType()
        self.baselineClassifier.train(detectionData.x_train, detectionData.y_train)
        self.featuresClassifier.train(detectionData.x_train, detectionData.y_train)
        #self.rnnClassifier.train(detectionData.x_train, detectionData.y_train)

    # Detect the probabilities of each class (pun or non pun) for the sentence passed in
    # TODO: Which classifier to use here? Or both?
    def do_detection (self, request):
        request_array = request.split(" ")
        baselineResult = self.baselineClassifier.test_with_probabilities([request_array])[0]
        featuresResult = self.featuresClassifier.test_with_probabilities([request_array])[0]
        #rnnResult = self.rnnClassifier.test_with_probabilities([request_array])[0]
        return (baselineResult, featuresResult)

class S(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
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
            response = {
                "baseline": {"non-pun": probabilities[0][0], "pun1": probabilities[0][1], "pun2": probabilities[0][2]},
                "features": {"non-pun": probabilities[1][0], "pun1": probabilities[1][1], "pun2": probabilities[1][2]}
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
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
    server_address = ('', 8081)
    httpd = HTTPServer(server_address, S)
    print('running server...')
    httpd.serve_forever()

server_main = ServerMain()
run()
