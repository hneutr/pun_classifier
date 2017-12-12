from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from classifiers.baseline import BaselinePunClassifier
from classifiers.pun_detection_with_features import PunDetectionWithFeaturesClassifier
from classifiers.pun_location_with_features import PunLocationWithFeaturesClassifier
from classifiers.pun_rnn import PunRNNClassifier
from classifiers.pun_rnn_detection import PunRNNDetectionClassifier
from classifiers.scikit_wrapper import ScikitWrapperClassifier
from classifiers.sliding_window import PunSlidingWindowClassifier
from classifiers.voting_classifier import PunVotingClassifier
from pun_data import DetectionData, LocationData


class ServerMain:
    def __init__(self):
        # Get classifiers to use for DETECTION
        baselinePunClassifier = BaselinePunClassifier(type="Detection")
        punRnnDetectionClassifier = PunRNNDetectionClassifier()
        punDetectionWithFeaturesClassifier = PunDetectionWithFeaturesClassifier()
        punVotingClassifier = PunVotingClassifier(type="Detection", classifiers=[baselinePunClassifier,punRnnDetectionClassifier,punDetectionWithFeaturesClassifier])
        self.detectionClassifiers = [baselinePunClassifier, punRnnDetectionClassifier, punDetectionWithFeaturesClassifier, punVotingClassifier]

        # Get classifiers to use for TYPE DETECTION
        baselinePunTypeClassifier = BaselinePunClassifier(type="Detection")
        punRnnDetectionTypeClassifier = PunRNNDetectionClassifier()
        punDetectionWithFeaturesTypeClassifier = PunDetectionWithFeaturesClassifier()
        typeClassifiers = [
            baselinePunClassifier,
            punRnnDetectionClassifier,
            punDetectionWithFeaturesClassifier
        ]
        punVotingTypeClassifier = PunVotingClassifier(type="Detection", classifiers=typeClassifiers)
        self.detectionTypeClassifiers = [baselinePunTypeClassifier, punRnnDetectionTypeClassifier,
                                     punDetectionWithFeaturesTypeClassifier, punVotingTypeClassifier]

        # Get classifiers to use for LOCATION PROBABILITIES
        baselinePunLocationClassifier = BaselinePunClassifier(type="Location")
        punRnnLocationClassifier = PunRNNClassifier(output="word")
        punDecisionTreeClassifier = PunLocationWithFeaturesClassifier(output="word")
        punSlidingWindowClassifier = PunSlidingWindowClassifier(output="word")
        punVotingLocationClassifier = PunVotingClassifier(type="Location", classifiers=[baselinePunLocationClassifier, punRnnLocationClassifier, punDecisionTreeClassifier, punSlidingWindowClassifier, ])
        self.locationClassifiers = [baselinePunLocationClassifier, punRnnLocationClassifier, punDecisionTreeClassifier, punSlidingWindowClassifier, punVotingLocationClassifier]

    # Initialize server so that it will be able to make predictions
    # This will involve training the needed classifiers
    def do_init (self):
        # initialize classifiers for DETECTION
        graphic = 'combined'
        detectionData = DetectionData(graphic, False)
        for classifier in self.detectionClassifiers:
            classifier.train(detectionData.x_train[:100], detectionData.y_train[:100])

        #initialize classifiers for TYPE DETECTION
        graphic = 'both'
        detectionData = DetectionData(graphic, False)
        for classifier in self.detectionTypeClassifiers:
            classifier.train(detectionData.x_train[:100], detectionData.y_train[:100])

        #initialize classifiers for LOCATION PROBABILITIES
        graphic = 'combined'
        locationData = LocationData(graphic)
        for classifier in self.locationClassifiers:
            classifier.train(locationData.x_train[:100], locationData.y_train[:100])

    # Detect the probabilities of each class (pun or non pun) for the sentence passed in
    def do_detection (self, request):
        results = []
        for classifier in self.detectionClassifiers:
            results.append(classifier.test_with_probabilities([request])[0])
        return results

        # Detect the probabilities of each class (pun or non pun) for the sentence passed in

    def do_type_detection(self, request):
        results = []
        for classifier in self.detectionTypeClassifiers:
            results.append(classifier.test_with_probabilities([request])[0])
        return results

    def do_location(self, request):
        results = []
        for classifier in self.locationClassifiers:
            results.append(classifier.test_with_probabilities([request])[0])
        return results

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

    def finish_repsonse(self, response):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(bytes(json.dumps(response), "utf8"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        request = self.rfile.read(content_length).decode("utf8")  # <--- Gets the data itself
        print(request)  # <-- Print post data
        request = request.strip().split(" ")

        if self.path == '/detection':
            probabilities = server_main.do_detection(request)
            response = {
                "baseline": {"non-pun": probabilities[0][0], "pun": probabilities[0][1]},
                "rnn": {"non-pun": probabilities[1][0], "pun": probabilities[1][1]},
                "features": {"non-pun": probabilities[2][0], "pun": probabilities[2][1]},
                "voting": {"non-pun": probabilities[3][0], "pun": probabilities[3][1]}
            }
            self.finish_repsonse(response)

        elif self.type == '/type':
            probabilities = server_main.do_type_detection(request)
            response = {
                "baseline": {"non-pun": probabilities[0][0], "homographic": probabilities[0][1], "heterographic": probabilities[0][2]},
                "rnn": {"non-pun": probabilities[1][0], "homographic": probabilities[1][1], "heterographic": probabilities[1][2]},
                "features": {"non-pun": probabilities[2][0],"homographic": probabilities[2][1], "heterographic": probabilities[2][2]},
                "voting": {"non-pun": probabilities[3][0], "homographic": probabilities[3][1], "heterographic": probabilities[3][2]},
            }
            self.finish_repsonse(response)

        elif self.type == '/location':
            #TODO
            response = None
            self.finish_repsonse(response)


        else : # not a good endpoint
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