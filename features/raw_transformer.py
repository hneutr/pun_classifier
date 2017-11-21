class RawTransformer:
    def __init__(self, features):
        """
        takes a dictionary of strings (keys) to functions (simple
        transformations)
        """
        self.features = features

    def fit(self, xs, ys):
        return self

    def transform(self, xs):
        output = {}
        for key, function in self.features.items():
            output[key] = [function(x) for x in xs]

        return output
