import os, pickle
class Cache(object):
    def __init__(self, path):
        self.path = path
    def load(self):
        if os.path.isfile(self.path):
            print('[LOG] loading data..')
            with open(self.path, 'rb') as f:
                return pickle.load(f)
        print('[WRN] data not found in', self.path)
        return None
    def save(self, data):
        with open(self.path, 'wb') as f:
            pickle.dump(data, f)
        print('[LOG] data saved..')

