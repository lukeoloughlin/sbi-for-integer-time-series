import os


def make_folder(folder):

    if not os.path.exists(folder):
        os.makedirs(folder)


class Logger(object):

    def __init__(self, filename):
        self.f = open(filename, 'w')
        self.round = 0

    def write(self, msg):
        self.f.write(msg)

    def increment_round(self):
        self.round += 1
        self.f.write(f"##########\nRound {self.round}:\n##########\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
        return False
