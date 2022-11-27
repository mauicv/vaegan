import toml


def load_config(path='./config.toml'):
    return toml.load(path)


class ConfigMixin:
    def __init__(self, path):
        self.cfg = load_config()