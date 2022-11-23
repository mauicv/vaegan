import toml


def load_config():
    return toml.load('./config.toml')
