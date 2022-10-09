from dataset import get_dataset
import toml


if __name__ == "__main__":
    config = toml.load('./config.toml')
    dataset, loader = get_dataset(config)
