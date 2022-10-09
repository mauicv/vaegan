from trainer import Trainer
from dataset import get_dataset
from model.util import setup_models
import toml


if __name__ == "__main__":
    config = toml.load('./config.toml')
    dataset, loader = get_dataset(config)
    new, save, load = setup_models(
        config['CRITIC_PARAMS'], 
        config['VAE_PARAMS'],
        config['CRITIC_OPT_PARAMS'],
        config['ENC_OPT_PARAMS'],
        config['DEC_OPT_PARAMS'],
        cuda=config['cuda']
    )
    try:
        models = load()
    except Exception as err:
        models = new()
    
    trainer = Trainer(**models, save_fn=save, config=config)
    trainer.train()
