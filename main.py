import toml
import click
from data.download import download_dataset
from data.dataset import get_dataset
from pprint import pprint
import matplotlib.pyplot as plt


def load_config():
    return toml.load('./config.toml')


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    ctx.obj['CONFIG'] = load_config()


@cli.command()
@click.pass_context
def cfg(ctx):
    pprint(ctx.obj['CONFIG'], indent=4)


@cli.command()
@click.pass_context
def check_dataset(ctx):
    _, dataset = get_dataset(ctx.obj['CONFIG'])
    img = dataset[0].permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


@cli.command()
@click.pass_context
def dl(ctx):
    download_dataset()


# @cli.command()
# def train(ctx):
#     config = ctx.obj['CONFIG']
#     dataset, loader = get_dataset(config)
#     new, save, load = setup_models(
#         config['CRITIC_PARAMS'],
#         config['VAE_PARAMS'],
#         config['CRITIC_OPT_PARAMS'],
#         config['ENC_OPT_PARAMS'],
#         config['DEC_OPT_PARAMS'],
#         cuda=config['cuda']
#     )
#     try:
#         models = load()
#     except Exception:
#         models = new()

#     trainer = Trainer(**models, save_fn=save, config=config)
#     trainer.train()


if __name__ == "__main__":
    cli()