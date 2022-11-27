import click
from duct.utils.download import download_dataset
from duct.utils.dataset import get_dataset
from pprint import pprint
import matplotlib.pyplot as plt
from duct.utils.config_mixin import load_config


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


if __name__ == "__main__":
    cli()