from distutils.core import setup


setup(
    name='duct',
    version='0.1.0',
    description='ML tools for generative model experiments using torch',
    author='Alexander Athorne',
    author_email='alexander.athorne@gmail.com',
    url='https://github.com/mauicv/vaegan/',
    packages=[
        'duct.model',
        'duct.utils',
        'duct.model.latent_spaces',
        'duct.model.transformer',
        'duct.model.samplers',
    ],
    install_requires=[
        "torch==1.13.1",
        "toml==0.10.2",
        "torchaudio==0.13.1",
        "torchvision==0.14.1",
        "einops==0.6.0"
    ],
)