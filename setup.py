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
        'duct.model.transformer'
    ],
    install_requires=[
        "torch>=1.7.0, <1.14.0",
        "toml>=0.10.1, <1.0.0"
    ],
)