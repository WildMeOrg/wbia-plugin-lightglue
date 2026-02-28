from setuptools import setup, find_packages

setup(
    name='wbia_lightglue',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'kornia>=0.6.11',
        'numpy',
        'opencv-python-headless',
        'tqdm',
    ],
)
