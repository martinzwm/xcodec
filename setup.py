from setuptools import find_packages
from setuptools import setup

setup(
    name="xcodec",
    version="1.0.0",
    url="https://github.com/boson-ai/xcodec",
    packages=find_packages(include=["xcodec", "xcodec.*"] + find_packages()),
    install_requires=[
        "torch",
        "omegaconf",
        "torchaudio",
        "einops",
        "numpy",
        "transformers",
        "tqdm",
        "tensorboard",
        "descript-audiotools>=0.7.2",
        "descript-audio-codec",
        "scipy==1.10.1"
    ],
)