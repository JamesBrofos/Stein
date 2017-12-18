from setuptools import setup


# Import __version__ from code base.
exec(open("stein/version.py").read())

setup(
    name="stein",
    version=__version__,
    description="A Library for inference with Gaussian processes.",
    author="James Brofos",
    author_email="james@brofos.org",
    url="http://brofos.org",
    keywords="machine learning statistics gaussian process",
    license="MIT License"
)
