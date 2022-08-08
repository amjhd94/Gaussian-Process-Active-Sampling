from setuptools import setup
import setuptools

with open("requirements.txt", "r") as req:
    requires = req.read().split("\n")


setup(name="GPsearch",
      version="0.1",
      description="Active learning with output-weighted importance sampling",
      install_requires=requires,
      packages=setuptools.find_packages(),
      include_package_data=True,
      license="MIT"
    )
