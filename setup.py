import setuptools

with open('Readme.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="arpys",
    version="0.0.1",
    author="Kyle Gordon",
    author_email="kgord831@gmail.com",
    description="ARPES analysis with python and xarray",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
