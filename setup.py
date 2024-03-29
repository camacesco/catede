import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setuptools.setup(
    name='catede',
    url="https://github.com/camacesco/catede",
    author="Francesco Camaglia",
    author_email="francesco.camaglia@phys.ens.fr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.1.00',
    description='Python package for entropy and divergence estimation.',
    license="GNU GPLv3",
    python_requires='>=3.5',
    install_requires = [
        "numpy",
        "pandas",
        "scipy",
        "mpmath",
        "tqdm",
        "matplotlib",
    ],
    packages=setuptools.find_packages(),
    data_files = data_files_to_include,
    include_package_data=True,
)
