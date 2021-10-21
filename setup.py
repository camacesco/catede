import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setuptools.setup(
    name='kamapack',
    #url="https://github.com/alfaceor/pygor3",
    author="Francesco Camaglia",
    author_email="francesco.camaglia@phys.ens.fr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.0.12',
    description='Python package to standard statbiophys analysis.',
    license="GNU GPLv3",
    python_requires='>=3.5',
    install_requires = [
        "numpy",
        "pandas",
        #"multiprocessing",
        "scipy",
        "mpmath",
    ],
    packages=setuptools.find_packages(),
    data_files = data_files_to_include,
    include_package_data=True,
    entry_points= {
        'console_scripts' : [
            'kamapack-ngram_entropy=kamapack.analysis.ngram_entropy:main',
            'kamapack-folder_stat=kamapack.analysis.folder_blind_stat:main'
        ],
    }
)
