from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='philharmonia_dataset',
    version='0.0.1',
    description='PyTorch dataset for philharmonia dataset',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Hugo Flores Garcia',
    author_email='hugofloresgarcia@u.northwestern.edu',
    keywords=['Audio', 'Dataset', 'PyTorch'],
    url='https://github.com/hugofloresgarcia/philharmonia-dataset',
    # download_url='https://pypi.org/project/philharmonia-dataset/'
)

install_requires = [
    'pandas', 
    'torch', 
    'torchaudio', 
    'numpy'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)