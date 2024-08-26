from setuptools import setup, find_packages

setup(
    name='pytorch_lightning_quick_start_utils',
    packages=find_packages(),
    version='0.1.1',
    license='MIT',
    description='A utility library for PyTorch Lightning that provides pre-configured training setups to quickly start training.',
    author='HiDolen',
    author_email='820859278@qq.com',
    url='https://github.com/hidolen/pytorch_lightning_quick_start_utils',
    keywords=['artificial intelligence', 'deep learning', 'pytorch', 'lightning'],
    install_requires=[
        'torch>=2.0',
        'pytorch_lightning>=2.0',
        'numpy',
    ],
)
