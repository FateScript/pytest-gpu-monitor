from setuptools import setup, find_packages


setup(
    name='pytest-gpu-monitor',
    version='1.0.0',
    description='A pytest plugin to monitor GPU memory usage during tests',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Feng Wang',
    author_email='wffatescript@gmail.com',
    url='https://github.com/FateScript/pytest-gpu-monitor',
    packages=find_packages(),
    install_requires=[
        'pytest>=6.0.0',
        'torch>=1.8.0',
    ],
    entry_points={
        'pytest11': [
            'gpu-monitor = pytest_gpu_monitor.plugin',
        ],
    },
    classifiers=[
        'Framework :: Pytest',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Testing',
    ],
    python_requires='>=3.7',
)
