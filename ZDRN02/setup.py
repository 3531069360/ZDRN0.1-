from setuptools import setup, find_packages

setup(
    name='ZDRN0.1',
    version='0.1',
    description='A high - performance package for jewelry knowledge classification using ZDRN model, designed for low CPU usage and rapid big - data computation',
    long_description='This package, ZDRN0.1, is developed with the core intention of minimizing CPU consumption while enabling rapid computation on large - scale data. It uses an optimized ZDRN model for jewelry knowledge classification tasks, leveraging efficient algorithms and libraries to ensure high - performance processing.',
    author='Your Name',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.22.0',  # 较新版本的 numpy 有更好的计算性能
        'pandas>=1.4.0',
        'scikit - learn>=1.1.0',
        'matplotlib>=3.5.0',
        'torch>=1.11.0',  # 较新版本的 PyTorch 对计算性能有优化
        'jieba>=0.42.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research'
    ],
    python_requires='>=3.9',
)
