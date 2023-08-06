from setuptools import setup, find_packages

setup(
    name='ezsynth',
    version='1.0.2.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    'opencv-python',
    'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
