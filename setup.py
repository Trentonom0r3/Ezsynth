from setuptools import setup, find_packages

setup(
    name='ezsynth',
    version='1.2.1.1',
    packages=find_packages(),
    include_package_data=True,
install_requires=[
    'torch',
    'Pillow', 
    'numpy',
    'phycv',  
    'opencv-contrib-python',  
    'scipy'
],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
