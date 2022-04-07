"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='eccv_caption',
    version='0.1.0',
    author='Sanghyuk Chun',
    author_email='sanghyuk.chun@gmail.com',
    description='A PyThon wrapper for Extended COCO Validation (ECCV) Caption dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/naver-ai/eccv-caption',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    zip_safe=False,
)
