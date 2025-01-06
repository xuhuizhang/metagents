from setuptools import setup, find_packages

setup(
    name='riskchat',
    version='0.2',
    packages=find_packages(),
    url='http://example.com',
    license='MIT',
    author='Hugo Zhang',
    author_email='hugo.zhangtj@homecreditcfc.cn',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
)