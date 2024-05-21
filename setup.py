from setuptools import setup, find_packages

setup(
    name='predictors',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "git+https://github.com/unslothai/unsloth.git",
        "modal>=0.62",
        "bitsandbytes==0.43.1",
        "xformers>=0.0.25",
        "trl==0.8.6",
        "datasets==2.19.1",
        "instructor==1.2.4",
        "Jinja2==3.1.3"
    ],
    description='Finetune LLMs, locally or remotely, for text classification.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/baptistecumin/predictors', 
    author='Baptiste Cumin',
    author_email='ba.cumin@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
