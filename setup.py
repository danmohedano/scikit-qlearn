from setuptools import setup, find_packages
import pathlib
import codecs
import os.path

here = os.path.abspath(os.path.dirname(__file__))


def read(rel_path):
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


DISTNAME = 'scikit-qlearn'
DESCRIPTION = 'A set of python modules for quantum enhanced machine learning algorithms'
MAINTAINER = 'Daniel Mohedano'
MAINTAINER_EMAIL = 'danimohedano1998@gmail.com'
URL = 'https://github.com/danmohedano/scikit-qlearn'
LICENSE = 'MIT'
DOWNLOAD_URL = URL
PACKAGE_NAME = 'skqlearn'
LONG_DESCRIPTION = read('README.md')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


if __name__ == '__main__':
    setup(
        name=DISTNAME,  # Required
        version=get_version(os.path.join(PACKAGE_NAME, '_version.py')),  # Required
        description=DESCRIPTION,  # Optional
        long_description=LONG_DESCRIPTION,  # Optional
        long_description_content_type='text/markdown',  # Optional (see note above)
        url=URL,  # Optional
        author=MAINTAINER,  # Optional
        author_email=MAINTAINER_EMAIL,  # Optional
        classifiers=[  # Optional
            'Development Status :: 1 - Planning',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            "Programming Language :: Python :: 3.10",
            'Programming Language :: Python :: 3 :: Only',
        ],
        keywords='quantum, machine learning, ai',  # Optional
        packages=[PACKAGE_NAME],  # Required
        python_requires='>=3.7, <4',
        install_requires=['qiskit',
                          'numpy'],  # Optional
    )
