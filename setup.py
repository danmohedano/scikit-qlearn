from setuptools import setup, find_packages
import pathlib
here = pathlib.Path(__file__).parent.resolve()

DISTNAME = 'scikit-qlearn'
DESCRIPTION = 'A set of python modules for quantum enhanced machine learning algorithms'
MAINTAINER = 'Daniel Mohedano'
MAINTAINER_EMAIL = 'danimohedano1998@gmail.com'
URL = 'https://github.com/danmohedano/scikit-qlearn'
LICENSE = 'MIT'
DOWNLOAD_URL = URL
PACKAGE_NAME = 'skqlearn'
LONG_DESCRIPTION = (here / 'README.md').read_text(encoding='utf-8')


def get_version():
    """Obtain the version number"""
    data = {}
    with open("skqlearn/version.py") as fp:
        exec(fp.read(), data)

    return data['__version__']

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


if __name__ == '__main__':
    setup(
        name=DISTNAME,  # Required
        version=get_version(),  # Required
        description=DESCRIPTION,  # Optional
        long_description=LONG_DESCRIPTION,  # Optional
        long_description_content_type='text/markdown',  # Optional (see note above)
        url=URL,  # Optional
        author=MAINTAINER,  # Optional
        author_email=MAINTAINER_EMAIL,  # Optional
        classifiers=[  # Optional
            'Development Status :: 1 - Planning',
            'Intended Audience :: Developers',
            'Intended Audience:: Science / Research',
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
        package_dir={'': PACKAGE_NAME},  # Optional
        packages=find_packages(where=PACKAGE_NAME),  # Required
        python_requires='>=3.7, <4',
        install_requires=[],  # Optional
        extras_require={  # Optional
            'dev': [],
            'test': [],
        },
    )
