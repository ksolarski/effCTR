from setuptools import setup, find_packages
import os.path

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_requirements(path):
    with open(os.path.join(_CURRENT_DIR, path)) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith(("#", "-i")))
        ]


_INSTALL_REQUIREMENTS = _parse_requirements(
    os.path.join(_CURRENT_DIR, "requirements.txt")
)


setup(
    install_requires=_INSTALL_REQUIREMENTS,
    name="effCTR",
    version="0.1.0",
    url="https://github.com/ksolarski/effCTR",
    author="Kacper Solarski",
    author_email="pchla10@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
)
