Trismik Python SDK
==================

This is the official Python SDK for Trismik. It provides a simple way to interact with the Trismik API.

Contributing
------------

1. Install [Python Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. ```git clone https://github.com/trismik/trismik-python.git```
3. ```cd ./trismik-python```
4. ```poetry install```
5. ```poetry run pytest```

Publishing to TestPyPi
----------------------
1. ```poetry config repositories.testpypi https://test.pypi.org/legacy/```
2. ```poetry config pypi-token.testpypi <token>```
3. ```poetry publish --build --repository testpypi```
