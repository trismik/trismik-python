Trismik Python SDK
==================

This is the official Python SDK for Trismik. It provides a simple way to interact with the Trismik
API.

Usage
-----

1. ```pip install trismik```
2. Set the following environment variable. Alternatively, put it into `.env` file
   in the root of your project, and load them using `python-dotenv` package:

   ```
   TRISMIK_API_KEY=<api_key>
   ```

3. Refer to examples:
   * [example_runner.py](./examples/example_runner.py) - run test using high-level `TrismikRunner`
     class
   * [example_runner_async.py](./examples/example_runner_async.py) - like above, but with async
     support
   * [example_client.py](./examples/example_client.py) - run test using `TrismikClient` directly
   * [example_client_async.py](./examples/example_client_async.py) - like above, but with async
     support

Contributing
------------

1. Install [Python Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. ```git clone https://github.com/trismik/trismik-python.git```, or if cloned previously:
   ```git pull``` to update
3. ```cd ./trismik-python```
4. ```poetry install```
5. ```poetry run pytest```

To test the latest source code as a package without publishing to TestPyPi, run `pip install -e .` from the repo root folder for `pip` to use local source files.

Publishing to TestPyPi
----------------------

1. ```poetry config repositories.testpypi https://test.pypi.org/legacy/```
2. ```poetry config pypi-token.testpypi <token>```
3. ```poetry publish --build --repository testpypi```
