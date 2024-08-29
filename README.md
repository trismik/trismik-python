Trismik Python SDK
==================

This is the official Python SDK for Trismik. It provides a simple way to interact with the Trismik
API.

Usage
-----

1. ```pip install trismik```
2. Set the following environment variables. Alternatively, put them into `.env` file
   in the root of your project, and load them using `python-dotenv` package:

   ```
   TRISMIK_SERVICE_URL=<service_url>
   TRISMIK_API_KEY=<api_key>
   ```

3. Refer to examples:
   * `examples/example_runner.py` - run test using high-level `TrismikRunner` class
   * `examples/example_runner_async.py` - like above, but with async support
   * `examples/example_client.py` - run test using `TrismikClient` directly
   * `examples/example_client_async.py` - like above, but with async support

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
