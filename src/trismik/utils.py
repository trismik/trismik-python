import os

from .exceptions import TrismikError


class TrismikUtils:

    @staticmethod
    def required_option(
            value: str,
            name: str,
            env: str
    ) -> str:
        if value is None:
            value = os.environ.get(env)
        if value is None:
            raise TrismikError(
                    f"The {name} client option must be set either by passing "
                    f"{env} to the client or by setting the {env} "
                    "environment variable"
            )
        return value
