import httpx


class TrismikResponseMocker:

    @staticmethod
    def auth() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json={"token": "token", "expires": "2024-08-28T14:18:10.0924006"},
        )

    @staticmethod
    def error(status: int) -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=status,
            json={
                "timestamp": "timestamp",
                "path": "path",
                "status": status,
                "error": "error",
                "requestId": "request_id",
                "message": "message",
            },
        )

    @staticmethod
    def tests() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json=[
                {"id": "fluency", "name": "Fluency"},
                {"id": "hallucination", "name": "Hallucination"},
                {"id": "layman-medical", "name": "Layman Medical"},
                {"id": "memorization", "name": "Memorization"},
                {"id": "toxicity", "name": "Toxicity"},
            ],
        )

    @staticmethod
    def session_start() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=201,
            json={
                "sessionInfo": {"id": "session_id"},
                "state": {
                    "responses": ["item_1"],
                    "thetas": [1.0],
                    "std_error_history": [0.5],
                    "kl_info_history": [0.1],
                    "effective_difficulties": [0.2],
                },
                "nextItem": {
                    "id": "item_1",
                    "question": "question 1",
                    "choices": [{"id": "choice_1", "value": "value 1"}],
                },
                "completed": False,
            },
        )

    @staticmethod
    def session_continue() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json={
                "sessionInfo": {"id": "session_id"},
                "state": {
                    "responses": ["item_1", "item_2"],
                    "thetas": [1.0, 1.2],
                    "std_error_history": [0.5, 0.4],
                    "kl_info_history": [0.1, 0.12],
                    "effective_difficulties": [0.2, 0.25],
                },
                "nextItem": {
                    "id": "item_2",
                    "question": "question 2",
                    "choices": [{"id": "choice_2", "value": "value 2"}],
                },
                "completed": False,
            },
        )

    @staticmethod
    def session_end() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json={
                "sessionInfo": {"id": "session_id"},
                "state": {
                    "responses": ["item_1", "item_2", "item_3"],
                    "thetas": [1.0, 1.2, 1.3],
                    "std_error_history": [0.5, 0.4, 0.3],
                    "kl_info_history": [0.1, 0.12, 0.13],
                    "effective_difficulties": [0.2, 0.25, 0.3],
                },
                "nextItem": None,
                "completed": True,
            },
        )

    @staticmethod
    def results() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json=[
                {
                    "trait": "trait",
                    "name": "name",
                    "value": "value",
                },
            ],
        )

    @staticmethod
    def responses() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json=[
                {
                    "itemId": "item_id",
                    "value": "value",
                    "score": 1.0,
                }
            ],
        )

    @staticmethod
    def no_content() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"), status_code=204, json=None
        )
