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
        if status == 413:
            # 413 errors have a different JSON structure with 'detail' field
            return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=status,
                json={
                    "detail": (
                        "metadata size must be less than 10KB (10240 bytes)."
                    )
                },
            )
        else:
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
                "runInfo": {"id": "session_id"},
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
                "runInfo": {"id": "session_id"},
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
                "runInfo": {"id": "session_id"},
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
    def session_summary() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json={
                "id": "session_id",
                "datasetId": "test_id",
                "state": {
                    "responses": ["item_1"],
                    "thetas": [1.0],
                    "std_error_history": [0.5],
                    "kl_info_history": [0.1],
                    "effective_difficulties": [0.2],
                },
                "dataset": [
                    {
                        "id": "item_id",
                        "question": "Test question",
                        "choices": [
                            {"id": "A", "value": "Choice A"},
                            {"id": "B", "value": "Choice B"},
                        ],
                    }
                ],
                "responses": [
                    {
                        "datasetItemId": "item_id",
                        "value": "value",
                        "correct": True,
                    }
                ],
                "metadata": {"foo": "bar"},
            },
        )

    @staticmethod
    def no_content() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"), status_code=204, json=None
        )

    @staticmethod
    def session_replay() -> httpx.Response:
        return httpx.Response(
            request=httpx.Request("method", "url"),
            status_code=200,
            json={
                "id": "replay_session_id",
                "datasetId": "test_id",
                "state": {
                    "responses": ["item_1"],
                    "thetas": [1.0],
                    "std_error_history": [0.5],
                    "kl_info_history": [0.1],
                    "effective_difficulties": [0.2],
                },
                "replayOfRun": "original_session_id",
                "completedAt": "2025-06-26T10:56:03.356Z",
                "createdAt": "2025-06-26T10:56:03.356Z",
                "metadata": {"foo": "bar"},
                "dataset": [
                    {
                        "id": "item_id",
                        "question": "Test question",
                        "choices": [
                            {"id": "A", "value": "Choice A"},
                            {"id": "B", "value": "Choice B"},
                        ],
                    }
                ],
                "responses": [
                    {
                        "datasetItemId": "item_id",
                        "value": "value",
                        "correct": True,
                    }
                ],
            },
        )
