import httpx


class TrismikResponseMocker:

    @staticmethod
    def auth_response() -> httpx.Response:
        return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=200,
                json={
                    "token": "token",
                    "expires": "2024-08-28T14:18:10.0924006"
                }
        )

    @staticmethod
    def error_response(status: int) -> httpx.Response:
        return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=status,
                json={
                    "timestamp": "timestamp",
                    "path": "path",
                    "status": status,
                    "error": "error",
                    "requestId": "request_id",
                    "message": "message"
                }
        )

    @staticmethod
    def tests_response() -> httpx.Response:
        return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=200,
                json=[
                    {
                        "id": "fluency",
                        "name": "Fluency"
                    },
                    {
                        "id": "hallucination",
                        "name": "Hallucination"
                    },
                    {
                        "id": "layman-medical",
                        "name": "Layman Medical"
                    },
                    {
                        "id": "memorization",
                        "name": "Memorization"
                    },
                    {
                        "id": "toxicity",
                        "name": "Toxicity"
                    }
                ]
        )

    @staticmethod
    def session_response() -> httpx.Response:
        return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=201,
                json={
                    "id": "id",
                    "url": "url",
                    "status": "status"
                }
        )

    @staticmethod
    def item_response() -> httpx.Response:
        return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=200,
                json={
                    "type": "multiple_choice_text",
                    "question": "question",
                    "choices": [
                        {
                            "id": "choice_id_1",
                            "text": "choice_text_1"
                        },
                        {
                            "id": "choice_id_2",
                            "text": "choice_text_2"
                        },
                        {
                            "id": "choice_id_3",
                            "text": "choice_text_3"
                        }
                    ]
                }
        )

    @staticmethod
    def no_content_response() -> httpx.Response:
        return httpx.Response(
                request=httpx.Request("method", "url"),
                status_code=204,
                json=None
        )
