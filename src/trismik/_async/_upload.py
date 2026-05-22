"""Helper for uploading payloads to a Trismik presigned S3 target."""

from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import httpx

from trismik.exceptions import TrismikApiError, TrismikValidationError
from trismik.types import TrismikUserDatasetPresignedPost

UploadSource = Union[str, Path, bytes, BinaryIO]


_EXTENSION_FILE_TYPE = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".parquet": "parquet",
}

_FILE_TYPE_CONTENT_TYPE = {
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "parquet": "application/x-parquet",
}


def _default_content_type(file_type: str) -> str:
    return _FILE_TYPE_CONTENT_TYPE.get(file_type, "application/octet-stream")


def _read_source_to_bytes(source: UploadSource) -> Tuple[bytes, Optional[str]]:
    """Read ``source`` fully into memory; also infer file_type when possible.

    Returns a tuple ``(data, file_type_or_None)``.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        with open(path, "rb") as f:
            return f.read(), _EXTENSION_FILE_TYPE.get(suffix)
    if isinstance(source, bytes):
        return source, None
    data = source.read()
    if not isinstance(data, bytes):
        raise TrismikValidationError("Binary file-like source must return bytes from read().")
    return data, None


async def upload_to_presigned(
    presigned: TrismikUserDatasetPresignedPost,
    data: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> None:
    """Upload ``data`` to an S3 presigned URL.

    Uses a separate ``httpx.AsyncClient`` with no Trismik authentication header
    (S3 rejects unexpected signatures).

    Args:
        presigned: Presigned target returned by the Trismik API.
        data: Bytes to upload.
        content_type: MIME type used for ``PUT`` uploads. Ignored for ``POST``
            uploads (the multipart form drives Content-Type itself).

    Raises:
        TrismikApiError: If S3 returns a non-2xx status.
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        if presigned.method.upper() == "PUT":
            response = await client.put(
                presigned.url,
                content=data,
                headers={"Content-Type": content_type},
            )
        else:
            files = {"file": ("payload", data, content_type)}
            response = await client.post(
                presigned.url,
                data=presigned.fields,
                files=files,
            )
    if response.status_code >= 400:
        raise TrismikApiError(
            f"S3 upload failed with status {response.status_code}: " f"{response.text[:500]}"
        )
