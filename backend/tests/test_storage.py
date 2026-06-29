import pytest

from biolit.storage import parse_s3_uri


def test_parse_s3_uri():
    assert parse_s3_uri("s3://temporary/processing/42/crop.jpg") == (
        "temporary",
        "processing/42/crop.jpg",
    )


@pytest.mark.parametrize("uri", ["https://example.org/a.jpg", "s3://bucket", "s3:///key"])
def test_parse_s3_uri_rejects_invalid_values(uri):
    with pytest.raises(ValueError):
        parse_s3_uri(uri)
