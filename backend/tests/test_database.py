from biolit.database import extract_photo_urls, image_rows_from_observations


def test_extract_photo_urls_accepts_api_shapes():
    assert extract_photo_urls("https://example.org/one.jpg") == ["https://example.org/one.jpg"]
    assert extract_photo_urls(
        [{"url": "https://example.org/one.jpg"}, {"src": "https://example.org/two.jpg"}]
    ) == ["https://example.org/one.jpg", "https://example.org/two.jpg"]
    assert extract_photo_urls('[{"large_url": "https://example.org/three.jpg"}]') == [
        "https://example.org/three.jpg"
    ]


def test_image_ids_are_stable_for_one_or_multiple_photos():
    rows = image_rows_from_observations(
        [
            {"id_observation": 42, "photos": "https://example.org/42.jpg"},
            {
                "id_observation": 43,
                "photos": ["https://example.org/a.jpg", "https://example.org/b.jpg"],
            },
            {
                "id_observation": 44,
                "photos": "https://example.org/already-validated.jpg",
                "validee": "true",
            },
        ]
    )

    assert [row["id_image"] for row in rows] == ["42", "43:0", "43:1"]
    assert [row["image_position"] for row in rows] == [0, 0, 1]
