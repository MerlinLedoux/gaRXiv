from garxiv.chunking.cleaner import clean_document


def _line(text: str, page: int) -> dict:
    return {"text": text, "page": page}


def test_removes_watermark_and_page_number_lines():
    parsed = {
        "page_count": 1,
        "sections": [
            {
                "title": "Intro",
                "lines": [
                    _line("arXiv:2401.00001v1 [cs.CL] 1 Jan 2024", 0),
                    _line("3", 0),
                    _line("Real content here.", 0),
                ],
            }
        ],
    }

    cleaned = clean_document(parsed)

    texts = [line["text"] for line in cleaned["sections"][0]["lines"]]
    assert texts == ["Real content here."]
    assert cleaned["sections"][0]["text"] == "Real content here."


def test_removes_recurring_header_footer_lines():
    header = "Running Title of the Paper"
    parsed = {
        "page_count": 5,
        "sections": [
            {
                "title": "Body",
                "lines": [_line(header, p) for p in range(5)]
                + [_line("Unique sentence one.", 0), _line("Unique sentence two.", 1)],
            }
        ],
    }

    cleaned = clean_document(parsed, header_footer_min_frequency=0.4)

    texts = [line["text"] for line in cleaned["sections"][0]["lines"]]
    assert header not in texts
    assert "Unique sentence one." in texts
    assert "Unique sentence two." in texts


def test_merges_hyphenated_word_split_across_lines():
    parsed = {
        "page_count": 1,
        "sections": [
            {
                "title": "Body",
                "lines": [_line("This is a hyphen-", 0), _line("ated word.", 0)],
            }
        ],
    }

    cleaned = clean_document(parsed)

    assert cleaned["sections"][0]["text"] == "This is a hyphenated word."
