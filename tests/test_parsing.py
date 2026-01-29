import pytest

from src.parsing import parse_filename


def test_parse_valid():
    v, c, s = parse_filename("1_10X3001.jpg")
    assert v == 1
    assert c == 3
    assert s == 1


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_filename("bad_name.jpg")
