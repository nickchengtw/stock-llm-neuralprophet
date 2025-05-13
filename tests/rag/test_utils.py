from datetime import date
from src.rag.utils import generate_date_range  # adjust import path as needed

def test_generate_date_range_normal():
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)
    expected = [
        date(2023, 1, 1),
        date(2023, 1, 2),
        date(2023, 1, 3),
        date(2023, 1, 4),
        date(2023, 1, 5),
    ]
    assert list(generate_date_range(start, end)) == expected

def test_generate_date_range_same_day():
    start = end = date(2024, 5, 13)
    expected = [date(2024, 5, 13)]
    assert list(generate_date_range(start, end)) == expected

def test_generate_date_range_reverse():
    start = date(2024, 5, 15)
    end = date(2024, 5, 13)
    expected = []
    assert list(generate_date_range(start, end)) == expected
