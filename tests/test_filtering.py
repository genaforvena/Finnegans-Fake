import pytest
from job_search_tool.filtering import is_job_relevant

@pytest.mark.parametrize("job_text, keywords, blacklist, expected", [
    # Basic keyword matching
    ("I am a python developer", ["python"], [], True),
    ("I am a java developer", ["python"], [], False),

    # Case-insensitivity
    ("I am a Python Developer", ["python"], [], True),
    ("I am a PYTHON DEVELOPER", ["python"], [], True),
    ("I am a developer", ["developer"], [], True),

    # Blacklist takes precedence
    ("I am a python and java developer", ["python"], ["java"], False),
    ("I am a python developer", ["python"], ["python"], False), # keyword is also blacklisted

    # Multiple keywords
    ("I use flask and django", ["flask", "fastapi"], [], True),

    # No keywords
    ("I am a developer", [], ["java"], False),

    # Empty blacklist
    ("I am a C++ developer", ["c++"], [], True),
])
def test_is_job_relevant(job_text, keywords, blacklist, expected):
    """
    Tests the is_job_relevant function with various scenarios.
    """
    assert is_job_relevant(job_text, keywords, blacklist) == expected
