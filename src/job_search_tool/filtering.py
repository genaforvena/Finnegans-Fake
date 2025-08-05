def is_job_relevant(job_text: str, keywords: list[str], blacklist: list[str]) -> bool:
    """
    Determines if a job is relevant based on keywords and a blacklist.

    Args:
        job_text: The text of the job to analyze (e.g., title + description).
        keywords: A list of keywords to search for.
        blacklist: A list of terms that, if present, make the job irrelevant.

    Returns:
        True if the job is relevant, False otherwise.
    """
    job_text_lower = job_text.lower()

    # Check for blacklisted terms first
    if any(term.lower() in job_text_lower for term in blacklist):
        return False

    # If no keywords are provided, nothing is relevant
    if not keywords:
        return False

    # Check for keywords
    if any(keyword.lower() in job_text_lower for keyword in keywords):
        return True

    return False
