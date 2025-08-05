import time

def get_summary_and_category(description: str) -> tuple[str, str]:
    """
    Mocks a call to an LLM to get a summary and category for a job description.

    In a real implementation, this function would use a library like `openai`
    to send the description to a large language model and parse the response.

    Args:
        description: The job description text.

    Returns:
        A tuple containing the summary and the category.
    """
    print("  (Mock LLM) Analyzing job description...")

    # Simulate network latency of an API call
    time.sleep(0.5)

    # Mocked response
    summary = "This is a mocked summary of the job description."
    category = "Mocked Category"

    # A tiny bit of logic to make the mock response seem more real
    if "python" in description.lower():
        category = "Python Development"
    elif "react" in description.lower():
        category = "Frontend Development"

    print("  (Mock LLM) Analysis complete.")
    return summary, category
