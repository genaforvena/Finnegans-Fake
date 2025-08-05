# Job Search Automation Tool

This project is a platform-agnostic, open-source tool that automates the discovery and pre-screening of freelance jobs with minimal human intervention.

## Features

- **Job Aggregation:** Scrapes job listings from various platforms.
- **Pre-screening & Filtering:** Filters jobs based on user-defined keywords and a blacklist.
- **Summarization & Categorization:** Uses a (mocked) LLM to summarize and categorize jobs.
- **Data Storage:** Stores all job data in a local SQLite database.
- **CLI:** Provides a command-line interface to interact with the tool.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the project in editable mode:**
    This will install the necessary dependencies and the `jobtool` command-line tool.
    ```bash
    pip install -e .
    ```

## Configuration

1.  Before running the application, you need to create a `config.yaml` file. You can do this by copying the template:
    ```bash
    cp config.yaml.template config.yaml
    ```

2.  Edit `config.yaml` to set your desired keywords, blacklist, and other settings.

## Usage

The tool is used via the `jobtool` command-line interface.

### 1. Scrape for Jobs

To scrape for jobs from a platform, use the `scrape run` command. Currently, only a mock scraper is available.

```bash
jobtool scrape run example
```
This will parse the `mock_jobs.html` file and save the jobs to the database.

### 2. Filter Jobs

After scraping, you can filter the new jobs based on your configuration:

```bash
jobtool filter
```
This will update the status of the jobs in the database to `relevant` or `irrelevant`.

### 3. Summarize Jobs

Finally, you can summarize the relevant jobs:

```bash
jobtool summarize
```
This will use the (mock) LLM to generate a summary and category for each relevant job and update its status to `summarized`.
