from pathlib import Path
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from job_search_tool.database.models import Job
from job_search_tool.scrapers import register_scraper
from job_search_tool.scrapers.base import BaseScraper

@register_scraper("example")
class ExampleScraper(BaseScraper):
    """
    A scraper for the mock_jobs.html file.
    """

    def scrape(self, session: Session):
        """
        Parses the mock_jobs.html file and stores the job listings in the database.
        """
        print("Scraping example jobs from mock_jobs.html...")

        mock_file = Path("mock_jobs.html")
        if not mock_file.is_file():
            print(f"Error: {mock_file} not found.")
            return

        with open(mock_file, "r") as f:
            soup = BeautifulSoup(f, "html.parser")

        job_listings = soup.find_all("div", class_="job-listing")
        seen_urls = set()

        for job in job_listings:
            title_tag = job.find("h2")
            link_tag = job.find("a")
            description_tag = job.find("p")

            if not all([title_tag, link_tag, description_tag]):
                continue

            title = title_tag.text.strip()
            url = link_tag.get("href")
            description = description_tag.text.strip()

            if url in seen_urls:
                print(f"URL '{url}' already processed in this session. Skipping.")
                continue

            seen_urls.add(url)

            # Check if the job already exists in the database
            existing_job = session.query(Job).filter_by(url=url).first()
            if existing_job:
                print(f"Job with URL '{url}' already in DB. Skipping.")
                continue

            new_job = Job(
                title=title,
                url=url,
                description=description,
            )
            session.add(new_job)
            print(f"Found new job: '{title}'")

        print("Example scraping complete.")
