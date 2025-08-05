import abc
import time
import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from job_search_tool.config import load_config

class BaseScraper(abc.ABC):
    """
    Abstract base class for all scrapers.
    """

    def __init__(self):
        config = load_config()
        self.user_agent = config.get("scraper", {}).get("user_agent", "job-search-tool")

    @abc.abstractmethod
    def scrape(self, session: Session):
        """
        The main method to perform scraping for a platform.
        This method should be implemented by all concrete scrapers.

        Args:
            session: The database session to use for storing scraped data.
        """
        raise NotImplementedError

    def get_page_content(self, url: str) -> BeautifulSoup:
        """
        Fetches the content of a webpage and returns a BeautifulSoup object.

        Args:
            url: The URL of the webpage to fetch.

        Returns:
            A BeautifulSoup object representing the parsed HTML of the page.
        """
        headers = {"User-Agent": self.user_agent}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes

            # A simple rate-limiting mechanism
            time.sleep(1) # wait 1 second between requests

            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
