import click
from job_search_tool.database.engine import SessionLocal
from job_search_tool.scrapers import SCRAPER_REGISTRY

@click.group()
def cli():
    """
    A command-line tool for automating the discovery and pre-screening of freelance jobs.
    """
    pass

@cli.group()
def scrape():
    """Commands for scraping job listings."""
    pass

from job_search_tool.config import load_config
from job_search_tool.database.models import Job
from job_search_tool.filtering import is_job_relevant

@cli.command()
def filter():
    """Filter jobs based on keywords and blacklist."""
    click.echo("Filtering jobs...")

    config = load_config()
    keywords = config.get("filtering", {}).get("keywords", [])
    blacklist = config.get("filtering", {}).get("blacklist", [])

    if not keywords:
        click.echo("Warning: No keywords defined in the configuration. All jobs will be marked as irrelevant.")

    session = SessionLocal()
    try:
        jobs_to_filter = session.query(Job).filter(Job.status == 'new').all()
        click.echo(f"Found {len(jobs_to_filter)} new jobs to filter.")

        for job in jobs_to_filter:
            job_text = job.title + " " + job.description
            if is_job_relevant(job_text, keywords, blacklist):
                job.status = 'relevant'
                click.echo(f"  - Job '{job.title}' is relevant.")
            else:
                job.status = 'irrelevant'
                click.echo(f"  - Job '{job.title}' is irrelevant.")

        session.commit()
        click.echo("Filtering complete.")
    except Exception as e:
        session.rollback()
        click.echo(f"An error occurred during filtering: {e}")
    finally:
        session.close()

from job_search_tool.llm import get_summary_and_category

@cli.command()
def summarize():
    """Summarize and categorize jobs using an LLM."""
    click.echo("Summarizing jobs...")

    session = SessionLocal()
    try:
        jobs_to_summarize = session.query(Job).filter(Job.status == 'relevant').all()
        click.echo(f"Found {len(jobs_to_summarize)} relevant jobs to summarize.")

        for job in jobs_to_summarize:
            click.echo(f"Summarizing job: '{job.title}'")
            summary, category = get_summary_and_category(job.description)

            job.summary = summary
            job.category = category
            job.status = 'summarized'

            click.echo(f"  - Summary: {summary}")
            click.echo(f"  - Category: {category}")

        session.commit()
        click.echo("Summarization complete.")
    except Exception as e:
        session.rollback()
        click.echo(f"An error occurred during summarization: {e}")
    finally:
        session.close()

@scrape.command(name="run")
@click.argument('platform')
def run_scraper(platform):
    """Run the scraper for a specific platform."""
    scraper_class = SCRAPER_REGISTRY.get(platform)
    if not scraper_class:
        click.echo(f"Error: No scraper found for platform '{platform}'.")
        available_platforms = ", ".join(SCRAPER_REGISTRY.keys())
        click.echo(f"Available platforms: {available_platforms}")
        return

    session = SessionLocal()
    try:
        scraper = scraper_class()
        scraper.scrape(session)
        session.commit()
        click.echo(f"Scraping for {platform} complete.")
    except Exception as e:
        session.rollback()
        click.echo(f"An error occurred during scraping: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    cli()
