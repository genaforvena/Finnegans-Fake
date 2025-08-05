import pkgutil
import importlib

# A registry for all available scrapers.
SCRAPER_REGISTRY = {}

def register_scraper(name):
    """A decorator to register a new scraper class."""
    def decorator(cls):
        SCRAPER_REGISTRY[name] = cls
        return cls
    return decorator

def import_scrapers():
    """Dynamically imports all scraper modules in this package."""
    # This ensures that the @register_scraper decorator is called.
    package = importlib.import_module(__name__)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{__name__}.{module_name}")

# Import all scrapers when the package is loaded.
import_scrapers()
