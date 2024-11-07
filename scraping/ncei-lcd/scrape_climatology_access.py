
import re
import pickle
import time
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager
from multiprocessing import Manager, Pool, Process
from datetime import datetime

def setup_logging(url):
    """
    Sets up logging configuration for a specific year.
    Returns both info and error loggers.
    """
    year = url.rstrip('/').split('/')[-1]
    log_base_dir = 'logs'
    info_dir = os.path.join(log_base_dir, year, 'info')
    error_dir = os.path.join(log_base_dir, year, 'error')
    
    # Create directories if they don't exist
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up info logger
    info_logger = logging.getLogger(f'info_logger_{year}')
    info_logger.setLevel(logging.INFO)
    info_handler = RotatingFileHandler(
        os.path.join(info_dir, f'info_{timestamp}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(info_formatter)
    info_logger.addHandler(info_handler)
    
    # Set up error logger
    error_logger = logging.getLogger(f'error_logger_{year}')
    error_logger.setLevel(logging.ERROR)
    error_handler = RotatingFileHandler(
        os.path.join(error_dir, f'error_{timestamp}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n')
    error_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_handler)
    
    return info_logger, error_logger

def setup_webdriver():
    """Sets up and returns a WebDriver instance using cached geckodriver."""
    try:
        geckodriver_path = "/home/stochastic1017/.wdm/drivers/geckodriver/linux64/v0.35.0/geckodriver"
        service = FirefoxService(executable_path=geckodriver_path)
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        
        driver = webdriver.Firefox(service=service, options=options)
        return driver
    except Exception as e:
        raise Exception(f"Failed to setup WebDriver: {str(e)}")

def fix_url(url, info_logger):
    """Forcefully fixes the URL by ensuring the correct 'index.html#v2/access/' structure."""
    fixed_url = url.replace('/v2/access/', '/index.html#v2/access/')
    info_logger.info(f"Processing URL: {fixed_url}")
    return fixed_url

def get_number_of_pages_and_entries(driver, info_logger, error_logger):
    """Returns the number of pages and total entries for a particular year."""
    try:
        time.sleep(1)
        pagination_info = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.dataTables_info"))
        ).text
        
        # Replace commas with an empty string to ignore them during integer parsing
        total_entries = int(pagination_info.split('of')[-1].split('entries')[0].strip().replace(',', ''))
        entries_per_page = int(pagination_info.split('to')[1].split('of')[0].strip().replace(',', ''))
        
        total_pages = (total_entries // entries_per_page) + (total_entries % entries_per_page > 0)
        
        info_logger.info(f"Total entries: {total_entries}, Entries per page: {entries_per_page}, Total pages: {total_pages}")
        return total_pages, total_entries
    except Exception as e:
        error_logger.error(f"Error calculating pages and entries: {str(e)}")
        return 1, 0

def collect_csv_links_from_page(driver, url, info_logger, error_logger):
    """Navigates to the given URL and collects all CSV file links."""
    csv_links = []
    csv_regex = r".*\.csv$"
    
    try:
        fixed_url = fix_url(url, info_logger)
        driver.get(fixed_url)
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
        
        hyperlinks = driver.find_elements(By.TAG_NAME, "a")
        for link in hyperlinks:
            csv_href = link.get_attribute('href')
            if csv_href and re.search(csv_regex, csv_href):
                csv_links.append(csv_href)
                
        info_logger.info(f"Found {len(csv_links)} CSV links on page")
        return csv_links
    except Exception as e:
        error_logger.error(f"Error collecting CSV links: {str(e)}")
        return []

def download_csv_files(csv_links, output_folder, driver, url, info_logger, error_logger):
    """Downloads CSV files with retry mechanism and proper error handling."""
    year = url.rstrip('/').split('/')[-1]
    year_output_folder = os.path.join(output_folder, year)
    os.makedirs(year_output_folder, exist_ok=True)
    
    for csv_link in csv_links:
        file_name = csv_link.split('/')[-1]
        output_path = os.path.join(year_output_folder, file_name)
        info_logger.info(f"Downloading: {csv_link} -> {output_path}")
        
        retry_count = 3
        for attempt in range(retry_count):
            try:
                subprocess.run(['wget', '-O', output_path, csv_link], 
                             check=True, 
                             capture_output=True, 
                             text=True)
                info_logger.info(f"Successfully downloaded {file_name} after {attempt + 1} attempt(s)")
                break
            except subprocess.CalledProcessError as e:
                error_logger.error(f"Download attempt {attempt + 1} failed for {file_name}: {str(e)}")
                if attempt < retry_count - 1:
                    info_logger.info(f"Retrying download: {file_name}")
                    driver.get(fix_url(url, info_logger))
                    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
                    time.sleep(2)
                else:
                    error_logger.error(f"Failed to download {file_name} after {retry_count} attempts")

def check_and_log_missing_files(output_folder, year, total_entries, info_logger, error_logger):
    """
    Checks the number of CSV files downloaded for a year and logs missing entries if any.
    """
    year_output_folder = os.path.join(output_folder, year)
    if not os.path.exists(year_output_folder):
        error_logger.error(f"Output folder for {year} does not exist.")
        return
    
    # Count the number of CSV files downloaded
    downloaded_files = [f for f in os.listdir(year_output_folder) if f.endswith('.csv')]
    num_downloaded_files = len(downloaded_files)
    
    info_logger.info(f"Downloaded {num_downloaded_files}/{total_entries} files for {year}.")
    
    # If the count doesn't match, log it as missing
    if num_downloaded_files != total_entries:
        missing_count = total_entries - num_downloaded_files
        missing_folder = os.path.join("missing")
        os.makedirs(missing_folder, exist_ok=True)
        missing_file_path = os.path.join(missing_folder, f'missing_{year}.txt')
        
        with open(missing_file_path, 'w') as f:
            f.write(f"Year: {year}\n")
            f.write(f"Total entries: {total_entries}\n")
            f.write(f"Downloaded: {num_downloaded_files}\n")
            f.write(f"Missing: {missing_count}\n")
        
        error_logger.error(f"{missing_count} files missing for {year}. Details saved to {missing_file_path}")
    else:
        info_logger.info(f"All files for {year} were downloaded successfully!")


def scrape_pages(driver, url, output_folder, info_logger, error_logger):
    """
    Scrapes all pages for a particular year.
    After scraping all pages, checks if all CSV files are downloaded.
    """
    fixed_url = fix_url(url, info_logger)
    retry_count = 3
    year = fixed_url.rstrip('/').split('/')[-1]

    for attempt in range(retry_count):
        try:
            driver.get(fixed_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.dataTables_info"))
            )
            info_logger.info(f"Page loaded successfully after {attempt + 1} attempts")
            
            total_pages, total_entries = get_number_of_pages_and_entries(driver, info_logger, error_logger)
            
            if total_entries == 0:
                info_logger.warning("No entries detected, refreshing page")
                driver.refresh()
                time.sleep(2)
                continue
            break
        except Exception as e:
            error_logger.error(f"Page load attempt {attempt + 1} failed: {str(e)}")
            driver.refresh()
            time.sleep(2)

    if total_entries == 0:
        error_logger.error(f"Failed to load data after {retry_count} attempts")
        return

    for page in range(1, total_pages + 1):
        info_logger.info(f"Processing page {page} of {total_pages}")
        if page > 1:
            try:
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, str(page)))
                ).click()
                time.sleep(1)
            except Exception as e:
                error_logger.error(f"Navigation failed for page {page}: {str(e)}")
                continue

        csv_links = collect_csv_links_from_page(driver, url, info_logger, error_logger)
        download_csv_files(csv_links, output_folder, driver, url, info_logger, error_logger)
        time.sleep(1)
    
    # Check if the number of files matches the total entries
    check_and_log_missing_files(output_folder, year, total_entries, info_logger, error_logger)


def parallel_scrape(driver, page, url, progress, lock, output_folder):
    """Handles parallel scraping with proper logging and missing file check."""
    info_logger, error_logger = setup_logging(url)
    try:
        scrape_pages(driver, url, output_folder, info_logger, error_logger)
        with lock:
            progress[page] += 1
        info_logger.info(f"Completed processing page {page}")
    except Exception as e:
        error_logger.error(f"Parallel scraping failed: {str(e)}")

def initialize_driver_in_process(url, page, progress, lock, output_folder):
    """Initializes WebDriver and processes URL within a process."""
    info_logger, error_logger = setup_logging(url)
    try:
        driver = setup_webdriver()
        parallel_scrape(driver, page, url, progress, lock, output_folder)
    except Exception as e:
        error_logger.error(f"Process initialization failed: {str(e)}")
    finally:
        if 'driver' in locals():
            driver.quit()

def update_progress_bar(progress, total_tasks):
    """Updates progress bar with error handling."""
    try:
        from tqdm import tqdm
        with tqdm(total=total_tasks) as pbar:
            completed = 0
            while completed < total_tasks:
                new_completed = sum(progress.values())
                if new_completed > completed:
                    pbar.update(new_completed - completed)
                    completed = new_completed
                time.sleep(1)
    except Exception as e:
        print(f"Progress bar error: {str(e)}")

if __name__ == '__main__':
    # Load the pickle file containing the URLs
    with open('hrefs_dict.pkl', 'rb') as f:
        page_links = pickle.load(f)

    # Set up base output folder
    output_folder = 'csv_downloads'
    os.makedirs(output_folder, exist_ok=True)

    # Initialize multiprocessing manager
    manager = Manager()
    progress = manager.dict({page: 0 for page in page_links.keys()})
    lock = manager.Lock()

    total_tasks = sum(len(urls) for urls in page_links.values())

    # Start progress bar process
    progress_bar_process = Process(target=update_progress_bar, args=(progress, total_tasks))
    progress_bar_process.start()

    # Create and run process pool
    with Pool(processes=15) as pool:
        tasks = []
        for page, urls in page_links.items():
            for url in urls:
                tasks.append(pool.apply_async(
                    initialize_driver_in_process,
                    (url, page, progress, lock, output_folder)
                ))

        # Wait for completion
        results = [task.get() for task in tasks]

    progress_bar_process.join()