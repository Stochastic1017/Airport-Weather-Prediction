
import math
import re
import os
import subprocess
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime
import time  # Added for retry delays

def extract_pagination_text(driver):
    """
    Extracts the pagination information text from the page using Selenium.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.

    Returns:
    str: The pagination text indicating the number of entries and pages.
    """
    logging.info("Extracting pagination information.")
    pagination_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.dataTables_info"))
    )
    pagination_text = pagination_element.text
    logging.info(f"Pagination text: {pagination_text}")
    return pagination_text

def compute_pages_to_scrape(text):
    """
    Computes the total number of pages to scrape from the pagination information.

    Parameters:
    text (str): The pagination information text from the webpage.

    Returns:
    int: The total number of pages to scrape.
    """
    logging.info("Computing total pages to scrape from pagination text.")
    total_entries = int(text.split('of')[-1].split('entries')[0].strip())
    entries_per_page = int(text.split('to')[1].split('of')[0].strip())
    total_pages = math.ceil(total_entries / entries_per_page)
    logging.info(f"Total entries: {total_entries}, Entries per page: {entries_per_page}, Total pages: {total_pages}")
    return total_pages

def click_pagination_link(driver, page_number):
    """
    Clicks on the pagination link for the given page number.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.
    page_number (int): The page number to click.
    """
    logging.info(f"Clicking on page {page_number}.")
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.LINK_TEXT, str(page_number)))
    ).click()
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
    )
    logging.info(f"Successfully clicked page {page_number}.")

def get_all_download_links(driver, master_url, retry_limit=3):
    """
    Returns a list of all .tar.gz file links (from all pages), with retry logic.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.
    master_url (str): The base URL of the page containing .tar.gz files.
    retry_limit (int): Number of retries if the page fails to load correctly.

    Returns:
    list: A list of all download links found on the pages.
    """
    logging.info(f"Navigating to base URL: {master_url}")
    driver.get(master_url)

    retries = 0
    while retries < retry_limit:
        try:
            logging.info("Extracting pagination text to calculate number of pages.")
            pagination_text = extract_pagination_text(driver)
            pages_to_collect = compute_pages_to_scrape(pagination_text)

            if pages_to_collect == 0:
                raise ValueError("Total entries returned as 0, retrying...")

            regex = r"lcd_v2.0.0_d.*.tar.gz$"
            download_links = []

            logging.info(f"Total pages to process: {pages_to_collect}")
            for page in range(1, pages_to_collect + 1):
                logging.info(f"Processing master page {page}.")
                click_pagination_link(driver, page)
                WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))

                hyperlinks = driver.find_elements(By.TAG_NAME, "a")
                logging.info(f"Found {len(hyperlinks)} hyperlinks on page {page}.")
                
                for link in hyperlinks:
                    href = link.get_attribute('href')
                    if href and re.search(regex, href):
                        download_links.append(href)
                        logging.info(f"Collected link: {href}")

            logging.info(f"Finished collecting download links from all pages. Total links found: {len(download_links)}")
            return download_links

        except Exception as e:
            retries += 1
            logging.error(f"Attempt {retries}/{retry_limit} failed: {str(e)}")
            time.sleep(2)  # Backoff before retrying

    logging.error("Failed to collect download links after multiple retries.")
    return []

def setup_webdriver():
    """
    Sets up and returns a WebDriver instance using cached geckodriver.

    Returns:
    webdriver: The Selenium WebDriver instance for Firefox in headless mode.
    """
    logging.info("Setting up the WebDriver.")
    geckodriver_path = "/home/stochastic1017/.wdm/drivers/geckodriver/linux64/v0.35.0/geckodriver"
    service = FirefoxService(executable_path=geckodriver_path)
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Firefox(service=service, options=options)
    logging.info("WebDriver successfully set up.")
    return driver

def setup_logging():
    """
    Sets up basic logging configuration.
    """
    log_base_dir = 'logs'
    os.makedirs(log_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=os.path.join(log_base_dir, f'scrape_{timestamp}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Logging setup complete.")

def try_download_url(url, output_path, retry_limit=5):
    """
    Attempts to download a file from a specific URL, with retry logic.

    Parameters:
    url (str): The URL of the file to download.
    output_path (str): The path where the file will be saved.
    retry_limit (int): Number of retries if the download fails.

    Returns:
    bool: True if the download was successful, False otherwise.
    """
    retries = 0
    while retries < retry_limit:
        try:
            logging.info(f"Attempting to download {url}.")
            subprocess.run(['wget', '-O', output_path, url], check=True, capture_output=True, text=True)
            logging.info(f"Successfully downloaded {url} to {output_path}.")
            return True
        except subprocess.CalledProcessError as e:
            retries += 1
            logging.error(f"Failed to download {url}. Attempt {retries}/{retry_limit}: {str(e)}")
            if retries < retry_limit:
                logging.info(f"Retrying download for {url}...")
                time.sleep(2)  # Exponential backoff can be added here if desired
            else:
                logging.error(f"Failed to download {url} after {retry_limit} attempts.")
                return False

def download_link_wrapper(args):
    """
    Wrapper function to handle the download of a specific URL.

    Parameters:
    args (tuple): A tuple containing the URL (str) and output folder (str).

    Returns:
    bool: True if the download was successful, False otherwise.
    """
    url, output_folder = args
    file_name = url.split('/')[-1]
    output_path = os.path.join(output_folder, file_name)
    return try_download_url(url, output_path)

def download_tar_gz_files_parallel(download_links, output_folder, num_workers=8):
    """
    Parallelizes the downloading of .tar.gz files.

    Parameters:
    download_links (list): List of URLs of .tar.gz files to download.
    output_folder (str): The folder where the downloaded files will be saved.
    num_workers (int): The number of parallel workers to use for downloading.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    logging.info(f"Starting download of {len(download_links)} .tar.gz files with {num_workers} workers.")
    download_args = [(link, output_folder) for link in download_links]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(download_link_wrapper, download_args),
            total=len(download_links),
            desc="Downloading .tar.gz files"
        ))
    
    failed_links = [download_links[i] for i, success in enumerate(results) if not success]
    if failed_links:
        logging.error(f"Failed to download {len(failed_links)} files.")
        logging.error(f"Failed links: {failed_links}")
    else:
        logging.info("All files downloaded successfully.")

if __name__ == '__main__':
    # Set up logging
    setup_logging()
    
    # Define the base URL for scraping
    master_url = "https://www.ncei.noaa.gov/oa/local-climatological-data/index.html#v2/archive/"
    
    # Set the output folder for downloaded files
    output_folder = 'tar_gz_downloads'
    
    # Define the number of parallel workers
    num_workers = 8

    # Set up Selenium WebDriver
    driver = setup_webdriver()

    try:
        logging.info("Starting to scrape download links.")
        download_links = get_all_download_links(driver, master_url)
    finally:
        driver.quit()

    logging.info(f"Total download links found: {len(download_links)}")

    # Start the parallel download process
    download_tar_gz_files_parallel(download_links, output_folder, num_workers=num_workers)
    
    logging.info("Download process completed.")
