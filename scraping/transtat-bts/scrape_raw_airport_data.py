
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

"""
This script automates the process of downloading raw airport data from the U.S. Bureau of 
Transportation Statistics (BTS) website using Selenium WebDriver. The script performs the following steps:

1. Sets up a headless Chrome WebDriver instance with necessary configurations.
2. Navigates to the BTS TranStats page where airport data can be downloaded.
3. Selects checkboxes for specific data fields and configures download settings.
4. Iterates over available years and months in dropdowns, downloading data for each month in each year.

The downloaded files are saved automatically to the default download directory set in the browser 
preferences, typically your system's "Downloads" folder.

Usage:
    python3 scrape_raw_airport_data.py

Requirements:
- Chrome WebDriver (managed automatically by `webdriver_manager`).
- The `selenium`, `webdriver_manager`, and `time` libraries.

Important Notes:
- Ensure the download location is set correctly in your browser's preferences.
- This script uses headless Chrome mode, which prevents visible UI interactions. Adjust options if 
  a visible UI is desired.
- Each download may take time; adjust the `time.sleep()` interval if needed for slower connections.
"""

def setup_webdriver():
    """Sets up and returns a WebDriver instance using cached geckodriver."""
    try:                                                                                    
        chrome_options = Options()
        chrome_options.add_argument("--headless")               # Run in headless mode                             
        chrome_options.add_argument("--disable-extensions")     # Disable extensions                               
        chrome_options.add_argument("--disable-gpu")            # Disable GPU                                      
        chrome_options.add_argument("--no-sandbox")             # No sandbox                                       
        chrome_options.add_argument("--disable-dev-shm-usage")  # Disable dev-shm usage                            

        # Initialize Chrome WebDriver with the correct options                                                     
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver

    except Exception as e:
        raise Exception(f"Failed to setup WebDriver: {str(e)}")
    
# Set up the WebDriver
driver = setup_webdriver()
driver.get("https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr")

print("\n")
print("clicking checkboxes...")

driver.find_element(By.ID, "chkDownloadZip").click()
driver.find_element(By.ID, "chkshowNull").click()

# Find all the checkboxes on the page and select them
checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
for checkbox in checkboxes:
    if "chk" not in checkbox.get_attribute('name') and not checkbox.is_selected():
        driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)  # Scroll into view
        checkbox.click()

# Scroll back up
driver.execute_script("window.scrollTo(0, 0);")  # Scrolls back to the top

print("\n")
print("Begin Downloading Data Chunks...")

# Find the dropdown element by its ID
dropdown_years = Select(driver.find_element(By.ID, 'cboYear'))
dropdown_months = Select(driver.find_element(By.ID, 'cboPeriod'))

total_start = time.time()
for year in dropdown_years.options:
    start_year = time.time()

    print("\n")
    print(f"Currently at year: {year.text}")
    dropdown_years.select_by_visible_text(year.text)

    for month in dropdown_months.options:
        start_month = time.time()
        print("\n")
        print(f"Currently at Month: {month.text}")
        dropdown_months.select_by_visible_text(month.text)

        try:
            # Click the download button
            driver.find_element(By.ID, "btnDownload").click()

            # Wait for the file to be downloaded and renamed
            print("\n")
            print(f"Downloading Data for: {year.text} <-> {month.text}.")
            print(f"Total time: {time.time() - total_start}")
            print(f"Year time: {time.time() - start_year}")
            print(f"Month time: {time.time() - start_month}")
    
        except Exception as e:
            print("Error:", e)
            print(f"Unable to process: {year.text} <-> {month.text}.")

        time.sleep(5)

driver.quit()
