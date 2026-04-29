from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Chrome Options Setup
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36" 
chrome_options.add_argument(f"user-agent={user_agent}")

# Note: Update chromedriver path for local execution
chromedriver_path = r"D:\python\chromedriver-win64\chromedriver.exe" 
driver = webdriver.Chrome(options=chrome_options)

url = "https://store.steampowered.com/app/1172470/EA_SPORTS_APEX_LEGENDS/" # Example URL
driver.get(url)

initial_wait_time = 30
WebDriverWait(driver, initial_wait_time).until(EC.presence_of_element_located((By.CLASS_NAME, "apphub_CardTextContent")))

# Scrolling and Scraping
scroll_pause_time = 5
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

reviews = driver.find_elements(By.CLASS_NAME, "apphub_Card")
reviews_list = []

for review in reviews:
    try:
        user_id = review.find_element(By.CLASS_NAME, "apphub_CardContentAuthorName").text
    except:
        user_id = "N/A"
    try:
        comment_text = review.find_element(By.CLASS_NAME, "apphub_CardTextContent").text
    except:
        comment_text = "N/A"
    try:
        playtime = review.find_element(By.CSS_SELECTOR, ".hours").text
    except:
        playtime = "N/A"
    try:
        recommendation = review.find_element(By.CLASS_NAME, "title").text
    except:
        recommendation = "N/A"
        
    reviews_list.append({
        "User_ID": user_id, 
        "Review_Text": comment_text, 
        "Playtime": playtime, 
        "Recommendation": recommendation
    })

file_path = "raw_scraped_reviews.csv"
df = pd.DataFrame(reviews_list)
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print("Reviews successfully saved to CSV:", file_path)
driver.quit()
