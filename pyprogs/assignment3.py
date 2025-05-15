from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
    
driver = webdriver.Chrome()
    
driver.get("https://www.instagram.com/")
    
sleep(2)
    
username_field = WebDriverWait(driver, 10).until(
  EC.presence_of_element_located((By.NAME, "username"))
)
password_field = WebDriverWait(driver, 10).until(
  EC.presence_of_element_located((By.NAME, "password"))
)
username_field.send_keys("_abc_23__")
password_field.send_keys("abcde12345")
password_field.send_keys(Keys.RETURN)
sleep(5)
    
driver.quit()