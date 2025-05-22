from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Firefox()
driver.maximize_window()
driver.get("https://www.instagram.com/")
wait = WebDriverWait(driver, 15)
# Click on Allow all cookies button
wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Allow all cookies']"))).click()

# Send text to username textbox
wait.until(EC.element_to_be_clickable((By.NAME, "username"))).send_keys("mic test 123")

# Send text to password textbox
wait.until(EC.element_to_be_clickable((By.NAME, "password"))).send_keys("mic test 456")