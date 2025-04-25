import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

driver.get("https://www.instagram.com/")
input("수동 로그인 완료 후 Enter를 누르세요: ")

pickle.dump(driver.get_cookies(), open("insta_cookies.pkl", "wb"))
driver.quit()
print("✅ 쿠키 저장 완료!")
