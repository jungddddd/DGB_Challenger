import time
import pickle
import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def load_cookies(driver, path="insta_cookies.pkl"):
    with open(path, "rb") as f:
        cookies = pickle.load(f)
        for cookie in cookies:
            driver.add_cookie(cookie)
    driver.refresh()
    time.sleep(5)


def get_post_links_by_hashtag(driver, hashtag, scrolls):
    driver.get("https://www.instagram.com/")
    time.sleep(5)
    load_cookies(driver)
    time.sleep(5)

    url = f"https://www.instagram.com/explore/tags/{hashtag}/"
    driver.get(url)
    time.sleep(5)

    post_links = set()
    prev_height = driver.execute_script("return document.body.scrollHeight")

    for i in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(6)

        for _ in range(5):
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height > prev_height:
                prev_height = new_height
                break
            time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.find_all("a", href=True)

        new_links = 0
        for link in links:
            href = link["href"]
            if "/p/" in href:
                full_url = "https://www.instagram.com" + href
                if full_url not in post_links:
                    post_links.add(full_url)
                    new_links += 1

        total = len(post_links)
        print(f"[{i}] 현재 수집된 게시물 수: {total} (+{new_links})")

        if new_links == 0:
            print(f"[{i}] 더 이상 새로운 게시물이 로딩되지 않습니다. 중단합니다.")
            break

    return list(post_links)


def extract_post_info(driver, url):
    driver.get(url)
    time.sleep(6)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    caption = ""
    caption_tag = soup.select_one("div span > div > span")
    if caption_tag:
        caption = " ".join(caption_tag.stripped_strings).strip()

    hashtags = re.findall(r"#\w+", caption)

    user_tag = soup.select_one("a.x1i10hfl.xjbqb8w")
    user = user_tag.text if user_tag else "알 수 없음"

    return {
        "url": url,
        "user": user,
        "caption": caption,
        "hashtags": hashtags,
    }


def create_driver():
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        """
    })

    return driver


# ------------------------- 실행 시작 -------------------------

hashtag = "부동산"
scroll_count = 20
restart_every = 100

driver = create_driver()

data = {"url": [], "user": [], "caption": [], "hashtags": []}

try:
    post_urls = get_post_links_by_hashtag(driver, hashtag, scrolls=scroll_count)
    print(f"[{hashtag}] 해시태그로 수집된 게시물 수: {len(post_urls)}")

    for i, url in enumerate(post_urls):
        if i > 0 and i % restart_every == 0:
            print(f"[{i}] 드라이버를 재시작합니다.")
            driver.quit()
            time.sleep(3)
            driver = create_driver()

        try:
            post_info = extract_post_info(driver, url)
            data["url"].append(post_info["url"])
            data["user"].append(post_info["user"])
            data["caption"].append(post_info["caption"])
            data["hashtags"].append(post_info["hashtags"])
            print(f"[{i}] {post_info}")
        except Exception as e:
            print(f"[{i}] 게시물 분석 중 오류 발생: {url} / 에러: {e}")

finally:
    driver.quit()

df = pd.DataFrame(data)
df.to_csv(f"{hashtag}.csv", index=False)
print(f"CSV 저장 완료: {hashtag}.csv")
