from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome('C:/Users/shbku/PycharmProjects/ML/Final-Project/chromedriver.exe')
driver.get('https://www.google.com/')

box = driver.find_element_by_xpath('//*[@id="tsf"]/div[2]/div[1]/div[1]/div/div[2]/input')
box.send_keys('Tree')
box.send_keys(Keys.ENTER)

driver.find_element_by_xpath('//*[@id="hdtb-msb-vis"]/div[2]/a').click()


#Will keep scrolling down the webpage until it cannot scroll no more
last_height = driver.execute_script('return document.body.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(2)
    new_height = driver.execute_script('return document.body.scrollHeight')
    try:
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
        time.sleep(2)
    except:
        pass
    if new_height == last_height:
        break
    last_height = new_height


for i in range(1, 400):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot('C:/Users/shbku/PycharmProjects/ML/Scraping-image/Tree ('+str(i)+').png')
    except:
        pass