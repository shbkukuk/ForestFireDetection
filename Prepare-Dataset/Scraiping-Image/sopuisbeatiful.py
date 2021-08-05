import requests
from bs4 import BeautifulSoup
import os

search = "forest fire"
save_folder = "Fire"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
url=("https://www.google.com/search?q=forest&safe=off&sxsrf=ALeKk018gS9BoGLDBNURnPpVmeBsqoxkCw:1611024525445&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjJt8CE_qbuAhVS6qQKHWXmAt0Q_AUoAXoECBEQAw&biw=1535&bih=762")
r = requests.get(url)
print(r)

soup = BeautifulSoup(r.text,'html.parser')
print(soup.title.text)

images = soup.find_all('img',limit=1000000)
image_link = []
for image in images :
    link =image['src']
    image_link.append(link)
for i, imglink in enumerate(image_link):
    if imglink == "/images/branding/searchlogo/1x/googlelogo_desk_heirloom_color_150x55dp.gif":
        continue
    response = requests.get(imglink)
    imagename = save_folder + '/' + search + str(i + 1) + '.jpg'
    with open(imagename ,'wb') as f:
        if link == "/images/branding/searchlogo/1x/googlelogo_desk_heirloom_color_150x55dp.gif":
            continue
        img = requests.get(link)
        f.write(response.content)



