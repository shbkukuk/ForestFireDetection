import os
import json
import requests  # to sent GET requests
from bs4 import BeautifulSoup  # to parse HTML

getty_images = \
                 'https://www.gettyimages.com/photos/x?phrase=x&sort=mostpopular#license'
usr_agent = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64 ; X64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

save_folder = "Fire_Dowload"

def main():
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    dowload_img()


def dowload_img():
    data = input("Enter the search key : ")
    n_images = int(input('How many pages do you want? (1-100 pages) :'))
    n_photo = int(input('how many images do you want ? :'))
    #print(data)
    search_url = f"https://www.gettyimages.com/photos/{data}?phrase={data}&sort=best#license"
    print(search_url)
    response = requests.get(search_url, headers=usr_agent)
    print(response)
    html = response.text
    print(len(html))

    soup = BeautifulSoup(html, 'html.parser')
    results = soup.find_all('img',{'class':'gallery-asset__thumb gallery-mosaic-asset__thumb'},limit=1000)
    print(len(results))

    if n_images > 1:
        more_results = []
        more_url = []
        for page in range(1,n_images+1):
            search_url = f"https://www.gettyimages.com/photos/{data}?page={str(page)}&phrase={data}&sort=mostpopular#license"
            more_url.append(search_url)
        for search_url in more_url:
            print(search_url)
            #print(type(search_url))
            response = requests.get(search_url, headers=usr_agent)
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            more_results.extend(soup.findAll('img',{'class':'gallery-asset__thumb gallery-mosaic-asset__thumb'}, limit=1000))
            print(more_results)
    print(len(more_results))
    top_result = results + more_results
    print(len(top_result))

    count = 0
    imagelinks = []
    for res in top_result:
        imagelinks.append(res["src"])
        print((imagelinks))
        #imagelinks.append(link)
        count = count + 1
        if (count >= n_photo):
            break


    print(f'Found {len(imagelinks)} images')

    for i, imagelink in enumerate(imagelinks):
        # open each image link and save the file
        response = requests.get(imagelink)

        imagename = save_folder + '/' + data + str(i + 1) + '.jpg'
        with open(imagename, 'wb') as file:
            file.write(response.content)


    print("done")

if __name__ == '__main__':
    main()
