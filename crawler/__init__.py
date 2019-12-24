import sys
import os
import random
import urllib.request
import requests as r
from bs4 import BeautifulSoup
import json

class GoogleImagesCrawler:
    def __init__(self):
        self.__found_urls = list()
        self.__crawl_time = 0
        self.__images_to_ignore = list()

    def crawl_for_images(self, url):
        #Open the url for reading
        header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        __request = r.Request(url, headers=header)
        __page = r.get(url, headers=header).content
        #Initiate beautiful soup
        __soup = BeautifulSoup(__page, 'html5lib')
        #Create a content variable
        for image in __soup.find_all("div",{"class":"rg_meta"}):
            link , Type =json.loads(image.text)["ou"]  ,json.loads(image.text)["ity"]
            #Get the image url
            __image_url = link
            #Check if the image url starts with http
            if(__image_url.startswith("http") is False):
                #Set the image url to this
                __image_url = "{}/{}".format(url, link)
            #Add the found urls
            if(__image_url not in self.__images_to_ignore):
                self.__found_urls.append(__image_url)

        return self.__found_urls

    def generate_random_text(self, max_length=10):
        #Create an alphanumerical variable
        __alpha_numeric = "0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"
        #Empty results variable
        __results = ""
        #Loop through a random range of 0 : random int of length of variable
        for __num in range(1, max_length):
            try:
                __num = random.randrange(0, random.randint(0, (len(__alpha_numeric))))
            except ValueError:
                __num = 0
            #Append the results variable
            __results += __alpha_numeric[__num]
        #Return the results
        return __results

    def crawl_and_download(self, text, download_path, max_crawl_times=10):
        #Set the url to google images search
        url = "https://www.google.com/search?q={}&tbm=isch".format(text)
        #Initiate a found images variable
        __found_images = self.crawl_for_images(url)
        #Loop through the urls
        for __image_url in __found_images:
            if(__image_url not in self.__images_to_ignore):
                if(self.__crawl_time < max_crawl_times):
                    #Open the url for reading
                    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
                    __request = r.Request(url, headers=header)
                    #Download the image
                    __page = r.get(__image_url, headers=header)
                    #Open the download path
                    __file = open("{}/{}.png".format(download_path, self.generate_random_text()), "wb")
                    #Write the lines
                    __file.write(__page.content)
                    #Close the file
                    __file.close()
                    #Notify the image is downloaded
                    print("Downloaded {}!".format(__image_url))
                    #Check if the crawl time is less than the max times
                    self.__images_to_ignore.append(__image_url)
                    #Update the crawl time
                    self.__crawl_time += 1
                    #Crawl and download
                    self.crawl_and_download(url, download_path, max_crawl_times=max_crawl_times)