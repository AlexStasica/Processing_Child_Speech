import os
import requests
from bs4 import BeautifulSoup
import re

def downloader(cookie,url):
    #Function arguments:
        #cookie: The personal childes talkbank cookie value manually obtained by the user
        #url: The url from which to access all desired download links, for instance: 

    ####################################################################################################################################
    #DATA SPECIFICATION

    #Building the path name for the output directory
    path = url.split("/")
    output_dir = "./CHILDES Downloads"
    for e in path:
        if path.index(e) > path.index("childes"):
            output_dir += ("/"+e)

    #Creating the output directoryhttps://media.talkbank.org/childes/Clinical-Other/Zwitserlood/TD/8910/0wav
    os.makedirs(output_dir, exist_ok=True)

    #Specifying the necessary info for the website to allow the download, i.e. the referer page, and the personal cookie ID
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": url,
        'Cookie' : f'talkbank={cookie}'
        }


    ####################################################################################################################################
    #LISTING ALL FILES TO DOWNLOADS

    # Send an HTTP GET request to the URL
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:

        # Access the HTML content of the webpage, parsing the HTML and finding all links, and creating the list of all the links that end in "f=save", i.e. all the download links
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        links = soup.find_all("a")
        files_to_download = []
        
        for link in links:
            if re.match(r".*f=save$",link.get("href")):
                files_to_download.append(link.get("href"))

    else:
        print(f"Failed to fetch webpage. Status code: {response.status_code}")


    ####################################################################################################################################
    #DOWNLOADING THE FILES

    print(f"Starting downloads, {len(files_to_download)} files found.\n")

    #Looping through the list of all the files we want to download
    for file_url in files_to_download:

        #Creating the file name (I.E. the last element of the url, minus the "?=save" suffix), creating the file path
        file_name_0 = file_url.split("/")
        file_name = str(file_name_0[-1]).removesuffix("?f=save")
        file_path = output_dir + "/" + file_name

        try:
            response = requests.get(file_url,allow_redirects=True,stream=True, headers=headers)

            if response.status_code == 200:
                #print(f"Status code : {response.status_code}\nResponse Headers: {response.headers}\n") #DEBUG
                print(f"Downloading file: {file_name}\nFile size : {response.headers.get('Content-Length')} bits")

                #Checking if the file already exists, AND, if the existing file size is the same as the one on the distant server.
                if os.path.exists(file_path) and os.path.getsize(file_path) == int(response.headers.get('Content-Length')):
                    print(f"File: {file_name} already downloaded, skipping\n\n")
                    continue

                #Downloading the file    
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                    print(f"Downloaded {file_name}\n\n")


            else:
                print(f"Failed to download {file_name}. Status: {response.status_code}")


        except Exception as e:
            print(f"Error downloading {file_name}: {e}")

    

downloader("<INSERT COOKIE VALUE HERE>","<INSERT URL HERE>")