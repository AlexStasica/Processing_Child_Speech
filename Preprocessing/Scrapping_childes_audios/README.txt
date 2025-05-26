# CHILDES Web Scraper

This script allows you to **automatically download all `.wav` audio files** from a specific CHILDES project directory hosted on [TalkBank](https://media.talkbank.org/childes/). 

---

## Usage

To run the scraper:

1. Open `webscrapingChildes.py`.
2. Modify the input of the `downloader()` function (line 96) with your desired parameters.
3. Put your personnal cookie
4. Run the script.

## To find your cookie:

Use the value of your personal CHILDES talkbank cookie, obtain it follwing these steps: 
1. Login into your CHILDES account 
2. Open any webpage within the CHILDES website
3. Open the developer tools window (Press F12) 
4. Access the Application tab (Chrome) or the Storage tab (Firefox) on the developer window 
5. In the left sidebar, access: Storage > Cookies > https://childes.talkbank.org 
6. Copy the string displayed in the "Cookie Value" (/!\ "Show URL-decoded" must NOT be checked /!)


Example:
```python
downloader(
    url="https://media.talkbank.org/childes/Clinical-Other/Zwitserlood/TD/8910",
    cookie="INSERT_YOUR_COOKIE_HERE"
)


## Contact

Louis Berard (berardlouis@gmail.com)
