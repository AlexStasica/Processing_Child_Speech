# CHILDES Web Scraper

This script allows you to **automatically download all `.wav` audio files** from a specific CHILDES project directory hosted on [TalkBank](https://media.talkbank.org/childes/). 

---

## Usage

To run the scraper:

1. Open `webscrapingChildes.py`.
2. Modify the input of the `downloader()` function (line 96) with your desired parameters.
3. Run the script.

Example:
```python
downloader(
    url="https://media.talkbank.org/childes/Clinical-Other/Zwitserlood/TD/8910",
    cookie="INSERT_YOUR_COOKIE_HERE"
)
