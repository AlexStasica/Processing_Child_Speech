To automatically download all files from a CHILDES webpage, open the webscrappingChildes.py script, modify the input of the function call "downloader()" on line 96, and run the script.


How to obtain the informations needed for the input:


- URL:  Use the complete URL of the webpage listing all the audio files
        (For instance: https://media.talkbank.org/childes/Clinical-Other/Zwitserlood/TD/8910 , or https://media.talkbank.org/childes/DutchAfrikaans/SchlichtingVanKampen/Sanne/0wav )


- COOKIE: Use the value of your personal CHILDES talkbank cookie, obtain it follwing these steps:
            1. Login into your CHILDES account
            2. Open any webpage within the CHILDES website
            3. Open the developer tools window (Press F12)
            4. Access the Application tab (Chrome) or the Storage tab (Firefox) on the developer window
            5. In the left sidebar, access: Storage > Cookies > https://childes.talkbank.org
            6. Copy the string displayed in the "Cookie Value" (/!\ "Show URL-decoded" must NOT be checked /!\)
