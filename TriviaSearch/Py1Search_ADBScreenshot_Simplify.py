import asyncio
import os
import time
import PIL

import cv2
import pytesseract
import requests
from PIL import Image
from bs4 import BeautifulSoup
from rake_nltk import Rake

import numpy as np
import cv2
import shlex
import subprocess
from matplotlib import pyplot as plt

command = r"adb shell screencap -p"
proc = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE)
out = proc.stdout.read(30000000)
img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_GRAYSCALE)
if img is not None:
    cv2.imwrite('fullsize.png', img)


# open the screenshot

screenshot = Image.open('fullsize.png')

# crop the question-------------------------------------
crop_question = screenshot.crop((30, 230, 690, 520))
crop_question.save('cropped_question.jpg')

# load the question and convert it to grayscale
cvquestion = cv2.imread('cropped_question.jpg')
gray = cv2.cvtColor(cvquestion, cv2.COLOR_BGR2GRAY)

gray = cv2.threshold(gray, 0, 255,
                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

ocr_question = pytesseract.image_to_string(Image.open(filename))
fixed_question = ocr_question.replace('\n', ' ')

os.remove(filename)

# crop answer 1------------------------------------------
crop_a1 = screenshot.crop((80, 525, 640, 620))
crop_a1.save('cropped_a1.jpg')

# load answer 1 and convert it to grayscale
answer1 = cv2.imread('cropped_a1.jpg')
gray = cv2.cvtColor(answer1, cv2.COLOR_BGR2GRAY)

gray = cv2.threshold(gray, 0, 255,
                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

ocr_a1 = pytesseract.image_to_string(Image.open(filename))

os.remove(filename)

# crop answer 2---------------------------------------------
crop_a2 = screenshot.crop((80, 635, 635, 730))
crop_a2.save('cropped_a2.jpg')

# load answer 1 and convert it to grayscale
answer2 = cv2.imread('cropped_a2.jpg')
gray = cv2.cvtColor(answer2, cv2.COLOR_BGR2GRAY)

gray = cv2.threshold(gray, 0, 255,
                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

ocr_a2 = pytesseract.image_to_string(Image.open(filename))

os.remove(filename)

# crop answer 3---------------------------------------------
crop_a3 = screenshot.crop((80, 750, 640, 840))
crop_a3.save('cropped_a3.jpg')

# load answer 1 and convert it to grayscale
answer3 = cv2.imread('cropped_a3.jpg')
gray = cv2.cvtColor(answer3, cv2.COLOR_BGR2GRAY)

gray = cv2.threshold(gray, 0, 255,
                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# write the grayscale image to disk as a temporary file so we can apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

ocr_a3 = pytesseract.image_to_string(Image.open(filename))

os.remove(filename)

# print results---------------------------------------------

print(fixed_question)
print()
print(ocr_a1)
print(ocr_a2)
print(ocr_a3)
print()

# Clean up OCR'd question to just keywords

r = Rake() # Uses stopwords for english from NLTK, and all punctuation characters.
r.extract_keywords_from_text(fixed_question)
phrases = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
phrases_clean = (' '.join('"{0}"'.format(w) for w in phrases))

print("Original Question: ",fixed_question)
print("Extracted Phrases: ", phrases_clean)
print()


# google for result count------------------------------------
print("Searching Simplified Terms")
search = phrases_clean, " +", "\"", ocr_a1, "\"", " -", "\"", ocr_a2, "\"", " -", "\"", ocr_a3, "\""
searchclean1 = ''.join(search)
print(searchclean1)


r1 = requests.get("https://www.google.com/search", params={'q': searchclean1})

soup = BeautifulSoup(r1.text, "lxml")
res1 = soup.find("div", {"id": "resultStats"})
print(res1.text, "for", ocr_a1)
result1 = (res1.text, "for", ocr_a1)

search1 = phrases_clean, " +", "\"", ocr_a2, "\"", " -", "\"", ocr_a1, "\"", " -", "\"", ocr_a3, "\""
searchclean2 = ''.join(search1)

r2 = requests.get("https://www.google.com/search", params={'q': searchclean2})

soup = BeautifulSoup(r2.text, "lxml")
res2 = soup.find("div", {"id": "resultStats"})
print(res2.text, "for", ocr_a2)
result2 = (res2.text, "for", ocr_a2)

search2 = phrases_clean, " +", "\"", ocr_a3, "\"", " -", "\"", ocr_a2, "\"", " -", "\"", ocr_a1, "\""
searchclean3 = ''.join(search2)

r3 = requests.get("https://www.google.com/search", params={'q': searchclean3})

soup = BeautifulSoup(r3.text, "lxml")
res3 = soup.find("div", {"id": "resultStats"})
print(res3.text, "for", ocr_a3)
result3 = (res3.text, "for", ocr_a3)