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


#google for result count------------------------------------

search = fixed_question, "+",ocr_a1, " -",ocr_a2, " -",ocr_a3

r = requests.get("https://www.google.com/search", params={'q':search})
print(search)
soup = BeautifulSoup(r.text, "lxml")
res = soup.find("div", {"id": "resultStats"})
print(res.text, "for", ocr_a1)

search1 = fixed_question, "+",ocr_a2, " -",ocr_a1, " -",ocr_a3

r = requests.get("https://www.google.com/search", params={'q':search1})

soup = BeautifulSoup(r.text, "lxml")
res = soup.find("div", {"id": "resultStats"})
print(res.text, "for", ocr_a2)

search2 = fixed_question, "+",ocr_a3, " -",ocr_a2, " -",ocr_a1

r = requests.get("https://www.google.com/search", params={'q':search2})

soup = BeautifulSoup(r.text, "lxml")
res = soup.find("div", {"id": "resultStats"})
print(res.text, "for", ocr_a3)
