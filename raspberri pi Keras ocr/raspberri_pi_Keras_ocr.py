import picamera
import time
import matplotlib.pyplot as plt
import keras_ocr
import os
from gtts import gTTS
from IPython.display import Audio

pipeline = keras_ocr.pipeline.Pipeline()

camera = picamera.PiCamera()
camera.resolution = (2592, 1944)

time.sleep(3)
camera.capture('snapshot.jpg')


custom_images = []
images = [ keras_ocr.tools.read(path) for path in ['/home/pi/picture1.jpg']]

prediction_groups = pipeline.recognize(images)

fig, axs = plt.subplots(nrows=len(images), figsize=(10, 10))
if(len(custom_images)==1):
  for image, prediction in zip(images, prediction_groups):
     keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=axs)
else:
  for ax, image, prediction in zip(axs, images, prediction_groups):
     keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=ax)

with open('results.txt', 'a+') as f:
   for idx, prediction in enumerate(predictions):
     if(idx != 0):
       print("\n")
       f.write("\n\n")
     for word, array in prediction:
       if word == "\n" :
         print("\n")
         f.write("\n")
       else:
         print(word, end = ' ')
         f.write(word + " ")

files.download("results.txt")

textpath = 'results.txt'

with open(textpath, mode='r', encoding='UTF-8') as text:
     script = text.read()

script.replace('\n', '')

tts = gTTS(text=script, lang='en')
tts.save("wordtts.mp3") 

display(Audio('wordtts.mp3', autoplay=True))

def removeAllFile(filePath):
     if os.path.exists(filePath):
         for file in os.scandir(filePath):
             os.remove(file.path)
         return 'Remove All File'
     else:
         return 'Directory Not Found'

print(removeAllFile('/home/pi'))

os.remove('/home/pi/results.txt')
os.remove('/home/pi/wordtts.mp3')
