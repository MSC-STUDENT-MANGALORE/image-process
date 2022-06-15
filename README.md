# image-process
pip install opencv_python<br>
pip install matplotlib<br>

PROGRAM 1<br>
import cv2<br>
img=cv2.imread('butterflypic.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

PROGRAM 2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('butterflypic.jpg')<br>
plt.imshow(img)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145365/173806136-56fc3fac-33a2-49f4-8ec1-c53624b88ff7.png)<br><br>

PROGRAm 3<br>
from PIL import Image<br>
img=Image.open("butterflypic.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

PROGRAm 4<br><br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("red")<br>
print(img1)<br>
OUTPUT:<br>
(255, 0, 0)<br>

PROGRAM 5<br>
from PIL import ImageColor<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>


PROGRAM 6<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('butterflypic.jpg')<br>
plt.imshow(img)<br>
plt.show()<br><br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145365/173806612-0b953b0c-7ab6-4915-9f73-8dfecb25e4fa.png)<br>


PROGRAM 7<br>
from PIL import Image<br>
image=Image.open('butterflypic.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

OUTPUT:<br><br>
Filename: butterflypic.jpg<br>
Format: JPEG<br><br><br>
Mode: RGB<br><br><br>
Size: (1024, 535)<br>
Width: 1024<br><br>
Height: 535<br><br>
