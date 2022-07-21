# image-process
http://localhost:8888/notebooks/imgprocessingJK.ipynb<br>
http://127.0.0.1:8889/?token=bdb3f00276480d39583fcc6a387dd6108d48ff8fc9bbc0de<br>
file:///C:/Users/Central%20Computer%20Lab/AppData/Roaming/jupyter/runtime/nbserver-8732-open.html<br>
http://localhost:8889/?token=bdb3f00276480d39583fcc6a387dd6108d48ff8fc9bbc0de<br>
http://localhost:8888/notebooks/image%20processing%20jayaram/Untitled.ipynb<br>

token/password bdb3f00276480d39583fcc6a387dd6108d48ff8fc9bbc0de<br>
pip install opencv_python<br>
pip install matplotlib<br>

PROGRAM 1<br>
import cv2<br>
img=cv2.imread('butterflypic.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145365/175269709-0fde0e91-765b-4178-aec1-302d09479c91.png)<br>


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
![image](https://user-images.githubusercontent.com/98145365/175269965-7636253b-89a6-4f13-aee2-289909bfcf76.png)<br>


PROGRAm 4<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("red")<br>
print(img1)<br>
OUTPUT:<br>
(255, 0, 0)<br>

PROGRAM 5<br>
from PIL import ImageColor<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
![image](https://user-images.githubusercontent.com/98145365/175270283-4d8bb0d5-5d4f-4ed9-993f-43f82da1e026.png)<br>


PROGRAM 6<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('butterflypic.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>


OUTPUT:<br>
![image](https://user-images.githubusercontent.com/98145365/173813837-cd414d2d-dc09-4985-b41b-10c5607db435.png)
![image](https://user-images.githubusercontent.com/98145365/173813887-14844208-6714-4025-9963-234528b779de.png)
![image](https://user-images.githubusercontent.com/98145365/173813939-4d41a836-bd42-4044-92b7-e18e96b273fa.png)

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

OUTPUT:<br>
Filename: butterflypic.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (1024, 535)<br>
Width: 1024<br>
Height: 535<br>

program8<br>
import cv2 <br>
img=cv2.imread('butterflypic.jpg')<br>
print('originally image length width',img.shape) <br>
cv2.imshow('original image',img) <br>
cv2.waitKey(0) <br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape) <br>
cv2.waitKey(0)<br>

output<br>
![image](https://user-images.githubusercontent.com/98145365/175270728-792f1f5f-0fc4-446f-9f36-34023a67efbf.png)<br>
originally image length width (535, 1024, 3)<br>
Resized image length width (160, 150, 3)<br>

Program9:<br>
import cv2 <br>
img=cv2.imread('butterflypic.jpg') <br>
cv2.imshow("RGB",img) <br>
cv2.waitKey(0) <br>
img=cv2.imread('butterflypic.jpg',0) <br>
cv2.imshow("Gray",img) <br>
cv2.waitKey(0) <br>
ret, bw_img=cv2.threshold(img,127,100,cv2.THRESH_BINARY) <br>
cv2.imshow("Binary",bw_img) <br>
cv2.waitKey(0) <br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145365/175271317-2b4191cd-ff9f-493c-9848-710ae55ec387.png)<br>
![image](https://user-images.githubusercontent.com/98145365/175271546-170786f8-e04d-4594-96fd-1625ecfbeefc.png)<br>
![image](https://user-images.githubusercontent.com/98145365/175271657-b63c4b39-5328-46d9-ac92-253ef9a99bc7.png)<br>





PROGRAM 11:<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt <br>
img=mping.imread('hgfc.jpg') <br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/175017356-395ef46b-757c-4eb1-bb6a-12c7d63b22f1.png)<br>
hsv_img= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)<br>
light_blue=(1,190,200) <br>
dark_blue=(18,255,255)<br>
mask=cv2.inRange(hsv_img, light_blue, dark_blue)<br>
result=cv2.bitwise_and(img,img, mask=mask)<br>
plt.subplot (1,2,1)<br>
plt.imshow(mask, cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/175017728-47a3218e-5e7c-427b-9fdb-aa0422000a67.png)<br>
![image](https://user-images.githubusercontent.com/98145365/175017848-2c760595-0eb8-4009-b9d9-2dd5a8bbf0e4.png)<br>
<br>
light_white= (0, 0, 200)<br>
dark_white = (145, 60, 255)<br>
mask_white = cv2.inRange (hsv_img, light_white, dark_white)<br>
result_white = cv2.bitwise_and (img, img, mask=mask_white)<br>
plt.subplot(1, 2, 1)<br>
plt.imshow(mask_white, cmap="gray")<br>
plt.subplot(1, 2, 2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/175017961-bd603fa8-50db-4edd-a519-01851eb4f8a8.png)<br>

![image](https://user-images.githubusercontent.com/98145365/175017979-af84f66d-75e3-4bba-b334-71683c410c01.png)<br>

final_mask= mask + mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1, 2, 1)<br>
plt.imshow(final_mask, cmap="gray")<br>
plt.subplot(1, 2, 2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/98145365/175018052-ac837edb-6656-44f8-9f57-3a27aa458dfb.png)<br>

![image](https://user-images.githubusercontent.com/98145365/175018080-0923dcec-2a88-4a3f-aada-f04c0c09cdc1.png)<br>

blur= cv2.GaussianBlur (final_result, (7, 7), 0) <br>
plt.imshow(blur) <br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/175018117-85841ae3-fdde-4e6e-80c8-0252100112f0.png)<br>

PROGRAM 12
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img1=cv2.imread('img1.jpg')
img2=cv2.imread('img2.jpg')
fimg1=img1+img2 
plt.imshow(fimg1)
plt.show()
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()
cv2.imwrite('output.jpg', fimg2)
fimg3=img1*img2 
plt.imshow(fimg3)
plt.show()
cv2.imwrite('output.jpg', fimg3)
fimg4=img1/img2 
plt.imshow(fimg4)
plt.show()
cv2.imwrite('output.jpg', fimg4)
![image](https://user-images.githubusercontent.com/98145365/175023051-c4f5b5b7-2024-4987-aa4a-48a7b1c5804d.png)
![image](https://user-images.githubusercontent.com/98145365/175023200-e3b8c7f8-8669-47c7-a424-1ed8bf0e1f2b.png)
![image](https://user-images.githubusercontent.com/98145365/175023227-4a060f09-216c-4c0e-9247-7d242a5945ab.png)
![image](https://user-images.githubusercontent.com/98145365/175023292-a645d1a3-63d9-474e-97ed-c84389832059.png)

PROGRAM 23/6/22<br>
import cv2 <br>
img = cv2.imread("D:\img1.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) <br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB) <br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV) <br>
cv2.imshow("GRAY image", gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image", hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145365/175261328-88985f9c-08eb-43fb-be1f-eb72c0c39bf5.png)<br>

![image](https://user-images.githubusercontent.com/98145365/175260889-abf9c5da-48c9-40a6-81d4-2eb8f6ca0b6a.png)<br>


import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array = np.zeros([100,200,3], dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img = Image.fromarray(array)<br>
img.save('igt.png')<br>
img.show()<br>
c.waitKey(0)<br>

![image](https://user-images.githubusercontent.com/98145365/175268353-8e64d798-6c50-427e-8ab9-cc4413d65f61.png)<br>

import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('img1.jpg')<br>
image2=cv2.imread('img1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd= cv2.bitwise_and(image1, image2) <br>
bitwiseor= cv2.bitwise_or(image1, image2)<br>
bitwiseXor=cv2.bitwise_xor (image1,image2) <br>
bitwiseNot_img1= cv2.bitwise_not(image1)<br>
#bitwiseNot_img1= cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseor)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1) <br>
#plt.subplot(155)<br>
#plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>
![image](https://user-images.githubusercontent.com/98145365/176402881-8696d728-b82e-4805-97fc-1b0a4f7a5fb2.png)<br>

import cv2<br>
import numpy as np<br>
image = cv2.imread('img1.jpg')<br>
cv2.imshow('Original Image', image)<br>
cv2.waitKey(0)<br>
Gaussian = cv2.GaussianBlur (image, (7, 7), 0)<br>
cv2.imshow('Gaussian Blurring', Gaussian)<br>
cv2.waitKey(0)<br>
median = cv2.medianBlur (image, 5) <br>
cv2.imshow('Median Blurring', median)<br>
cv2.waitKey(0)<br>
bilateral = cv2.bilateralFilter(image, 9, 75, 75)<br>
cv2.imshow('Bilateral Blurring', bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145365/176405600-d5c9ed3c-9ac0-4788-bed3-e8b4ecf9f462.png)<br><br>
![image](https://user-images.githubusercontent.com/98145365/176405880-2d999d6f-bb31-4cd1-b437-8c450033c0be.png)<br><br>
![image](https://user-images.githubusercontent.com/98145365/176406410-671dbadf-8912-48dc-8565-aed6f081f9b0.png)<br>



from PIL import Image<br>
from PIL import ImageEnhance <br>
image =Image.open('img2.jpg')<br>
image.show()<br>
enh_bri =ImageEnhance.Brightness(image)<br>
brightness= 1.5<br>
image_brightened= enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col= ImageEnhance.Color(image)<br>
color= 1.5<br>
image_colored =enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con =ImageEnhance.Contrast (image) <br>
contrast = 1.5<br>
image_contrasted =enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha =ImageEnhance.Sharpness(image)<br>
sharpness =3.0<br>
image_sharped= enh_sha. enhance (sharpness)<br>
image_sharped.show()<br>

![image](https://user-images.githubusercontent.com/98145365/176407863-c45bce7b-7e32-4730-9cd1-93dac0628f88.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176407931-53d481f2-151f-4343-8bd6-8e49401a4640.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176408003-5a9b3ea6-5ddd-4276-8d84-a516205f79a5.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176408163-5e702827-152d-4fc2-90d7-7d6ada59529b.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176408262-5c24e4a6-a9dc-469c-8abc-a1a0ac0696fc.png)<br>


import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image, ImageEnhance<br>
img= cv2.imread('img2.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel = np.ones((5,5), np. uint8)<br>
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) <br>
closing=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)<br>
erosion= cv2. erode (img, kernel, iterations = 1)<br>
dilation=cv2.dilate (img, kernel, iterations = 1)<br>
gradient= cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)<br>
plt.subplot (151)<br>
plt.imshow(opening)<br>
plt.subplot (152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot (155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
![image](https://user-images.githubusercontent.com/98145365/176412514-7a90ff71-acdd-4491-aec2-a8af6af238da.png)<br>

SAME AS PREVIOUS<br>
import cv2<br>
import numpy as np<br>
import matplotlib.image as mpimg<br>
from matplotlib import pyplot as plt<br>
from PIL import Image, ImageEnhance<br>
img= cv2.imread('img1.jpg',0)<br>
#ax=plt.subplots(figsize=(20,10))<br>
kernel = np.ones((5,5), np. uint8)<br>
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) <br>
closing=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)<br>
erosion= cv2. erode (img, kernel, iterations = 1)<br>
dilation=cv2.dilate (img, kernel, iterations = 1)<br>
gradient= cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)<br>
cv2.imshow('opening',opening)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
cv2.imshow('c',closing)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
cv2.imshow('e',erosion)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
cv2.imshow('d',dilation)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
cv2.imshow('d',gradient)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145365/176417486-a7388e27-d82d-4b50-8cfe-e1215ea5070a.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176417549-8d285116-5fe1-4129-bae1-2c763f44f3af.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176417616-bc072bb9-7135-4123-b132-3ba0e8baea00.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176417679-9f58beef-2d0b-4ed2-96f4-b4e70b1b7e09.png)<br>
![image](https://user-images.githubusercontent.com/98145365/176417772-d6f982d4-f8a4-4f61-ae0e-24e29556b3c4.png)<br>


program:to save grayscale image in d drive<br>

import cv2<br>
OriginalImg=cv2.imread('img1.jpg')<br>
GrayImg=cv2.imread('img1.jpg',0)<br>
isSaved=cv2.imwrite('D:/i.jpg', GrayImg) <br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image', GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
 print("The image is successfully saved")<br>
 
 output
 The image is successfully saved
 
![image](https://user-images.githubusercontent.com/98145365/178697014-e9071a0a-9159-448a-b740-ff1a25fcb4fa.png)<br>


import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt <br>
image=cv2.imread('img2.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
 for j in range(0,y):<br>
  if(image[i][j]>50 and image[i][j]<150): <br>
    z[i][j]=255<br>
  else:<br>
    z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/178699466-2b261172-94c0-4766-abba-e8ed402aeea7.png)<br>

without background<br>
import cv2<br>
import numpy as np
from matplotlib import pyplot as plt <br>
image=cv2.imread('img1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x): <br>
 for j in range(0,y):<br>
  if(image[i][j]>50 and image[i][j]<150): z[i][j]=255<br>
 else:<br>
  z[i][j]=0<br>
equ=np.hstack((image, z)) <br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/178702825-790e6bb1-a705-4f0b-bd48-6bb7013b7642.png)<br>


import cv2<br>
from matplotlib import pyplot as plt<br>
img = cv2.imread('img1.jpg',0)<br>
histr = cv2.calcHist([img],[0],None,[256],[0,256])<br>
plt.plot(histr)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/98145365/178945020-acf06160-5acb-4fa3-b3ad-26104225adf9.png)<br>

program 23:
Image Processing Lab Exercise :20/07/2022
1. Program to perform basic image data analysis using intensity transformation:
a) Image negative
b) Log transformation
c) Gamma correction
d)<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings <br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('img1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145365/179943733-83045574-a784-479a-80f5-82b43c4e883d.png)<br>
negative =255- pic<br>
plt.figure(figsize= (6,6))<br>
plt.imshow(negative); <br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145365/179943800-f55240ed-d722-4a2d-8395-0346485351be.png)<br>

%matplotlib inline<br>
import imageio<br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('img2.jpg')<br>
gray=lambda rgb: np.dot(rgb[...,:3], [0.299,0.587,0.114]) <br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(), cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145365/179946734-c39bdc89-3d31-4516-aeb3-2037b962f9be.png)<br>

import imageio<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('img1.jpg')<br>
gamma=2.2<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145365/179948829-86cfde62-0b2d-4d0e-baa7-bc1a607bdf2a.png)<br>

program 24:
2. Program to perform basic image manipulation:
a) Sharpness
b) Flipping
c) Cropping<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
my_image=Image.open('img2.jpg')<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
sharp.save('D:/ip2022.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/179950876-da71dcbd-879a-4505-834d-f8fec2724697.png)<br>
![image](https://user-images.githubusercontent.com/98145365/179950989-69d2e850-e246-4d92-8bc9-fda1ea38bdd5.png)<br>

import matplotlib.pyplot as plt<br>
img=Image.open('img1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
flip=img.transpose (Image.FLIP_LEFT_RIGHT)<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/179953006-7780b2ee-55b6-4b39-8709-0267bd50e012.png)<br>

import matplotlib.pyplot as plt<br>
from PIL import Image<br>
im = Image.open('img2.jpg')<br>
width, height=im.size<br>
im1= im.crop ((280,100,600,400))<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/179955317-aa0a3ff5-ff3d-4c09-9c54-d41e411dd067.png)<br>

![image](https://user-images.githubusercontent.com/98145365/179955181-599b3748-2b02-45ed-903f-276c4fa93d8b.png)<br>

MATRIX TO IMAGE<br>

import cv2<br>
import numpy as np<br>
img = np.random.randint(222, size=(300, 600,3))<br>
gen = np.array(img ,dtype=np.uint8)<br>
cv2.imshow('i',gen)<br>
cv2.waitKey(0)<br>
cv2.destroyWindow('i')<br>

import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0, 125, 255)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        arr[y, x] = (int(r), int(g), int(b))<br>

plt.imshow(arr, cmap='gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145365/180191227-7cac8291-322d-4eec-9767-fadc25bffa5e.png)<br>
















