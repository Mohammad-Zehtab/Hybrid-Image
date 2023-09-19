import numpy as np
import cv2
import matplotlib.pyplot as plt



#importing cat image
gray = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)


n=gray.shape [0] #number of rows in the mask
m=gray.shape [1]
#number of columns in the mask
x=0
y=0
sigma = 10

#identifying the center
Centerx = (n-1)/2
Centery = (m-1)/2


array = np.zeros((n,m))

#making the desired low-pass Gaussian matrix

while (x < n):
  array [x,y] =(1/((2*3.14*sigma*sigma)**0.5))*np.exp(-(((x-Centerx)*(x-Centerx))+((y-Centery)*(y-Centery)))/(2*sigma*sigma))
  x = x +1

  if (x == n):
    x = 0
    y = y+1
  if (y==m):
    break

array = ((array- np.min (array))/(np.max(array)-np.min(array))*255)


#fft of the cat image
f = np.fft.fft2 (gray)

#fft of the Gaussian Matrix
array = np.fft.fftshift (array)

#using the filter
filteredf = np.multiply(f, array) 

#final cat image
filteredimg = np.fft.ifft2(filteredf)
filteredimg = np.real (filteredimg)
filteredimg = (filteredimg/np.max (filteredimg))*255

#cat image plot
fig1 = plt.figure(figsize = (7, 5)) 
fig1.add_subplot(121)
plt.title("Original Image")
plt.imshow(gray,  cmap= 'gray' , vmin=0, vmax=255)
fig1.add_subplot(122)
plt.title("Filtered Image")
plt.imshow(filteredimg, cmap= 'gray' , vmin=0, vmax=255)
plt.show()


#importing dog image
gray1 = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)


n=gray1.shape [0] #number of rows in the mask
m=gray1.shape [1]
 #number of columns in the mask
x=0
y=0
sigma = 10
#identifying thecenter
Centerx = (n-1)/2
Centery = (m-1)/2


array = np.zeros((n,m))

#making the desired high-pass Gaussian matrix

while (x < n):
  array [x,y] =1-(1/((2*3.14*sigma*sigma)**0.5))*np.exp(-(((x-Centerx)*(x-Centerx))+((y-Centery)*(y-Centery)))/(2*sigma*sigma))
  x = x +1

  if (x == n):
    x = 0
    y = y+1
  if (y==m):
    break

array = ((array- np.min (array))/(np.max(array)-np.min(array))*255)

#fft of the dog image
f1 = np.fft.fft2 (gray1)

#fft of the Gaussian Matrix
array = np.fft.fftshift (array)

#using the filter
filteredf1 = np.multiply(f1, array) 

#final dog image
filteredimg1 = np.fft.ifft2(filteredf1)
filteredimg1 = np.real (filteredimg1)
filteredimg1 = (filteredimg1/np.max (filteredimg1))*255

#dog image plot
fig2 = plt.figure(figsize = (7, 5)) 
fig2.add_subplot(121)
plt.title("Original Image")
plt.imshow(gray1,  cmap= 'gray' , vmin=0, vmax=255)
fig2.add_subplot(122)
plt.title("Filtered Image")
plt.imshow(filteredimg1, cmap= 'gray' , vmin=0, vmax=255)
plt.show()

#making the hybrid image
# Resize filteredimg1 to match the dimensions of filteredimg
filteredimg1 = cv2.resize(filteredimg1, (gray.shape [1], gray.shape [0]))

# Create the hybrid image by adding the two filtered images
hybridimg = cv2.addWeighted(filteredimg, 0.5, filteredimg1, 0.5, 0)

#hybrid image plot
fig3 = plt.figure(figsize = (5, 3)) 
fig3.add_subplot(121)
plt.title("Hybrid Image")
plt.imshow(hybridimg,  cmap= 'gray' , vmin=0, vmax=255)
fig3.add_subplot(154)
plt.title("Hybrid Image 2")
plt.imshow(hybridimg,  cmap= 'gray' , vmin=0, vmax=255)
plt.show()