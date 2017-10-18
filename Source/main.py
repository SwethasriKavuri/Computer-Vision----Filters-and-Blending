# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   height, width = img_in.shape[:2]
   blue_ch, green_ch , red_ch = cv2.split(img_in);
   blue_hist = cv2.calcHist([blue_ch],[0],None,[256],[0,256]);
   cdf_blue = np.cumsum(blue_hist);
   green_hist = cv2.calcHist([green_ch],[0],None,[256],[0,256]);
   cdf_green = np.cumsum(green_hist);
   red_hist = cv2.calcHist([red_ch],[0],None,[256],[0,256]);
   cdf_red  = np.cumsum(red_hist);
   h_blue =np.around(np.subtract(cdf_blue, np.amin(cdf_blue)));
   cv2.divide(h_blue,blue_ch.size, h_blue);
   cv2.multiply(h_blue, 255, h_blue);
   h_green =np.around(np.subtract(cdf_green, np.amin(cdf_green)));
   cv2.divide(h_green,green_ch.size, h_green);
   cv2.multiply(h_green, 255, h_green);
   h_red =np.around(np.subtract(cdf_red, np.amin(cdf_red)));
   cv2.divide(h_red,red_ch.size, h_red);
   cv2.multiply(h_red, 255, h_red);
   blue_n = h_blue[blue_ch.ravel()].reshape(blue_ch.shape);
   green_n = h_green[green_ch.ravel()].reshape(green_ch.shape);
   red_n = h_red[red_ch.ravel()].reshape(red_ch.shape);
   image = cv2.merge([blue_n, green_n, red_n]);
   cv2.waitKey(0)
   img_out =  np.array(image).astype('uint8')

   return True, img_out
    
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
   img = img_in
   fourier = np.fft.fft2(img)
   fourier_shift = np.fft.fftshift(fourier)
   magnitude_spectrum = 20*np.log(np.abs(fourier_shift))
   rows = img.shape[0]
   cols = img.shape[1]
   crow,ccol = rows/2 , cols/2
   fourier_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
   fourier_ishift = np.fft.ifftshift(fourier_shift)
   img_back = np.fft.ifft2(fourier_ishift)
   img_back = np.abs(img_back)
   img_out = img_back
   img_out = np.array(img_out).astype('uint8')
   return True, img_out   

def high_pass_filter(img_in):
   img = img_in
   fourier = np.fft.fft2(img)
   fourier_shift = np.fft.fftshift(fourier)
   magnitude_spectrum = 20*np.log(np.abs(fourier_shift))
   rows = img.shape[0]
   cols = img.shape[1]
   crow,ccol = rows/2 , cols/2
   mask = np.zeros(fourier_shift.shape)
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1
   fourier_shift = fourier_shift * mask
   fourier_ishift = np.fft.ifftshift(fourier_shift)
   img_back = np.fft.ifft2(fourier_ishift)
   img_back = np.abs(img_back)
   img_out = img_back
   img_out = np.array(img_out).astype('uint8')
   return True, img_out 

def deconvolution(img_in): 
   blurred = img_in
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   def ft(im, newsize=None):
       dft = np.fft.fft2(np.float32(im),newsize)
       return np.fft.fftshift(dft)

   def ift(shift):
       f_ishift = np.fft.ifftshift(shift)
       img_back = np.fft.ifft2(f_ishift)
       return np.abs(img_back)
   imf = ft(blurred, (blurred.shape[0],blurred.shape[1]))
   gkf = ft(gk, (blurred.shape[0],blurred.shape[1]))
   imconvf = imf / gkf
   recovered = ift(imconvf)  
   img_out = recovered
   img_out = np.array(img_out*255).astype('uint8')
   img_out = np.clip(img_out, 0, 255)
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], 0);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

	levels = 5

# Generate Gaussian pyramid for imgA
	gaussianPyramidA = [img_in1.copy()]
	for i in range(1, levels):
    		gaussianPyramidA.append(cv2.pyrDown(gaussianPyramidA[i - 1]))

# Generate Gaussian pyramid for imgB
	gaussianPyramidB = [img_in2.copy()]
	for i in range(1, levels):
    		gaussianPyramidB.append(cv2.pyrDown(gaussianPyramidB[i - 1]))

# Generate the inverse Laplacian Pyramid for imgA
	laplacianPyramidA = [gaussianPyramidA[-1]]
	for i in range(levels - 1, 0, -1):
    		size = (gaussianPyramidA[i-1].shape[1], gaussianPyramidA[i-1].shape[0])   
    		laplacian = cv2.subtract(gaussianPyramidA[i - 1], cv2.pyrUp(gaussianPyramidA[i],dstsize = size))
    		laplacianPyramidA.append(laplacian)

# Generate the inverse Laplacian Pyramid for imgB
	laplacianPyramidB = [gaussianPyramidB[-1]]
	for i in range(levels - 1, 0, -1):
    		size = (gaussianPyramidB[i-1].shape[1], gaussianPyramidB[i-1].shape[0])   
   		laplacian = cv2.subtract(gaussianPyramidB[i - 1], cv2.pyrUp(gaussianPyramidB[i],dstsize = size))
   		laplacianPyramidB.append(laplacian)

# Add the left and right halves of the Laplacian images in each level
	laplacianPyramidComb = []
	for laplacianA, laplacianB in zip(laplacianPyramidA, laplacianPyramidB):
    		rows, cols, dpt = laplacianA.shape
    		laplacianComb = np.hstack((laplacianA[:, 0:cols / 2], laplacianB[:, cols / 2:]))
    		laplacianPyramidComb.append(laplacianComb)

# Reconstruct the image from the Laplacian pyramid
	img = laplacianPyramidComb[0]
	for i in range(1, levels):
    		size = (laplacianPyramidComb[i].shape[1], laplacianPyramidComb[i].shape[0])
    		img = cv2.add(cv2.pyrUp(img,dstsize = size), laplacianPyramidComb[i])
# Display the result
	#cv2.imshow('image', img)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()

   	img_out = img # Blending result
   
  	return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
