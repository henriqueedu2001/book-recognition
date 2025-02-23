import numpy as np
import cv2
from typing import *
from numpy.typing import *
from cv2.typing import *
from matplotlib.typing import *
from PIL.Image import Image

class Segmenter:
    def segment_img(img: Union[ArrayLike, Image, MatLike], alpha: float = 5., beta: float = 1.5):
        gray_img = Segmenter.convert_to_gray(img)
        smooth_img = Segmenter.gaussian_blur(gray_img)

        sobel_hor, sobel_ver = Segmenter.sobel(smooth_img, orientation = 0), Segmenter.sobel(smooth_img, orientation = 1)
        spikes_map_hor, spikes_map_ver = Segmenter.get_spikes_map(sobel_hor, orientation = 0), Segmenter.get_spikes_map(sobel_ver, orientation = 1)
        spikes_diff_hor, spikes_diff_ver = Segmenter.get_spikes_diff(spikes_map_hor), Segmenter.get_spikes_diff(spikes_map_ver)

        orientation = Segmenter.detect_orientation(spikes_diff_hor, spikes_diff_ver)

        # horizontal
        if orientation == 0:
            spikes_mask = Segmenter.get_spikes_mask(spikes_diff_hor, alpha = alpha, beta = beta)
            spikes_centroids = Segmenter.get_spikes_centroids(spikes_mask)
            slices = Segmenter.slice_img(img, spikes_centroids, orientation)
            return slices
        
        # vertical
        elif orientation == 1:
            spikes_mask = Segmenter.get_spikes_mask(spikes_diff_ver, alpha = alpha)
            spikes_centroids = Segmenter.get_spikes_centroids(spikes_mask)
            slices = Segmenter.slice_img(img, spikes_centroids, orientation)
            return slices
    
    
    def convert_to_gray(img: Union[ArrayLike, Image, MatLike]) -> Union[ArrayLike, Image, MatLike]:
        """Converts an image to gray scale.

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image.

        Returns:
            Union[ArrayLike, Image, MatLike]: The gray scaled image.
        """

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def gaussian_blur(img: Union[ArrayLike, Image, MatLike], kernel_size: int = 3) -> Union[ArrayLike, Image, MatLike]:
        """Blurs an image with gaussian blur.

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image.
            kernel_size (int, optional): the size of the gaussian kernel. Defaults to 3.

        Returns:
            Union[ArrayLike, Image, MatLike]: the blurred image.
        """

        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    

    def laplacian(img: Union[ArrayLike, Image, MatLike]) -> Union[ArrayLike, Image, MatLike]:
        """Convolve the laplacian filter and the input image.

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image.

        Returns:
            Union[ArrayLike, Image, MatLike]: the image after the convolution.
        """

        return cv2.Laplacian(img, cv2.CV_64F)
    

    def convolve(img: Union[ArrayLike, Image, MatLike], kernel: NDArray) -> Union[ArrayLike, Image, MatLike]:
        """Convolves the input image with the given kernel.

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image.
            kernel (NDArray): the kernel matrix.

        Returns:
            Union[ArrayLike, Image, MatLike]: the convolved image.
        """

        return cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    

    def sobel(img: Union[ArrayLike, Image, MatLike], orientation: Literal[0, 1] = 0) -> Union[ArrayLike, Image, MatLike]:
        """Convolves the input image with the sobel kernels, for edge detection. It can be convolved both in the 
        horizontal axis (Gy) and the vertical axis (Gx).

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image.
            orientation (int, optional): the orientation; 0 means horizontal (Gy sobel) and 1 means vertical (Gx sobel). Defaults to 0.

        Returns:
            Union[ArrayLike, Image, MatLike]: the convolved image.
        """

        # horizontal edges
        if orientation == 0:
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.convertScaleAbs(sobel_y) 
            return sobel_y
        
        # vertical edges
        elif orientation == 1:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_x = cv2.convertScaleAbs(sobel_x) 
            return sobel_x
    

    def get_spikes_map(img: Union[ArrayLike, Image, MatLike], orientation: Literal[0, 1] = 0) -> NDArray:
        """Generates a vector s = (s_1, s_2, s_3, ..., s_n), where s_k is bigger when there is an edge on that position.

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image, after the edge detection convolutions.
            orientation (int, optional): the orientation of the mean; 0 means horizontal and 1 means vertical. Defaults to 0.

        Returns:
            NDArray: a vector s = (s_1, s_2, s_3, ..., s_n) where s_k is the mean along a given axis of the image.
        """
        # horizontal edges
        if orientation == 0: return np.mean(img, axis=1)
        
        # vertical edges
        elif orientation == 1: return np.mean(img, axis=0)
        

    def get_spikes_diff(spikes_map: NDArray) -> NDArray:
        """Computes the discrete difference on the spike map.

        Args:
            spikes_map (NDArray): the spike map.

        Returns:
            NDArray: the spikes map discrete differences.
        """
        diff = np.diff(spikes_map)
        mean, max, min = np.mean(diff), np.max(diff), np.min(diff)
        amp = np.max([max, -min])
        diff = (diff - mean)/amp

        return diff
    

    def detect_orientation(horizontal_spikes_diff: NDArray, vertical_spikes_diff: NDArray) -> Literal[0, 1]:
        """Detects the orientation of the books based on the variance of the spikes discrete difference series.

        Args:
            horizontal_spikes (NDArray): the horizontal spikes discrete difference series.
            vertical_spikes (NDArray): the vertical spikes discrete difference series.

        Returns:
            Literal[0, 1]: the orientation; 0 means horizontal and 1 means vertical.
        """
        var_hor = np.var(horizontal_spikes_diff)
        var_ver = np.var(vertical_spikes_diff)

        return 1 if var_hor > var_ver else 0
    

    def get_spikes_mask(spikes_diff: NDArray, alpha: float = 5., beta: float = 1.5) -> NDArray:
        """Generates a mask, True in the positions where there is a spike and False elsewhere.

        Args:
            spikes_diff (NDArray): the spikes map discrete differences. 
            alpha (float, optional): percentage or sequence of percentages for the percentiles to compute. Defaults to 5.
            beta (float, optional): the sensibility to outliers. Defaults to 1.5.

        Returns:
            NDArray: a vector m = (m_1, m_2, m_3, ..., m_n) where m_k is True if there is a spike on that
        position and False otherwise.
        """
        Q1 = np.percentile(spikes_diff, alpha, method='midpoint')
        Q3 = np.percentile(spikes_diff, 100. - alpha, method='midpoint')
        IQR = Q3 - Q1
        upper = Q3 + beta*IQR
        lower = Q1 - beta*IQR

        new_series = spikes_diff > upper
        new_series = spikes_diff < lower
        
        return new_series
    

    def get_spikes_centroids(spikes_mask: NDArray) -> List[int]:
        """Computes the centroids positions of the spikes mask.

        Args:
            spikes_mask (NDArray): the spikes mask.

        Returns:
            list: a list with the centroids positions.
        """
        positions = []
        n = len(spikes_mask)

        on_spike = False
        spike_start = 0
        for i in range(n):
            if spikes_mask[i] == True and on_spike == False:
                on_spike = True
                spike_start = i
                
            elif spikes_mask[i] == False and on_spike == True:
                spike_end = i
                spike_pos = (spike_start + spike_end)//2
                positions.append(spike_pos)
                on_spike = False
        
        return positions
    

    def slice_img(img: Union[ArrayLike, Image, MatLike], slice_pos: list, orientation: Literal[0, 1] = 0) -> List[ArrayLike, Image, MatLike]:
        """Slices the input image with the sequence of slice positions.

        Args:
            img (Union[ArrayLike, Image, MatLike]): the input image.
            slice_pos (list): the sequence of slice positions.
            orientation (Literal[0, 1], optional): the orientation; 0 mean horizontal and 1 means vertical. Defaults to 0.

        Returns:
            List[ArrayLike, Image, MatLike]: a list with the cropped images.
        """
        slices = []
        w, h = img.shape[1], img.shape[0]

        # horizontal
        if orientation == 0:
            last_y = 0
            slice_pos.append(h)
            for new_y in slice_pos:
                cropped_img = img[last_y:new_y, 0:w]
                last_y = new_y
                slices.append(cropped_img)
        
        # vertical
        elif orientation == 1:
            last_x = 0
            slice_pos.append(w)
            for new_x in slice_pos:
                cropped_img = img[0:h, last_x:new_x]
                last_x = new_x
                slices.append(cropped_img)
            
        
        return slices