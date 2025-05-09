from datetime import timedelta
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit

class AnalyzeMOT() : 
    """
    Class for automating data analysis of TIFF-based data from a Magneto-Optical Trap.
    """
    def __init__(self, path, sorting_method=sorted, x_offset=None, y_offset=None, image_size=None) : 
        """ 
        ---- CONSTRUCTOR ---- 
        path            :   relative or absolute path to directory containing the data.
        sorting_method  :   optional sorting function applicable to list of files.
                            defaults to alphabetical sorting.
        """
        self._path = path
        self._sorting_method = sorting_method
        
        self.files = self.load_files(self.sorting_method)

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.image_size = image_size
    
    def _gaussian(self, x, *args) : 
        A, sigma, mu, D = args[0], args[1], args[2], args[3]
        return A * np.exp(-0.5 * (x-mu)**2 / sigma**2) + D
    
    @property
    def path(self) : 
        return self._path

    @path.setter
    def path(self, new_path) : 
        self._path = new_path
        self.files = self.load_files()

    @property
    def sorting_method(self) : 
        return self._sorting_method
    
    @sorting_method.setter
    def sorting_method(self, new_sorting_method) : 
        self._sorting_method = new_sorting_method
        self.files = self.load_files(self.sorting_method)

    def get_image_matrix(self, filename, apply_zoom) :
        """
        Returns the bit-depth matrix of <filename> element in <self._path> directory.
        """
        if apply_zoom and (self.image_size is None or self.x_offset is None or self.y_offset is None):
            raise ValueError("To apply zoom, you must first set image_size, x_offset, and y_offset.")
        
        image = plt.imread( os.path.join(self._path, filename) )
        if(apply_zoom) : image = self.zoom_matrix(image)
        
        return image
    
    def subtract_background(self, file_matrix, background_matrix) : 
        """
        Subtracts background bit-depth matrix <background_matrix> from <file_matrix>.  
        """
        return np.clip(file_matrix-background_matrix, a_min=0, a_max=255)

    def show_image(self, filename) : 
        """ 
        Returns a <matplotlib.image.AxesImage> element based on file <filename> in directory <self._path>. 
        """
        return plt.imshow( plt.imread( os.path.join(self._path, filename) ) )

    def parse_time_stamp(self, filename) : 
        """ 
        Converts time stamp from untriggered saved image from Camera Controls. 
        Note: These time stamps do not consider change of day, month, year, and etc.
        """
        float_times = [float(t) for t in filename.replace('.tiff', '').split('-')[:3]]
        t_seconds = timedelta(hours=float_times[0], minutes=float_times[1], seconds=float_times[2]).total_seconds()
        return t_seconds

    def load_files(self, sorting_method=None) : 
        """ 
        Load files from directory and apply sorting method.
        Returns sorted list of files.
        """
        return self.sorting_method(os.listdir(self.path))
    
    def process_filenames(self, method, relative_time=True) :
        """ 
        Processes all files using class-defined or user-defined <method>. 
        Class-defined <method>'s to use are: 
        *  parse_time_stamp  * 

        """ 
        metadata = np.array([method(file) for file in self.files])
        if(relative_time) : metadata = metadata - metadata[0]
        return np.array(metadata)
    
    def get_summed_bit_depths(self, threshold=0, normed=True, background_image=None, apply_zoom=False) : 
        """ 
        Computes the sum of all camera pixels of <self.files> above provided threshold
        threshold       :       pixel threshold for summation
        normed          :       return normed depth values
        """

        background_matrix = background_image if background_image is not None else 0 
        summed_depths = []
        for file in self.files : 
            image_matrix = self.get_image_matrix(file, apply_zoom)

            image_matrix = self.subtract_background( image_matrix, background_matrix )

            summed_depths.append( np.sum(image_matrix[image_matrix > threshold]) )

        if(normed) : summed_depths /= max(summed_depths)
        return np.array( summed_depths )
    
    def get_saturation(self, file, threshold=0, background_image=None, maxValue=None, apply_zoom=False) : 
        """ 
        Computes the degree of saturation of the image <file> by comparing the size of entries with the value <maxValue> 
        or the maximum bit-depth         and comparing it to all entries above <threshold>.
        file            :       TIFF-image to analyze
        threshold:      :       pixel threshold to calculate size
        background_image:       image subtracted from <file> to account for background
        maxValue        :       maximum bit-depth value to use as measure of saturation
        """
        background_matrix = background_image if background_image else 0
        matrix = self.get_image_matrix(file, apply_zoom=apply_zoom) - background_matrix
        if(maxValue is None) : maxValue = np.max(matrix)
        countmax = np.sum( matrix == np.max(matrix) )
        countgreater = np.sum( matrix > threshold )
        return countmax/countgreater

    def get_pixel_areas(self, threshold=0, normed=False, background_image=None) : 
        """ 
        Computes the total pixel area (with intensities above provided threshold) of all <self.files>.
        threshold       :       minimum pixel threshold  
        """
        background_matrix = self.get_image_matrix(background_image) if background_image else 0 
        areas = []
        for file in self.files : 
            matrix = self.subtract_background(self.get_image_matrix(file), background_matrix)
            areas.append( np.sum(matrix > threshold) )

        if(normed) : areas /= max(areas)
        return np.array( areas )
    
    def get_data_alongx(self, filename, x) : 
        matrix = self.get_image_matrix(filename)
        return matrix[x]
    
    def get_data_alongy(self, filename, y) : 
        matrix = self.get_image_matrix(filename)
        return matrix[:, y]

    def doSummedGaussianFitting(self, image_matrix, axis, **kwargs) : 
        p0 = kwargs.get("p0", None)
        absolute_sigma = kwargs.get("absolute_sigma", None)
        sigma = kwargs.get("sigma", None)
        bounds = kwargs.get("bounds", ([0,0,0, -np.inf], [np.inf, np.inf, np.inf, np.inf]))  # Ensure A, sigma, mu is within [0, infty].

        data = np.sum(image_matrix, axis=axis)/(np.shape(image_matrix)[axis])
        if p0 is None : p0 = [ np.max(data), len(data)/2, len(data)/2, np.mean(data) ]

        popt, pcov = curve_fit(self._gaussian, np.arange(0, len(data)), data, p0=p0, absolute_sigma=absolute_sigma, sigma=sigma, bounds=bounds)

        return (popt, pcov, data)
    
    def zoom_matrix(self, image):
        shape = np.shape(image)
        center_x, center_y = shape[0] // 2, shape[1] // 2
        start_x = max(center_x - (self.image_size - self.y_offset), 0)
        end_x = min(center_x + (self.image_size + self.y_offset), shape[0])
        start_y = max(center_y - (self.image_size + self.x_offset), 0)
        end_y = min(center_y + (self.image_size - self.x_offset), shape[1])
        
        return image[start_x:end_x, start_y:end_y]
