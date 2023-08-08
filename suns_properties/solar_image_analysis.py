from image_services import calculate_probabilities
from scipy.signal import convolve2d, find_peaks
import matplotlib.pyplot as plt
import numpy as np
import cv2


def compute_entropy(image: np.ndarray) -> float:
    """
    Compute the entropy of an image.
    
    The entropy is a statistical measure of randomness that 
    can be used to characterize the texture of an input image.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    entropy : float
        Calculated entropy.
        
    """
    # Calculate the probabilities
    probabilities = calculate_probabilities(image)
    
    # Exclude zero probabilities
    probabilities = probabilities[probabilities != 0]
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def compute_mean_intensity(image: np.ndarray) -> float:
    """
    Compute the mean intensity of an image.
    
    The mean intensity is a measure that 
    represents the average level of brightness of the image.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    mean_intensity : float
        Calculated mean intensity.
        
    """
    # Calculate the probabilities
    probabilities = calculate_probabilities(image)
    
    # Compute the pixel intensities
    intensities = np.arange(256)
    
    # Compute mean intensity
    mean_intensity = np.sum(intensities * probabilities)
    
    return mean_intensity

def compute_standard_deviation(image: np.ndarray) -> float:
    """
    Compute the standard deviation of the pixel intensity of an image.
    
    The standard deviation is a measure that 
    represents the dispersion of the pixel intensities in the image.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    standard_deviation : float
        Calculated standard deviation of pixel intensity.
        
    """
    # Calculate the probabilities
    probabilities = calculate_probabilities(image)
    
    # Compute the pixel intensities
    intensities = np.arange(256)
    
    # Compute mean intensity
    mean_intensity = compute_mean_intensity(image)
    
    # Compute standard deviation
    standard_deviation = np.sqrt(np.sum(probabilities * (intensities - mean_intensity)**2))
    
    return standard_deviation

def compute_fractal_dimension(image: np.ndarray, threshold: float=0.5) -> float:
    """
    Computes the fractal dimension of a 2D grayscale image using the box-counting method.
    
    Parameters
    ----------
    image : np.ndarray
        2D input image data. Will be converted to grayscale if not already.
    threshold : float, optional
        Threshold to convert the grayscale image to binary, by default 0.5.

    Returns
    -------
    fractal_dimension : float
        Calculated fractal dimension of the image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert(len(image.shape) == 2)# "image must be 2D"
    

    # Binarize the image using the given threshold
    image = image < threshold
    
    # Define the boxcount function
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1)
        return np.count_nonzero((0 < S) & (S < k*k))

    # Define the box sizes
    p = min(image.shape)
    n = int(np.floor(np.log2(p)))
    sizes = 2**np.arange(n, 1, -1)

    # Count the number of boxes for each size
    counts = [boxcount(image, size) for size in sizes]

    # Perform a linear fit (y = ax + b) to the sizes and counts
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    # Return the fractal dimension (-a)
    return -coeffs[0]

def compute_skewness(image: np.ndarray) -> float:
    """
    Compute the skewness of an image.

    The skewness is a measure of the asymmetry of the probability distribution
    of a real-valued random variable about its mean. Skewness can be positive or negative,
    or undefined, and it quantifies the extent and direction of skew (departure from horizontal symmetry).
    
    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    skewness : float
        Calculated skewness of the image pixel intensity distribution.
        
    """
    # Calculate the probabilities
    probabilities = calculate_probabilities(image)
    
    # Compute the pixel intensities
    intensities = np.arange(256)

    # Compute mean intensity
    mean_intensity = compute_mean_intensity(image)

    # Compute standard deviation
    standard_deviation = compute_standard_deviation(image)

    # Compute skewness
    skewness = (1/standard_deviation**3)*(np.sum(probabilities * (intensities - mean_intensity)**3))

    return skewness

def compute_kurtosis(image: np.ndarray) -> float:
    """
    Compute the kurtosis of an image.

    The kurtosis is a measure of the "tailedness" of the probability distribution
    of a real-valued random variable. In particular, kurtosis quantifies the extent
    to which a distribution shows a peaked or flat shape.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    kurtosis : float
        Calculated kurtosis of the image pixel intensity distribution.
        
    """
    # Calculate the probabilities
    probabilities = calculate_probabilities(image)
    
    # Compute the pixel intensities
    intensities = np.arange(256)

    # Compute mean intensity
    mean_intensity = compute_mean_intensity(image)

    # Compute standard deviation
    standard_deviation = compute_standard_deviation(image)

    # Compute kurtosis
    kurtosis = (1/standard_deviation**4)*(np.sum(probabilities * (intensities - mean_intensity)**4)) 

    return kurtosis

def compute_uniformity(image: np.ndarray) -> float:
    """
    Compute the uniformity of an image.

    The uniformity is a measure of the texture of an image. It measures the sum of 
    the squares of the pixel intensities, normalized to be in the range [0, 1]. 
    A value close to 1 indicates an image with low variation in pixel intensities 
    (like a completely black or white image), while a value close to 0 indicates 
    high variation in pixel intensities.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    uniformity : float
        Calculated uniformity of the image pixel intensity distribution.
        
    """
    # Calculate the probabilities
    probabilities = calculate_probabilities(image)

    # Compute uniformity
    uniformity =  np.sum(probabilities**2)

    return uniformity

def compute_relative_smoothness(image: np.ndarray) -> float:
    """
    Compute the relative smoothness of an image.
    
    The relative smoothness is a measure of variation in the pixel intensity levels.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    relative_smoothness : float
        Calculated relative smoothness.
        
    """
    # Compute standard deviation
    sigma = compute_standard_deviation(image)

    # Compute relative smoothness
    relative_smoothness = 1 - (1 / (1 + sigma**2))

    return relative_smoothness

def compute_taruma_contrast(image: np.ndarray) -> float:
    """
    Compute the Taruma contrast of an image.
    
    The Taruma contrast is a measure derived from the image's standard deviation 
    and kurtosis, giving insight into its contrast characteristics.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Will be converted to float.

    Returns
    -------
    taruma_contrast : float
        Calculated Taruma contrast.
        
    """
    # Compute standard deviation
    sigma = compute_standard_deviation(image)

    # Compute kurtosis
    kurtosis = compute_kurtosis(image)

    # Compute Taruma contrast
    taruma_contrast = (sigma**2) / (kurtosis**(0.25))
    
    return taruma_contrast

def compute_taruma_directionality(image:np.ndarray, plot:bool = False)-> float:
    """
    Compute the Taruma directionality of a given image.

    This function employs gradient filters to compute the horizontal and 
    vertical gradient components of an image. Subsequently, it uses the magnitude
    and direction of these gradients to construct a directionality histogram.
    From this histogram, dominant direction peaks are identified. Finally, a 
    Taruma directionality value is computed based on these peaks.

    Parameters:
    -----------
    image : np.ndarray
        An image in ndarray format. Can be colored or grayscale.
        If colored, it will automatically be converted to grayscale.
    
    plot : bool, optional
        If set to True, a directionality histogram with highlighted dominant
        peaks will be displayed. Default is False.

    Returns:
    --------
    float
        A Taruma directionality value ranging between 0 and 1, where values
        closer to 1 indicate high directionality and values closer to 0 indicate 
        low directionality.

    Example:
    --------
    >>> image = np.array(io.imread('path_to_image.jpg'))
    >>> directionality_value = compute_taruma_directionality(image, plot=True)

    Notes:
    ------
    - The function uses convolutions to compute gradients, so performance 
      may vary based on the image size.
    - Ensure that the input image has an appropriate value range (e.g., between 0 and 255 
      for 8-bit images).

    """
    image = np.array(image, dtype='int64')
    image = np.mean(image, axis=-1) 

    h = image.shape[0]
    w = image.shape[1]

    # Kernels de convolución
    convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])  
    
    # Calcula componentes horizontales y verticales usando convolución
    deltaH = convolve2d(image, convH, mode='same', boundary='symm')
    deltaV = convolve2d(image, convV, mode='same', boundary='symm')

    # Calcula la magnitud de gradiente
    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0

    # Calcula el ángulo de dirección
    theta = np.arctan2(deltaV, deltaH) + np.pi / 2.0

    # Cuantización y histograma de dirección
    n = 90
    hist, edges = np.histogram(theta, bins=n, range=(0, np.pi), weights=deltaG)
    
    # Normalizar el histograma
    hist = hist / np.max(hist)

    # Calcular el umbral usando la media
    threshold = np.mean(hist)

    # Encuentra todos los picos que están por encima de la media
    all_peaks, properties = find_peaks(hist, height=threshold)

    # De esos picos, solo nos quedamos con los 5 más altos
    if len(all_peaks) > 5:
        sorted_peak_indices = np.argsort(properties['peak_heights'])[-5:]
        peaks = all_peaks[sorted_peak_indices]
        peak_properties = properties['peak_heights'][sorted_peak_indices]
    else:
        peaks = all_peaks
        peak_properties = properties['peak_heights']

    np_ = len(peaks)  # número de picos

    # Calcula F_dir según la formulación dada
    r = 1.0 / n  # factor de normalización
    phi = np.linspace(0, np.pi, n, endpoint=False) + np.pi / (2 * n)
    F_dir = 0
    for p in peaks:
        phi_p = phi[p]
        F_dir +=  np.sum((phi - phi_p) ** 2 * hist)

    if plot:
        # Visualización
        plt.bar(edges[:-1], hist, width=np.pi/n, align='center', alpha=0.75, label='Histograma')
        plt.scatter(edges[peaks], peak_properties, color='red', marker='x', label='Picos mayores')
        plt.xlabel('Ángulo (radianes)')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Direccionalidad')
        plt.xlim(0, np.pi)
        plt.xticks(np.arange(0, np.pi + 0.1, np.pi/4), ['0', 'π/4', 'π/2', '3π/4', 'π'])
        plt.legend()
        plt.show()

    return 1 - (r * np_ * F_dir)





