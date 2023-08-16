from typing import List
import pandas as pd
from bs4 import BeautifulSoup
import requests

def get_image_urls(image_type: str, year:str = None, start:str = None, end:str = None,  frequency:str=  None ) -> List[str]:
    """
    This function generates the URLs to download solar images from the SOHO NASCOM website for a given date range.

    Parameters:
    start (str): The start date in the format 'MM/DD/YYYY HH/MM/SS'.
    end (str): The end date in the format 'MM/DD/YYYY HH/MM/SS'.
    image_type (str): The type of the image. Default is 'hmiigr'.
    frequency (str): The frequency at which to generate dates in the range. Default is '90T' (every 90 minutes).

    Returns:
    List[str]: A list of URLs to download the solar images.
    """

    if year is None:
        # Generate dates in the range
        dates = pd.date_range(start=start, end=end, freq=frequency)

        # Base path for the images
        path="https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"

        urls = []  # List to store the URLs

        # Loop through each date
        for date in dates:
            # Format the date
            formatted_date = date.strftime('%Y%m%d_%H%M')

            # Construct the URL
            url = path+'{}/{}/{}/{}_{}_1024.jpg'.format(date.year,image_type,formatted_date[:8], formatted_date,image_type)

            # Append the URL to the list
            urls.append(url)
            
        return urls
    if year is not None:
        base_url = f'https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/{year}/{image_type}/'

        def get_links_from_url(url):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')
            return [link.get('href') for link in links]

        # Obtener carpetas
        folders = get_links_from_url(base_url)
        folders = [folder for folder in folders if folder.endswith('/')]

        image_links = []

        # Buscar imágenes en cada carpeta
        for folder in folders:
            folder_url = base_url + folder
            files = get_links_from_url(folder_url)

            # Filtrar imágenes con "1024" en el nombre
            images = [file for file in files if '1024' in file]

            # Construir las URLs completas de las imágenes y agregarlas a la lista
            for image in images:
                image_links.append(folder_url + image)

        return image_links



from typing import List
import numpy as np
import cv2

def calculate_probabilities(image: np.ndarray) -> np.ndarray:
    """
    Calculate the probabilities of occurrence of each intensity level in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR color space.

    Returns
    -------
    probabilities : np.ndarray
        Array of probabilities for each intensity level in the grayscale image.
    """
    # Convert the image to grayscale
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the histogram of the grayscale image
    histogram, _ = np.histogram(gray_image.ravel(), bins=256, range=[0,256])

    # Calculate probabilities
    probabilities = histogram / histogram.sum()

    return probabilities
