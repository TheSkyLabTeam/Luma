from image_services import get_image_urls
from skimage import io
import solar_image_analysis as sia
import re 
from datetime import datetime
import pandas as pd
from urllib.error import HTTPError
import sys

image_type=sys.argv[1]
start_date0=pd.to_datetime(sys.argv[2])

output_name="output_"+sys.argv[1]+"_"+str("{:04d}".format(start_date0.year))+".csv"
def compute_metrics(image_type:str, year:str = None, start:str = None, end:str = None,  frequency:str=  None ) -> pd.DataFrame:
    
    """
    Computes various image metrics for a set of images within a given date range and specified type and frequency.

    Parameters
    ----------
    start_date : str
        Start date for retrieving the images, formatted as 'YYYY-MM-DD'.
    
    end_date : str
        End date for retrieving the images, formatted as 'YYYY-MM-DD'.
    
    image_type : str
        Type/category of the image to retrieve. 
    
    frequency : str
        Frequency for retrieving images, e.g., 'daily', 'hourly'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing computed metrics for each image, with one row per image 
        and columns for each feature. The index of the DataFrame corresponds to the datetime 
        extracted from the image URL.
        
    Notes
    -----
    The function fetches image URLs based on the provided date range, type, and frequency. 
    For each URL, it reads the image and computes a set of metrics, such as entropy, mean 
    intensity, and fractal dimension. If the image at a particular URL is inaccessible, it 
    skips that URL. The extracted date and time from the image URL is used as the index 
    for the resulting DataFrame.
    """

    image_urls = get_image_urls(image_type, year, start, end,  frequency )
    
    #Create an empty DataFrame to store the results
    df = pd.DataFrame(columns=['entropy', 'mean_intensity', 'std_deviation',
    'fractal_dimension', 'skewness', 'kurtosis', 'uniformity', 'relative_smoothness', 
    'taruma_contrast', 'taruma_directionality', 'taruma_coarseness',
    'taruma_linelikeness', 'taruma_roughness', 'taruma_regularity'])

    for url in image_urls[:10]:
        try:
            # Try to read the image
            image = io.imread(url)

        except HTTPError:
            #print(f"Unable to access image at {url}")
            continue

        except:
            pass
            
        # Compute the features
        entropy = sia.compute_entropy(image)
        mean_intensity = sia.compute_mean_intensity(image)
        std_deviation = sia.compute_standard_deviation(image)
        fractal_dimension = sia.compute_fractal_dimension(image)
        skewness = sia.compute_skewness(image)
        kurtosis = sia.compute_kurtosis(image)
        uniformity = sia.compute_uniformity(image)
        relative_smoothness = sia.compute_relative_smoothness(image)
        taruma_contrast = sia.compute_taruma_contrast(image)
        taruma_directionality = sia.compute_taruma_directionality(image)
        taruma_coarseness = sia.compute_taruma_coarseness(image)
        taruma_linelikeness = sia.compute_taruma_linelikeness(image)
        taruma_regularity = sia.compute_taruma_regularities(image)
        taruma_roughness = sia.compute_taruma_roughness(image)

        # Extract the date from the URL
        date_string = re.findall(r"\d{8}_\d{4}", url)[0]
        date = datetime.strptime(date_string, '%Y%m%d_%H%M')

        # Append the results to the DataFrame
        df.loc[date] = [entropy, mean_intensity, std_deviation, fractal_dimension, skewness, kurtosis, uniformity, relative_smoothness,taruma_contrast, taruma_directionality, taruma_coarseness, taruma_linelikeness, taruma_roughness, taruma_regularity]


    return df

import numpy as np
if __name__ == '__main__':
    #data = compute_metrics('01/01/2023 00:00:00', '01/01/2023 05:00:00','hmiigr', '90T')

    data = compute_metrics( image_type =sys.argv[1] , year= sys.argv[2])
    #print(data)
    data.to_csv(output_name,index=True)
    print("Save data type {} from year {}:".format(str(sys.argv[1]),str("{:04d}".format(start_date0.year))),output_name)
    
