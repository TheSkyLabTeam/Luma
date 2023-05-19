from typing import List
import pandas as pd

def get_image_urls(start: str, end: str, image_type: str = 'hmiigr', frequency: str = '90T') -> List[str]:
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