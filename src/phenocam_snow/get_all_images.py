"""Script to download all the image URLs for a given site.
The results are written to a .txt file.
Example usage:
    python get_all_images.py delnortecounty2
Returns URLs in a file called delnortecounty2_urls.txt
"""

# Standard library
import requests
import random


def output_image_paths(site_name):

    # api_url uses the api url to obtain a list of middayimages for every available date for the site
    # root_url is used for construction of the full path of the jpg image to be downloaded
    api_url = f"https://phenocam.nau.edu/api/middayimages/{site_name}/"
    root_url = "https://phenocam.nau.edu"

    try:
        # For the new api logic, request should get a single response with all the image paths available for the site
        resp = requests.get(
            api_url, headers={"accept": "application/json"},
            timeout=5
        )

        # If request is successful
        if resp.status_code == 200:
            # Parse the JSON response
            data = resp.json()

            # Ensure the data is valid and not empty
            if isinstance(data, list) and data:
                # Extract imgpath values from each dictionary
                image_paths = [entry['imgpath'] for entry in data if 'imgpath' in entry]

                if image_paths:
                    # Save the image paths to a file
                    output_file = f"{site_name}_urls.txt"
                    with open(output_file, "w") as file:
                        for path in image_paths:
                            file.write(root_url + path + "\n")
                    print(f"get_all_images.py: Midday image paths saved to {output_file}")
                else:
                    print("get_all_images.py: No image paths found in the data.")
            else:
                print("get_all_images.py: No valid data found in the response.")
        else:
            print(f"get_all_images.py: Failed to retrieve data. HTTP Status Code: {resp.status_code}")

    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    return output_file

def read_image_paths(output_file):
    # Read the image paths from the file into a list
    image_paths_list = []
    try:
        with open(output_file, "r") as file:
            # Read each line (image path) and strip any leading/trailing whitespace
            image_paths_list = [line.strip() for line in file]
        print(f"Read {len(image_paths_list)} image paths from {output_file}")
    except FileNotFoundError:
        print(f"The file {output_file} was not found.")
    
    return image_paths_list

def select_random_photos(image_paths_list, n_photos):
    # Ensure that n_photos does not exceed the length of image_paths_list
    if n_photos > len(image_paths_list):
        print(f"Warning: Requested {n_photos} photos, but only {len(image_paths_list)} available. Selecting all images.")
        n_photos = len(image_paths_list)

    # Use random.sample to select n_photos unique random items from the list
    random_photos = random.sample(image_paths_list, n_photos)
    return random_photos