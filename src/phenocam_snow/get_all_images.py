"""Script to download all the image URLs for a given site.
The results are written to a .txt file.
Example usage:
    python get_all_images.py delnortecounty2
Returns URLs in a file called delnortecounty2_urls.txt
"""

# Standard library
from argparse import ArgumentParser
import re
import requests

# Third party
import pandas as pd
from tqdm import tqdm


# Parse arguments
parser = ArgumentParser()
parser.add_argument("site_name")
args = parser.parse_args()

# api_url uses the api url to obtain a list of middayimages for every available date for the site
# root_url is used for construction of the full path of the jpg image to be downloaded
api_url = f"https://phenocam.nau.edu/api/middayimages/{args.site_name}/"
root_url = "https://phenocam.nau.edu/"

try:
    ### Original code:

    # resp = requests.get(
    #     f"https://phenocam.nau.edu/webcam/browse/{args.site_name}/",
    #     timeout=5
    # )


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
                output_file = f"{args.site_name}_urls.txt"
                with open(output_file, "w") as file:
                    for path in image_paths:
                        file.write(root_url + path + "\n")
                print(f"Midday image paths saved to {output_file}")
            else:
                print("No image paths found in the data.")
        else:
            print("No valid data found in the response.")
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {resp.status_code}")

except requests.exceptions.RequestException as e:
    raise SystemExit(e)


### Original code:


# content = resp.content.decode()
# year_tags = re.findall(r"<a name=\"[0-9]{4}\">", content)
# years = [int(re.search(r"\d+", yt).group()) for yt in year_tags]
# dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31").strftime("%Y/%m/%d")

# # Loop through all dates
# root = "https://phenocam.nau.edu"
# pattern = re.compile(rf"\/data\/archive\/{args.site_name}\/[0-9]{{4}}\/[0-9]{{2}}\/{args.site_name}_[0-9]{{4}}_[0-9]{{2}}_[0-9]{{2}}_[0-9]{{6}}\.jpg")
# all_photos = []
# for d in tqdm(dates):
#     try:
#         resp = requests.get(
#             # f"https://phenocam.nau.edu/webcam/browse/delnortecounty2/{d}/",
#             f"https://phenocam.nau.edu/webcam/browse/{args.site_name}/{d}/",
#             timeout=5
#         )
#     except requests.exceptions.RequestException as e:
#         continue
#     if resp.ok:
#         content = resp.content.decode()
#         matches = pattern.finditer(content)
#         for m in matches:
#             all_photos.append(f"{root}{m.group()}")

# # Save to file
# with open(f"{args.site_name}_urls.txt", "w+") as f:
#     for url in all_photos:
#         print(url, file=f)