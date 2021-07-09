from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys

# API KeyとSecret Key
key = "e2285a17d43facdec501606eb1434e21"
secret = "660af1b75058b9db"

# wait for 1.1s due to API limitations.
wait_time = 1

# search keyword
keyword = sys.argv[1]
savedir = "./" + keyword
os.makedirs(savedir, exist_ok=True)

flickr = FlickrAPI(key, secret, format='parsed-json')
count_page = 0

for i in range(20):
    count_page += 1

    result = flickr.photos.search(
        text = keyword,
        per_page = 250,
        media = 'photos',
        sort = 'relevance',
        # set limitations here to obtain more results.
        # min_upload_date = '1283228800',
        min_upload_data = '1262304000',

        safe_search = 1,
        extras = 'url_l, license',
        page = count_page,
        geo_context = 2,
    )

    # get reusult and save it.
    photos = result['photos']
    count = 0
    for i, photo in enumerate(photos['photo']):
        try:
            url = photo['url_l']
        except KeyError:
            continue
        count += 1
        print(count)
        filepath = savedir + '/' + photo['id'] + '.jpg'
        #if os.path.exists(filepath): continue
        urlretrieve(url, filepath)
        time.sleep(wait_time)
