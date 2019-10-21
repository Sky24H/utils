from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys

# 「事前準備」で取得したAPI KeyとSecret Keyを設定
key = "e2285a17d43facdec501606eb1434e21"
secret = "660af1b75058b9db"

# 1秒間隔でデータを取得(サーバー側が逼迫するため)
wait_time = 1.1

# 検索キーワード(実行時にファイル名の後に指定)
keyword = sys.argv[1]
# 保存フォルダ
savedir = "./" + keyword + "/"
count = 45000
count_page = 91
# 接続クライアントの作成とサーチの実行
flickr = FlickrAPI(key, secret, format='parsed-json')
for i in range(11):
    result = flickr.photos.search(
        text = keyword,           # 検索キーワード
        per_page = 500,           # 取得データ数
        page = count_page,
        media = 'photos',         # 写真を集める
        sort = 'relevance',       # 最新のものから取得
        safe_search = 1,          # 暴力的な画像を避ける
        extras = 'url_n, license' # 余分に取得する情報(ダウンロード用のURL、ライセンス)
    )
    photos = result['photos']
    for i, photo in enumerate(photos['photo']):
        url_n = photo['url_n']
        filepath = savedir + "{:0>2d}".format(count_page) + photo['id'] + '.jpg'
        count += 1
        print("now in page "+"{:0>2d}".format(count_page)+", total counts for images :"+str(count)+"/50000")
        if os.path.exists(filepath):
            pass
        else:
            urlretrieve(url_n, filepath)
        time.sleep(wait_time)
    count_page += 1
