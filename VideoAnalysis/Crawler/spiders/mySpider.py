import scrapy
import pandas as pd
import os, sys
sys.path.append(r'P:\PycharmProjects\FinalEvent\FinalEvent')
import config

# 还没想好要爬什么，现在只需要首帧
class DataFetcher(scrapy.Spider):
    name = "mySpider"
    allowed_domains = ["adwetec.com"]
    data_folder = "../Data/"
    img_sources = pd.read_csv(
        config.raw_data_file,
        encoding="utf-8",
    )
    image_folder = config.img_folder
    image_list = list(filter(lambda x: x[-4:] == "jpeg", os.listdir(image_folder)))
    ids_already = list(map(lambda x: x.split(".")[0], image_list))
    img_dict = {}
    urls = []
    for id, url in zip(img_sources["id"], img_sources["first_frame"]):
        if str(id) not in ids_already:
            target_url = "https://constrain.adwetec.com/material/creative/image/" + url
            img_dict[target_url] = str(id) + ".jpeg"
            urls.append(target_url)
    print("%d images waiting to download" % len(urls))
    start_urls = urls
    # img_dict = {k: str(v) + '.jpeg' for k, v in zip(start_urls, img_sources["id"])}

    def parse(self, response):
        img_local_path = self.image_folder + self.img_dict[str(response.url)]

        with open(img_local_path, "wb") as f:
            f.write(response.body)
