import scrapy
import pandas as pd

# 还没想好要爬什么，现在只需要首帧
class DataFetcher(scrapy.Spider):
    name = "mySpider"
    allowed_domains = ["adwetec.com"]
    data_folder = "../Data/"
    img_sources = pd.read_csv(
       data_folder + "kuaishou_data_0421.csv",
        encoding="utf-8",
    )
    image_folder = data_folder + 'images/'
    urls = img_sources["first_frame"]
    start_urls = [
        "https://constrain.adwetec.com/material/creative/image/" + url
        for url in urls
    ]
    img_dict = {k: str(v) + '.jpeg' for k, v in zip(start_urls, img_sources["id"])}

    def parse(self, response):
        img_local_path = self.image_folder + self.img_dict[str(response.url)]

        with open(img_local_path, "wb") as f:
            f.write(response.body)
