import scrapy
import pandas as pd

# 还没想好要爬什么，现在只需要首帧and封面
class DataFetcher(scrapy.Spider):
    name = "mySpider"
    allowed_domains = ["adwetec.com"]
    video_sources = pd.read_csv(
        r"C:\Users\zrf19\PycharmProjects\pythonProject\FinalEvent\RawDataURL\test.csv",
        encoding="utf-8",
    )
    video_folder = r"C:\Users\zrf19\PycharmProjects\pythonProject\FinalEvent\videos/"
    print(video_sources.head(10))
    urls = video_sources["file"]
    start_urls = [
        "https://constrain.adwetec.com/material/creative/video/" + url
        for url in urls[0:10]
    ]
    video_dict = {k: v for v, k in zip(video_sources["name"], start_urls)}

    def parse(self, response):
        video_local_path = self.video_folder + self.video_dict[str(response.url)]

        with open(video_local_path, "wb") as f:
            f.write(response.body)
