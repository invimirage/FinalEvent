import scrapy
import pandas as pd
import os
import sys
sys.path.append(r'P:\PycharmProjects\FinalEvent\FinalEvent')
import subprocess
import config
from scrapy.cmdline import execute
import time
import cv2

# 还没想好要爬什么，现在只需要首帧
class VideoFetcher(scrapy.Spider):
    name = "VideoFetcher"
    allowed_domains = ["adwetec.com"]
    data_folder = config.data_folder
    video_sources = pd.read_csv(
        config.raw_data_file,
        encoding="utf-8",
    )
    video_folder = config.video_folder
    video_list = list(filter(lambda x: x[-4:] == ".mp4", os.listdir(video_folder)))
    ids_already = list(map(lambda x: x.split(".")[0], video_list))
    video_dict = {}
    urls = []
    for id, url in zip(video_sources["id"], video_sources["file"]):
        if str(id) not in ids_already:
            target_url = config.video_url_prefix + url
            video_dict[target_url] = str(id)
            urls.append(target_url)
    print("%d videos waiting to download" % len(urls))
    start_urls = urls[::50]
    # img_dict = {k: str(v) + '.jpeg' for k, v in zip(start_urls, img_sources["id"])}

    def get_vedio_height_width(self, filename):
        cap = cv2.VideoCapture(filename)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width < height:
            width = 180
            height = 320
        else:
            width = 320
            height = 180
        return width, height

    def parse(self, response):
        video_name = self.video_dict[str(response.url)]
        video_local_path = os.path.join(self.video_folder, video_name + '_origin.mp4')
        video_target_path = os.path.join(self.video_folder, video_name + '.mp4')
        with open(video_local_path, "wb") as f:
            f.write(response.body)
        width, height = self.get_vedio_height_width(video_local_path)
        trans = r'ffmpeg -y  -i "%s" -r 5 -vf scale=%d:%d -map 0:0 "%s"' % (video_local_path, width, height, video_target_path)
        pipeline = subprocess.Popen(
            trans, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            outs, errs = pipeline.communicate(timeout=15)
            if pipeline.returncode != 0:
                print(errs.decode("utf-8"))
            else:
                os.remove(video_local_path)
        except subprocess.TimeoutExpired:
            pipeline.kill()
            outs, errs = pipeline.communicate()
            print(errs.decode("utf-8"))
        # out = pipeline.stdout.read()
        # temp = str(out.decode("utf-8"))
        # print(temp)


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    execute(["scrapy", "crawl", "VideoFetcher"])