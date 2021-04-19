#!/usr/bin/env python
# encoding: utf-8
"""
@author: caopeng
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software:XXXX
@file: get_audios.py
@time: 2021/3/15 16:34
@desc:
"""
import subprocess
import soundfile
import os


def get_audio(video_path, audio_path):
    trans = r'ffmpeg -i "%s" -ac 1 -ar 16000 -y -vn "%s"' % (video_path, audio_path)
    result = subprocess.Popen(
        trans, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out = result.stdout.read()
    temp = str(out.decode("utf-8"))
    print(temp)
    data_sf, sample_rate = soundfile.read(audio_path)
    print("sound sample:%d" % sample_rate)


if __name__ == "__main__":
    videos_dir = os.listdir("../Data/videos")
    for video_dir in videos_dir:
        if video_dir[-3:] == "mp4":
            get_audio(
                "../Data/videos/%s" % video_dir,
                "../Data/audios/%s.wav" % video_dir[:-4],
            )
