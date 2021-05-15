#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: basic.py
@time: 2021/4/2 15:17
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
import pandas as pd
import numpy as np
import logging
from elasticsearch import Elasticsearch
import json
import config
import pymysql
import os
import re
from matplotlib import pyplot as plt
from utils import *


class DataFetcher:
    def __init__(self, data_path="Data/Data", **kwargs):
        self.logger = init_logger(kwargs["log_level"], name="DataFetcher")
        self.data_folder = config.data_folder
        self.logger.debug("folder : %s" % self.data_folder)
        if kwargs["from_db"]:
            data = self.get_data_from_db()
            data.to_csv(data_path, index=False)
        else:
            data = pd.read_csv(data_path)
        # print(data.columns)
        # data.drop(axis=1, columns=redundant_cols)
        self.logger.info("Data length: %d" % len(data))
        self.logger.debug("Data containing columns: {}".format(data.columns))
        # data.dropna(axis=0, subset=['file', 'first_frame', 'audio_text'], inplace=True)
        # logger.info('Data length after dropping nan: %d' % len(data))

        rows_to_delete = []
        audio_texts = []
        for index, row in data.iterrows():
            audio_text = self.parse_speech_text(row["audio_text"])
            if audio_text == -1:
                rows_to_delete.append(index)
            else:
                audio_texts.append(audio_text)

        data.drop(index=rows_to_delete, inplace=True)
        data["full_texts"] = audio_texts

        self.logger.info(
            "%d rows dropped due to speech recognition error" % len(rows_to_delete)
        )

        baidu_data = data[data["baidu_feed_upload"] > 0]
        kuaishou_data = data[data["kuaishou_feed_upload"] > 0]
        toutiao_data = data[data["toutiao_feed_upload"] > 0]
        tencent_data = data[data["tencent_feed_upload"] > 0]
        self.logger.info(
            "Baidu data size: %d\nKuaishou data size: %d\nToutiao data size: %d\nTencent data size: %d"
            % (
                len(baidu_data),
                len(kuaishou_data),
                len(toutiao_data),
                len(tencent_data),
            )
        )
        self.logger.info(
            "Total data size: %d"
            % sum(
                [
                    len(baidu_data),
                    len(kuaishou_data),
                    len(toutiao_data),
                    len(tencent_data),
                ]
            )
        )

        if kwargs["analyze"]:
            self.get_data_status(baidu_data, "baidu")
            self.get_data_status(kuaishou_data, "kuaishou")
            self.get_data_status(toutiao_data, "toutiao")
            self.get_data_status(tencent_data, "tencent")

        self.data = kuaishou_data.copy()

        if kwargs["es"]:
            # 目前只搞快手
            self.get_es_data(kuaishou_data["id"])

        self.data.to_csv(
            os.path.join(self.data_folder, kwargs["target_file_name"]), index=False
        )

    def get_data_from_db(self):
        # 打开数据库连接
        db = pymysql.connect(
            host="192.168.0.18",
            user="zhangruifeng",
            passwd="t0dG18PjAJ8c8EgR",
            database="adwetec_prod",
        )
        self.logger.info("Successfully connected to database")
        # 使用 cursor() 方法创建一个游标对象 cursor
        cursor = db.cursor()

        # 使用 execute()  方法执行 SQL 查询
        sql = "select {} from adwetec_material_upload \
        where audio_text != '' and !ISNULL(audio_text)".format(
            ",".join(config.include_cols)
        )
        cursor.execute(sql)

        # 使用 fetchone() 方法获取单条数据.
        data = cursor.fetchall()

        df = pd.DataFrame(data, columns=config.include_cols)

        # 关闭数据库连接
        db.close()
        return df

    def parse_speech_text(self, json_string):
        try:
            data = json.loads(json_string)["Response"]["Data"]["ResultDetail"]
            final_sentence = ""
            for sentence in data:
                final_sentence += sentence["FinalSentence"]
            return final_sentence
        except:
            return -1

    def get_data_status(self, data, kind):
        prefix = kind + "_feed_"
        imp = np.array(data[prefix + "imp"])
        clk = np.array(data[prefix + "clk"])
        cost = np.array(data[prefix + "cost"])
        # imp, clk, cost = parse_nan([imp, clk, cost])
        ctr = clk / (imp + 1)
        cpi = cost / (clk + 1)
        keys_targets = {"imp": imp, "clk": clk, "cost": cost, "ctr": ctr, "cpi": cpi}
        self.logger.info("Kind: %s" % kind)
        for key, val in keys_targets.items():
            self.logger.info(
                "%s avg %lf max %lf min %lf"
                % (key, val.mean(), np.max(val), np.min(val))
            )

    # 从es中获取额外数据
    def get_es_data(self, ids):
        matids = list(ids)
        batch_size = 10000
        n_batch = math.ceil(len(matids) / batch_size)
        # 连接ES
        es = Elasticsearch(
            [
                "http://192.168.0.4:9200/",
                "http://192.168.0.28:9200/",
                "http://192.168.0.29:9200/",
                "http://192.168.0.44:9200/",
            ],
            timeout=3600,
        )
        agg_name = "sale_data"
        # 选择要查询的数据
        agg_fields = {
            "bclk": "bclk",
            "pclk": "pclk",
            "cost": "cost",
            "clk": "clk",
            "imp": "imp",
            "share": "share",
            "comment": "comment",
            "like": "like",
            "follow": "follow",
            "cancelfollow": "cancelfollow",
            "report": "report",
            "block": "block",
            "negative": "negative",
            "paly3s": "play3s",
            "upload_time": "vtime",
        }
        aggs = {}
        for agg_field, target in agg_fields.items():
            if agg_field != "upload_time":
                aggs[agg_field] = {"sum": {"field": target}}
            else:
                aggs["upload_time"] = {"min": {"field": "vtime"}}
        es_dataframe = []
        for i in range(n_batch):
            sta = i * batch_size
            end = (i + 1) * batch_size
            matids_this_batch = matids[sta:end]
            query = {
                "size": 1,
                "_source": {"includes": ["matid", "date", "vtime"]},
                "aggs": {
                    agg_name: {
                        "terms": {
                            "size": len(matids),
                            "script": "doc['matid'].value +','+doc['advid'].value",
                            "order": {"cost": "desc"},
                        },
                        "aggs": aggs,
                    }
                },
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"matid": matids_this_batch}},
                            {"term": {"medid": 8}},
                            {
                                "script": {
                                    "script": {
                                        "source": "doc['date'].value.toInstant().toEpochMilli() - "
                                        "doc['vtime'].value.toInstant().toEpochMilli() <= params.aMonth",
                                        "params": {
                                            "aMonth": 2592000000,
                                            "aWeek": 604800000,
                                        },
                                    }
                                }
                            },
                        ],
                        "filter": {"range": {"vtime": {"lte": "now-30d/d"}}},
                    }
                },
                "sort": [{"date": {"order": "desc"}}],
            }
            self.logger.info("Collecting es data, batch %d/%d" % (i, n_batch))
            self.logger.debug(query)
            result = es.search(index="creative-report-*", body=query)
            self.logger.debug(result["_shards"])
            # self.logger.debug(result["hits"]["hits"][0])
            sale_data = result["aggregations"][agg_name]["buckets"]
            self.logger.debug(
                "Bucket length: %d, id length %d" % (len(sale_data), len(matids))
            )
            es_data = []
            upload_time = []
            for dta in sale_data:
                id, advid = dta["key"].split(",")
                this_data = [id, advid]
                for field in agg_fields:
                    if field != "upload_time":
                        this_data.append(dta[field]["value"])
                    else:
                        upload_time.append(dta[field]["value_as_string"].split('T')[0])
                es_data.append(this_data)

            # 包含没有es数据和es数据量太少的
            es_data = np.array(es_data, dtype=int)
            es_dataframe_this_batch = pd.DataFrame(
                data=es_data, columns=["id", "advid"] + [key for key in agg_fields.keys() if key != "upload_time"]
            )
            es_dataframe_this_batch["upload_time"] = upload_time
            es_dataframe.append(es_dataframe_this_batch)
        es_dataframe = pd.concat(es_dataframe, axis=0, ignore_index=True)
        db_dataframe = self.data.set_index("id")
        # print(es_dataframe.head(), db_dataframe.head())
        merged_data = es_dataframe.join(db_dataframe, on="id", lsuffix="_db")
        self.logger.info(
            "Data length before merging: %d\nData length after merging: %d"
            % (len(db_dataframe), len(merged_data))
        )
        self.data = merged_data
        # print(self.data.index)
        # for col_num, id in enumerate(matids):
        #     for field in agg_fields:
        #         try:
        #             new_cols[field].append(id_dict[str(id)][field])
        #         except KeyError:
        #             no_data_cols.append(col_num)
        #             break
        # self.logger.info("%d cols dropped due to lack of es data" % len(no_data_cols))
        # self.data.drop(index=self.data.index[no_data_cols], inplace=True)
        # for field in agg_fields:
        #     self.data[field] = new_cols[field]


if __name__ == "__main__":
    data_folder = config.data_folder
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # 一般来说，这个文件不需要存储两次，其区别仅仅是是否包含了es中的数据
    data_path = os.path.join(data_folder, config.raw_data_file)
    DataFetcher(
        data_path=data_path,
        target_file_name=config.raw_data_file,
        es=True,
        analyze=False,
        from_db=False,
        log_level=logging.INFO,
    )

# type WetecMaterialDailyReport struct {
# 	Id           string    `json:"-"`
# 	Medid        int64     `json:"medid"`
# 	Matid        int64     `json:"matid"`
# 	Vmatids      []string  `json:"vmatids"`
# 	Date         time.Time `json:"date"`
# 	Imp          int32     `json:"imp"`          // 快手 封面曝光数: show
# 	Clk          int32     `json:"clk"`          // 快手 素材曝光数: aclick
# 	Cost         float64   `json:"cost"`         // 快手 花费(元): charge
# 	Bclick       int32     `json:"bclk"`         // 快手 行为数据
# 	Pclick       int32     `json:"pclk"`         // 快手 封面点击数
# 	Inv          []string  `json:"inv"`          // 头条 广告位置
# 	Totalplay    int32     `json:"tplay"`        // 头条
# 	Validplay    int32     `json:"vplay"`        // 头条
# 	Share        int32     `json:"share"`        // 快手 分享数
# 	Comment      int32     `json:"comment"`      // 快手 评论数
# 	Like         int32     `json:"like"`         // 快手 点赞数
# 	Follow       int32     `json:"follow"`       // 快手 新增关注数
# 	Cancelfollow int32     `json:"cancelfollow"` // 快手 取消关注数
# 	Report       int32     `json:"report"`       // 快手 举报数
# 	Block        int32     `json:"block"`        // 快手 拉黑数
# 	Negative     int32     `json:"negative"`     // 快手 减少此类作品数
# 	Play3scount  int32     `json:"play3s"`       // 快手 3秒播放数
# }
#
# type UploadInfo struct {
# 	CreateId   int64     // 视频或图片上传者ID
# 	RegionId   int64     // 地域ID
# 	DirectId   int64     // 编导ID
# 	ShootId    int64     // 摄影ID
# 	ScriptId   int64     // 脚本ID
# 	CreateTime time.Time // 上传时间
# 	Sharedids  []int64   // 分享用户
# }
# type MaterialInfo struct {
# 	Materialid int64
# 	Source     int32
# }
#
# //	PUT /creative-report-2019-*/_mapping
# //	{
# //		"properties": {
# //			"source": {
# //				"type": "keyword"
# //			}
# //		}
# //	}
#
# type WetecCreativeDailyReport struct { // KEYWORD：ES使用的是倒排索引(NUMERIC类型为了能有效的支持范围查询，它的存储结构并不是倒排索引。)
# 	Id       string    `json:"-"`
# 	Name     string    `json:"name" type:"text,keyword"`              //
# 	Advid    int64     `json:"advid" type:"keyword"`                  //
# 	Medid    int64     `json:"medid" type:"keyword"`                  //
# 	Acctp    int32     `json:"acctp" type:"keyword"`                  //
# 	Frmtp    int32     `json:"frmtp" type:"keyword"`                  //
# 	Cretp    int32     `json:"cretp" type:"keyword"`                  //
# 	Accid    int64     `json:"accid" type:"keyword"`                  //
# 	Cmpid    int64     `json:"cmpid" type:"keyword"`                  //
# 	Adgid    int64     `json:"adgid" type:"keyword"`                  //
# 	Creid    int64     `json:"creid" type:"keyword"`                  //
# 	Matid    int64     `json:"matid" type:"keyword"`                  // 致维ID
# 	Source   int32     `json:"source" type:"keyword"`                 //
# 	Vusrid   int64     `json:"vusrid" type:"keyword"`                 // 视频上传用户
# 	Vregid   int64     `json:"vregid" type:"keyword"`                 // 视频上传用户地域
# 	Directid int64     `json:"directid" type:"keyword"`               //
# 	Shootid  int64     `json:"shootid" type:"keyword"`                //
# 	Scriptid int64     `json:"scriptid" type:"keyword"`               //
# 	Shareids []int64   `json:"sharedids" type:"keyword"`              //
# 	Pusrid   int64     `json:"pusrid" type:"keyword"`                 // 图片上传用户
# 	Vmatids  []string  `json:"vmatids" type:"text,keyword,fielddata"` //
# 	Pmatids  []string  `json:"pmatids" type:"text,keyword,fielddata"` //
# 	Vtime    time.Time `json:"vtime" type:"date"`                     // 视频上传时间
# 	Ptime    time.Time `json:"ptime" type:"date"`                     // 图片上传时间
# 	Date     time.Time `json:"date"  type:"date"`                     //
# 	Inv      []string  `json:"inv" type:"text,keyword,fielddata"`     // 头条 广告位置
#
# 	Imp  int32   `json:"imp" type:"integer"` // 快手 封面曝光数: show
# 	Clk  int32   `json:"clk" type:"integer"` // 快手 素材曝光数: aclick
# 	Cost float64 `json:"cost" type:"float"`  // 快手 花费(元): charge
#
# 	Bclick int32 `json:"bclk" type:"integer"` // 快手 行为数据
# 	Pclick int32 `json:"pclk" type:"integer"` // 快手 封面点击数
#
# 	Totalplay int32 `json:"tplay" type:"integer"` // 头条
# 	Validplay int32 `json:"vplay" type:"integer"` // 头条
#
# 	Share        int32   `json:"share" type:"integer"`        // 快手 分享数
# 	Comment      int32   `json:"comment" type:"integer"`      // 快手 评论数
# 	Like         int32   `json:"like" type:"integer"`         // 快手 点赞数
# 	Follow       int32   `json:"follow" type:"integer"`       // 快手 新增关注数
# 	Cancelfollow int32   `json:"cancelfollow" type:"integer"` // 快手 取消关注数
# 	Report       int32   `json:"report" type:"integer"`       // 快手 举报数
# 	Block        int32   `json:"block" type:"integer"`        // 快手 拉黑数
# 	Negative     int32   `json:"negative" type:"integer"`     // 快手 减少此类作品数
# 	Play3scont   int32   `json:"play3s" type:"integer"`       // 快手 3秒播放数
# 	DownStart    int32   `json:"downstart" type:"integer"`    // 快手
# 	DownComplte  int32   `json:"downcomplte" type:"integer"`  // 快手
# 	Nstay        int32   `json:"nstay" type:"integer"`        // 快手 次留存
# 	Nstaycost    float64 `json:"nstaycost" type:"float"`      // 快手 次留成本
#
# 	Activation int32 `json:"activation" type:"integer"` // 百度快手
#
# 	PlayCount0 int32 `json:"playcount0" type:"integer"` // 百度 播放完成数
# 	PlayCount1 int32 `json:"playcount1" type:"integer"` // 百度 25%进度播放数
# 	PlayCount2 int32 `json:"playcount2" type:"integer"` // 百度 50%进度播放数
# 	PlayCount3 int32 `json:"playcount3" type:"integer"` // 百度 75%进度播放数
# 	PlayCount4 int32 `json:"playcount4" type:"integer"` // 百度 100%进度播放数
#
# 	Dcount int32 `json:"dcount" type:"integer"` // 腾讯 APP下载完成量
# 	Acount int32 `json:"acount" type:"integer"` // 腾讯 APP激活总量
# 	Kcount int32 `json:"kcount" type:"integer"` // 腾讯 关键页面访问成本
# 	Ccount int32 `json:"ccount" type:"integer"` // 腾讯 目标转化量
# }
