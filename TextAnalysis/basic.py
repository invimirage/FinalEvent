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


class DataParser:
    def __init__(
        self,
        data_path=r"C:\Users\zrf19\PycharmProjects\pythonProject\FinalEvent\RawDataURL\Data.csv",
        **kwargs
    ):
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
        )
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)
        self.data_folder = "\\".join(data_path.split("\\")[:-1]) + "\\"
        self.logger.debug("folder : %s" % self.data_folder)
        if kwargs["from_db"]:
            data = self.get_data_from_db()
            data.to_csv(data_path)
        else:
            data = pd.read_csv(data_path)
        # print(data.columns)
        # redundant_cols = [
        #     "cluster",
        #     "file_type",
        #     "status",
        #     "creator_id",
        #     "phash",
        #     "advids",
        #     "ffmpeg_info",
        #     "entities",
        #     "phash2",
        #     "materialid",
        #     "materialurl",
        #     "direct_id",
        #     "shoot_id",
        #     "script_id",
        #     "subtitle",
        #     "describe",
        #     "report_unchange",
        #     "report_updatetime",
        # ]
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

    def get_data_from_db(self):
        # 打开数据库连接
        db = pymysql.connect(
            host="192.168.0.18",
            user="zhangruifeng",
            passwd="t0dG18PjAJ8c8EgR",
            database="adwetec_prod",
        )

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
        agg_fields = {"bclk": "bclk", "pclk": "pclk", "cost": "cost", "clk": "clk"}
        aggs = {}
        for agg_field, target in agg_fields.items():
            aggs[agg_field] = {"sum": {"field": target}}
        query = {
            "size": 0,
            "_source": {"includes": ["matid", "date"]},
            "aggs": {
                agg_name: {
                    "terms": {
                        "size": len(matids),
                        "field": "matid",
                        "order": {"cost": "desc"},
                    },
                    "aggs": aggs,
                }
            },
            "query": {
                "bool": {"must": [{"terms": {"matid": matids}}, {"term": {"medid": 8}}]}
            },
        }
        result = es.search(index="creative-report-2021-*", body=query)
        self.logger.debug(result["_shards"])
        sale_data = result["aggregations"][agg_name]["buckets"]
        self.logger.debug(
            "Bucket length: %d, id length %d" % (len(sale_data), len(matids))
        )
        id_dict = {}

        for dta in sale_data:
            # clk即素材曝光数量少于阈值
            if dta['clk']['value'] < config.threshold:
                continue
            id_dict[dta["key"]] = {}
            for field in agg_fields:
                id_dict[dta["key"]][field] = dta[field]["value"]

        # 包含没有es数据和es数据量太少的
        no_data_cols = []
        new_cols = {field: [] for field in agg_fields}
        for col_num, id in enumerate(matids):
            for field in agg_fields:
                try:
                    new_cols[field].append(id_dict[str(id)][field])
                except KeyError:
                    no_data_cols.append(col_num)
                    break
        self.logger.info("%d cols dropped due to lack of es data" % len(no_data_cols))
        self.data.drop(index=self.data.index[no_data_cols], inplace=True)
        for field in agg_fields:
            self.data[field] = new_cols[field]
        self.data.to_csv(self.data_folder + "kuaishou_data_es.csv")


if __name__ == "__main__":
    DataParser(es=True, analyze=False, from_db=False)
