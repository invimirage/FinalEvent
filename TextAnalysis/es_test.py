#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangruifeng
@contact: zrf1999@pku.edu.cn
@file: es_test.py
@time: 2021/4/18 19:58
@github: local 16351726fa15c85f565b7d5fecdf320ea67a72ef
"""
from elasticsearch import Elasticsearch


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
    "size": 1,
    "aggs": {
        "advid_video_count": {
            "terms": {"field": "advid", "size": 2000, "order": {"count": "desc"}},
            "aggs": {"count": {"cardinality": {"field": "matid"}}},
        }
    },
    "query": {
        "bool": {
            "must": [{"term": {"medid": 8}}],
            "must_not": [{"terms": {"matid": [0, -1]}}],
        }
    },
}
result = es.search(index="creative-report-*", body=query)
buckets = result["aggregations"]["advid_video_count"]["buckets"]
print(len(buckets))
print(result["aggregations"]["advid_video_count"]["sum_other_doc_count"])
sum_ad = 0
one_hots = []
for bucket in buckets:
    sum_ad += int(bucket["count"]["value"])
    if bucket["count"]["value"] < 500:
        break
    one_hots.append(bucket["key"])
    # print('Advid: %s \n Distinct ad count: %s \n Total ad count: %s' % (bucket['key'], bucket['count']['value'], bucket['doc_count']))
print(len(one_hots), one_hots)
