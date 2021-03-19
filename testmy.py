#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software:XXXX
@file: testmy.py
@time: 2021/3/16 15:09
@desc:
'''
import json
str = '[{"bg":"0","ed":"16400","onebest":"孩子上了四年级后，英语就变成了每次考试的必考科目之一，当家长们还在为数学语文给孩子报补习班的时候，英语往往成了被忽视的那一个学科，因为在家长的理念里英语就是要背单词，而且要大量","si":"0","speaker":"1","wordsResultList":[{"alternativeList":[],"wc":"1.0000","wordBg":"1","wordEd":"36","wordsName":"孩子","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"37","wordEd":"60","wordsName":"上","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"61","wordEd":"72","wordsName":"了","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"73","wordEd":"100","wordsName":"四","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"101","wordEd":"136","wordsName":"年级","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"137","wordEd":"176","wordsName":"后","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"176","wordEd":"176","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"177","wordEd":"228","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"229","wordEd":"244","wordsName":"就","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"245","wordEd":"280","wordsName":"变成","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"281","wordEd":"296","wordsName":"了","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"297","wordEd":"336","wordsName":"每次","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"337","wordEd":"376","wordsName":"考试","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"377","wordEd":"392","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"393","wordEd":"420","wordsName":"必","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"421","wordEd":"440","wordsName":"考","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"441","wordEd":"472","wordsName":"科目","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"473","wordEd":"500","wordsName":"之","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"501","wordEd":"516","wordsName":"一","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"516","wordEd":"516","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"517","wordEd":"536","wordsName":"当","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"537","wordEd":"572","wordsName":"家长","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"573","wordEd":"584","wordsName":"们","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"585","wordEd":"608","wordsName":"还","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"609","wordEd":"620","wordsName":"在","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"621","wordEd":"648","wordsName":"为","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"649","wordEd":"692","wordsName":"数学","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"693","wordEd":"736","wordsName":"语文","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"737","wordEd":"752","wordsName":"给","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"753","wordEd":"780","wordsName":"孩子","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"781","wordEd":"796","wordsName":"报","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"797","wordEd":"840","wordsName":"补习班","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"841","wordEd":"848","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"849","wordEd":"900","wordsName":"时候","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"900","wordEd":"900","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"901","wordEd":"964","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"965","wordEd":"1004","wordsName":"往往","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1005","wordEd":"1044","wordsName":"成了","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1045","wordEd":"1076","wordsName":"被","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1077","wordEd":"1116","wordsName":"忽视","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1117","wordEd":"1132","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1133","wordEd":"1144","wordsName":"那","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1145","wordEd":"1168","wordsName":"一个","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1169","wordEd":"1220","wordsName":"学科","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"1220","wordEd":"1220","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"1221","wordEd":"1268","wordsName":"因为","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1269","wordEd":"1292","wordsName":"在","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1293","wordEd":"1332","wordsName":"家长","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1333","wordEd":"1340","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1341","wordEd":"1380","wordsName":"理念","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1381","wordEd":"1404","wordsName":"里","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1405","wordEd":"1444","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1445","wordEd":"1468","wordsName":"就是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1469","wordEd":"1480","wordsName":"要","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1481","wordEd":"1492","wordsName":"背","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1493","wordEd":"1548","wordsName":"单词","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"1548","wordEd":"1548","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"1549","wordEd":"1584","wordsName":"而且","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1585","wordEd":"1604","wordsName":"要","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1605","wordEd":"1632","wordsName":"大量","wp":"n"}]},{"bg":"16420","ed":"18590","onebest":"的背，但是家长们你们有想过吗？","si":"0","speaker":"1","wordsResultList":[{"alternativeList":[],"wc":"1.0000","wordBg":"1","wordEd":"8","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"9","wordEd":"24","wordsName":"背","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"24","wordEd":"24","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"25","wordEd":"48","wordsName":"但是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"49","wordEd":"84","wordsName":"家长","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"85","wordEd":"104","wordsName":"们","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"105","wordEd":"132","wordsName":"你们","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"133","wordEd":"144","wordsName":"有","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"145","wordEd":"164","wordsName":"想","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"165","wordEd":"180","wordsName":"过","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"181","wordEd":"208","wordsName":"吗","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"208","wordEd":"208","wordsName":"？","wp":"p"}]},{"bg":"18790","ed":"35200","onebest":"其实学习英语也是有方法的，学习英语也是有启蒙期的吗？四到六年级是孩子学习英语的启蒙期，也是黄金期，一旦错过未来给孩子投入再多的时间和金钱都是没有用的，来试试跟谁学。","si":"0","speaker":"1","wordsResultList":[{"alternativeList":[],"wc":"1.0000","wordBg":"1","wordEd":"40","wordsName":"其实","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"41","wordEd":"80","wordsName":"学习","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"81","wordEd":"116","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"117","wordEd":"136","wordsName":"也","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"137","wordEd":"148","wordsName":"是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"149","wordEd":"168","wordsName":"有","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"169","wordEd":"220","wordsName":"方法","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"221","wordEd":"252","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"252","wordEd":"252","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"253","wordEd":"288","wordsName":"学习","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"289","wordEd":"332","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"333","wordEd":"348","wordsName":"也","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"349","wordEd":"364","wordsName":"是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"365","wordEd":"380","wordsName":"有","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"381","wordEd":"420","wordsName":"启蒙","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"421","wordEd":"436","wordsName":"期","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"437","wordEd":"452","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"453","wordEd":"492","wordsName":"吗","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"492","wordEd":"492","wordsName":"？","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"493","wordEd":"556","wordsName":"四","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"557","wordEd":"572","wordsName":"到","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"573","wordEd":"604","wordsName":"六","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"605","wordEd":"652","wordsName":"年级","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"653","wordEd":"680","wordsName":"是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"681","wordEd":"720","wordsName":"孩子","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"721","wordEd":"752","wordsName":"学习","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"753","wordEd":"780","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"781","wordEd":"796","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"797","wordEd":"836","wordsName":"启蒙","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"837","wordEd":"868","wordsName":"期","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"868","wordEd":"868","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"869","wordEd":"888","wordsName":"也","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"889","wordEd":"916","wordsName":"是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"917","wordEd":"976","wordsName":"黄金","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"977","wordEd":"1020","wordsName":"期","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"1020","wordEd":"1020","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"1021","wordEd":"1060","wordsName":"一旦","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1061","wordEd":"1116","wordsName":"错过","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1117","wordEd":"1148","wordsName":"未来","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1149","wordEd":"1164","wordsName":"给","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1165","wordEd":"1192","wordsName":"孩子","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1193","wordEd":"1236","wordsName":"投入","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1237","wordEd":"1260","wordsName":"再","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1261","wordEd":"1280","wordsName":"多","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1281","wordEd":"1292","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1293","wordEd":"1336","wordsName":"时间","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1337","wordEd":"1348","wordsName":"和","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1349","wordEd":"1392","wordsName":"金钱","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1393","wordEd":"1424","wordsName":"都是","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1425","wordEd":"1448","wordsName":"没有","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1449","wordEd":"1468","wordsName":"用","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1469","wordEd":"1488","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"1488","wordEd":"1488","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"1489","wordEd":"1508","wordsName":"来","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1509","wordEd":"1548","wordsName":"试试","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1549","wordEd":"1568","wordsName":"跟","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1569","wordEd":"1592","wordsName":"谁","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"1593","wordEd":"1636","wordsName":"学","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"1636","wordEd":"1636","wordsName":"。","wp":"p"}]},{"bg":"35230","ed":"45850","onebest":"4天英语特训班吧陈军老师主讲总结10多种速记单词的方法，让孩子爱上学英语扫码进去，还有更多的学习资料可以领取。","si":"0","speaker":"1","wordsResultList":[{"alternativeList":[],"wc":"1.0000","wordBg":"1","wordEd":"8","wordsName":"4","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"9","wordEd":"36","wordsName":"天","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"37","wordEd":"68","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"69","wordEd":"84","wordsName":"特","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"85","wordEd":"96","wordsName":"训","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"97","wordEd":"116","wordsName":"班","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"117","wordEd":"156","wordsName":"吧","wp":"s"},{"alternativeList":[],"wc":"1.0000","wordBg":"157","wordEd":"184","wordsName":"陈","wp":"n"},{"alternativeList":[{"wc":"0.1993","wordsName":"君","wp":"n"}],"wc":"0.8007","wordBg":"185","wordEd":"204","wordsName":"军","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"205","wordEd":"232","wordsName":"老师","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"233","wordEd":"280","wordsName":"主讲","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"281","wordEd":"340","wordsName":"总结","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"341","wordEd":"376","wordsName":"10","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"377","wordEd":"424","wordsName":"多种","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"425","wordEd":"436","wordsName":"速","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"437","wordEd":"452","wordsName":"记","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"453","wordEd":"480","wordsName":"单词","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"481","wordEd":"492","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"493","wordEd":"548","wordsName":"方法","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"548","wordEd":"548","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"549","wordEd":"568","wordsName":"让","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"569","wordEd":"608","wordsName":"孩子","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"609","wordEd":"652","wordsName":"爱上","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"653","wordEd":"664","wordsName":"学","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"665","wordEd":"700","wordsName":"英语","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"701","wordEd":"720","wordsName":"扫","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"721","wordEd":"740","wordsName":"码","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"741","wordEd":"776","wordsName":"进去","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"776","wordEd":"776","wordsName":"，","wp":"p"},{"alternativeList":[],"wc":"1.0000","wordBg":"777","wordEd":"816","wordsName":"还有","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"817","wordEd":"848","wordsName":"更多","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"849","wordEd":"860","wordsName":"的","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"861","wordEd":"884","wordsName":"学习","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"885","wordEd":"924","wordsName":"资料","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"925","wordEd":"944","wordsName":"可以","wp":"n"},{"alternativeList":[],"wc":"1.0000","wordBg":"945","wordEd":"976","wordsName":"领取","wp":"n"},{"alternativeList":[],"wc":"0.0000","wordBg":"976","wordEd":"976","wordsName":"。","wp":"p"},{"alternativeList":[],"wc":"0.0000","wordBg":"976","wordEd":"976","wordsName":"","wp":"g"}]},{"bg":"45980","ed":"47080","onebest":"嗯","si":"1","speaker":"1","wordsResultList":[{"alternativeList":[],"wc":"1.0000","wordBg":"9","wordEd":"16","wordsName":"嗯","wp":"s"}]}]'
dictobj = json.loads(str)
for obj in dictobj:
    print(obj['onebest'])
    print(obj['si'])
    for word in obj['wordsResultList']:
        print(word)
