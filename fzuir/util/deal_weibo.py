# -*- coding:utf-8 -*-

import re

def dealWeiboContent(title):
    # result, number = re.subn("[\\s*]", "", title)
    # print(result)
    result = str(title)
    index1 = result.find("http://t.cn/")
    if index1 > -1:
        result = result.replace(result[index1:index1 + 19], "").strip()
    index1 = result.find("转发了", 0, 6)
    index2 = result.find("的微博:", 4)
    if index1 > -1 and index2 > -1:
        if index1 + 30 <= index2 or index1 >= index2:
            return ""
        # 去掉转发了...的微博:  只要微博正文
        result = result[index2 + 4:]
        # 转发理由：***** 赞[4] 转发[4] 评论[1]
        index1 = result.find("转发理由:", 25)
        index2 = result.find("赞[", 30)
        if index1 > -1 and index2 > -1 and index1 < index2:
            result = result[0:index1]
        # [组图共4张]  原图  赞[2]  原文转发[1] 原文评论[1]  转发理由:
        index1 = result.find("[组图共", 15)
        index2 = result.find("转发理由:", 25)
        if index1 < 0:
            # 原图 赞[18587] 原文转发[3567] 原文评论[1991]转发理由:轉發微博 全文 原图 赞[103]
            # 秒拍视频?赞[29]?原文转发[6]?原文评论[2]转发理由:
            index1 = result.find("原图", 30)
            if index1 < 0:
                index1 = result.find("秒拍视频", 30)
                if index1 < 0:
                    index1 = result.find("赞[", 30)

        if index1 > -1 and index2 > -1 and index1 < index2:
            result = result.replace(result[index1:index2 + 5], "").strip()
    else:
        # 非转发的微博
        index1 = result.find("赞[")
        index2 = result.find("原图")
        if index1 > -1 and index2 > -1 and index2 + 3 == index1:
            result = result[0: index2].strip()
        elif index1 != -1:
            result = result[0: index1].strip()
        elif index2 != -1:
            result = result[0: index2].strip()

        result, number = re.subn("[\\s*]", "", result)

    if result.endswith("原图"):
        result = result[0: len(result) - 2]
    index1 = result.find("组图共")
    if index1 > -1:
        result = result[0: index1 - 1].strip()

    result = result.replace("分享网易新闻", "")
    result = result.replace("分享新浪新闻", "")
    result = result.replace("分享腾讯新闻", "")
    result = result.replace("分享搜狐新闻", "")
    result = result.replace("来自@网易新闻客户端", "")
    result = result.replace("来自@新浪新闻客户端", "")
    result = result.replace("来自@搜狐新闻客户端", "")
    result = result.replace("来自@腾讯新闻客户端", "")
    result = result.replace("来自@腾讯新闻客户端", "")
    result = result.replace("分享自凤凰新闻客户端", "")
    result = result.replace("分享自@凤凰视频客户端", "")
    result = result.replace("分享自@凤凰新闻客户端", "")
    result = result.replace("好文分享", "")
    result = result.replace("显示地图", "")
    result = result.replace("阅读全文请戳右边", "")
    result = result.replace("下载地址", "")
    result = result.replace("我在看新闻", "")

    return result.strip()


def removeExpression(title, expression_list):
    new_title = str(title).strip()
    for ex in expression_list:
        if "[" in new_title and "]" in new_title:
            # print("存在[ and ]  %s " % ex)
            if ex[0] in new_title:
                # print("%s出现" % ex)
                new_title = new_title.replace(ex[0], "")
        else:
            break
    return new_title


def removePunctuation(title):
    """
    去除标点符号，需要把每个标点的地方换成空格，这样便于分词
    :param title:
    :return:
    """
    punctuation = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗#〞︰︱︳﹐､﹒
            ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
            々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

    title = str(title)
    for ch in title:
        if ch in punctuation:
            title = title.replace(ch, " ")
    return title


def readExpression(conn):
    """
    读数据库中的表情符号  后面要去掉微博文本中的表情符号
    """
    expression_list = conn.queryData("select expression from Expression")
    return expression_list