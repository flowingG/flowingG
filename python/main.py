import re
import html
import jieba
import jieba.analyse
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity(object):

    #余弦相似度
  
    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 正则过滤 html 标签
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        # html 转义符实体化
        content = html.unescape(content)
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=200, withWeight=False)
        return keywords    
        
    @staticmethod
    def one_hot(dict, keywords):  # oneHot编码
        lens = len(dict)
        code = [0]*lens
        for word in keywords:
            code[dict[word]] += 1
        return code

    def __init__(self, x, y):
        self.s1 = x
        self.s2 = y

    def main(self):
        # 编码
        dict = {}
        i = 0        
        # 去除停用词
        jieba.analyse.set_stop_words(r'C:\test\stopwords.txt')

        # 提取关键词
        keywords1 = self.extract_keyword(self.s1)
        keywords2 = self.extract_keyword(self.s2)
        # 词的并集
        union = set(keywords1).union(set(keywords2))
        for word in union:
            dict[word] = i
            i += 1
        # oneHot编码
        code1 = self.one_hot(dict, keywords1)
        code2 = self.one_hot(dict, keywords2)
        # 余弦相似度计算
        sample = [code1, code2]
        # 除零处理
        try:
            sim = cosine_similarity(sample)
            return sim[1][0]
        except Exception as e:
            print(e)
            return 0.0

            # 测试
if __name__ == '__main__':
    with open(r'C:\test\orig.txt', 'r',encoding='utf-8') as x, open(r'C:\test\orig_0.8_add.txt', 'r',encoding='utf-8') as y:
        content_x = x.read()
        content_y = y.read()
        similarity = CosineSimilarity(content_x, content_y)
        similarity = similarity.main()
        result = similarity*100
        print('相似度: %.1f%%' % result)
        f3 = open(r'C:\test\result.txt','w')
        f3.write('文本相似度:'+str(result)+'%')     
    