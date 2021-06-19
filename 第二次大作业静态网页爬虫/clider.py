import requests
from bs4 import BeautifulSoup
import time
import os
import jieba
import csv

class BasicAnalyzer(object):
    """
    一个最基本的网页解析器，从response对象中获取text字段
    """

    def __call__(self, task_name, r):  # 重载__call__函数
        return r.text
##已经写好了的两个类，自己补充说明完成题目条件，以及根据这两个类完成的示例
##页面的请求与抓取，注意类的接口
class Cralwer(object):
    """
    一个用来爬取网页的类，其主要功能是依次抓取URL，并将返回的结果交给后续的解析器（Analyzer）进行处理。
    """

    def __init__(self, task_or_tasks, analyzer=BasicAnalyzer(),
                 headers={}, timeout=30, encoding=None, wait_time=-1):
        if isinstance(task_or_tasks, str):
            self.tasks = [task_or_tasks]
        if isinstance(task_or_tasks, list) or isinstance(task_or_tasks, tuple):
            self.tasks = list(task_or_tasks)
        print(self.tasks)
        self.analyzer = analyzer
        self.headers = headers
        self.timeout = timeout
        self.encoding = encoding
        self.wait_time = wait_time

        # 用于保存抓取请求返回的状态码
        self.response_codes = []

        # 用于遍历所有任务的迭代器
        self.__iterator = iter(self.tasks)

    def add_tasks(self, task_or_tasks):
        if isinstance(task_or_tasks, str):
            self.tasks.append(task_or_tasks)
        if isinstance(task_or_tasks, list) or isinstance(task_or_tasks, tuple):
            self.tasks += list(task_or_tasks)

    def crawl(self):
        return self.__next__()

    def crawl_all(self):
        return [result for result in self]

    def __process_task(self, task):
        if isinstance(task, str):  # 如果task是一个字符串，那么task代表要抓取的网页URI
            task_name = None
        elif isinstance(task, tuple) and len(task) == 2:  # 如果task是一个长度为2的元组，那么task表示（任务名，网页URI）
            task_name, task = task
        else:  # 否则报错
            raise ValueError("无法识别任务:{}".format(task))
        try:
            print(task_name, task)
            r = requests.get(task, headers=self.headers, timeout=self.timeout)
            if self.encoding is not None:
                r.encoding = self.encoding
            # print(r.text)
            self.response_codes.append((task_name, r.status_code))
        except:
            self.response_codes.append((task_name, None))  # 若遇到链接错误等问题，则此次任务的响应状态码为None
            return None
        return self.analyzer(task_name, r)  # 将response对象交给analyzer处理

    def __iter__(self):
        return self

    def __next__(self):
        task_uri = next(self.__iterator)
        if self.wait_time > 0:
            print("等待{}秒后开始抓取".format(self.wait_time))
            time.sleep(self.wait_time)
        return self.__process_task(task_uri)
##接下来是自己创建的一个类，它继承了BasicAnalyzer的方法,将文件保存到文档link.csv中
class LinkAnalyzer(BasicAnalyzer):
    def __init__(self, filename, encodings=None):

        self.filename = filename

    def __call__(self, task_name, r):
        html_text = super().__call__(task_name, r)
        soup = BeautifulSoup(html_text, 'html.parser')  # 使用自带的解析器，解析上述html文档
        with open(self.filename, 'w',encoding='utf-8') as fout:
            tags = soup.find("dl",class_="cat_box").find_all('a')
            for tag in tags:##soup没问题
                fout.write("{}\n".format( tag.get('href')))

##下面的这个类目的就是为了将之前的网址爬下来储存在电脑中
class FileStorageAnalyzer(BasicAnalyzer):
    def __init__(self, dir_path,x):
        self.dir_path = dir_path
        self.cnt = x##计数的写文件名


    def __call__(self, task_name, r):
        html_text = super().__call__(task_name, r)  # 继承父类，获取html text
        # 将其保存到文件
        self.dir_path = (self.dir_path + '第{:>02}章小说'+".html").format(str(self.cnt))


        print(type(self.cnt))
        print(self.dir_path)


        fout =  open(self.dir_path, 'w',encoding='utf-8')
        fout.write(html_text)
        fout.close()
        return html_text



def Step1():
##第一步提取小说各个章节的网址
    x1 = Cralwer(['http://jinyongxiaoshuo.com/xiaoaojianghu/'],analyzer=LinkAnalyzer("1.csv"), encoding='utf8', wait_time=2)
##到这里完成了把笑傲江湖所有章节的链接，放入1.csv文件中内
    x1.crawl()
    webs = []
    with open("1.csv", 'r') as fin:  # 注意这里的1.csv文件是上一个cell运行后生成的文件。
        for line in fin:
            line=line.replace('\n','')
            webs.append(line)
##提取小说的全部网页
    global x
    x = 1
    for web in webs:
        x2 = Cralwer(web,analyzer=FileStorageAnalyzer("笑傲江湖",x),encoding='utf-8',wait_time=2)
        x+=1
        x2.crawl_all()##成功把小说每个网页全部爬下来保存到本地


##接下来提取这些文件中的网页
def Step2():
    files = [x for x in  os.listdir() if os.path.splitext(x)[1] == '.html']##这个顺序有一点问题
    x = 1
    print(files)
    for file in files:

        txt1 = open(file,'r',encoding='utf-8').read()

        txt2 = BeautifulSoup(txt1)

        p = txt2.find('div',class_="entry").find_all('p')

        tags = [x.string for x in p if x.string is not None ]
        parsed_txt = "\n".join(tags)

        path = file[:-5]+'.txt'
        f = open(path,'w',encoding='utf-8')
        f.write(parsed_txt)
##至此，第二部完成，下面开始第三步
###第三步，Step3
def Step3():
    counts = {}
    files = [x for x in os.listdir() if os.path.splitext(x)[1] == '.txt']
    f1 = open("words.csv",'w',newline='')

    f1_csv = csv.writer(f1)
    for file in files:
        txt = open(file, 'r', encoding='utf-8').read()
        words = jieba.lcut(txt)
        for word in words:
            if len(word) == 1:
                continue
            else:
                counts[word] = counts.get(word, 0) + 1
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    f1_csv.writerows(items)
###至此，统计词频顺利完成
##接下来，统计人物的出场
def Step4():
    counts = {}
    files = [x for x in os.listdir() if os.path.splitext(x)[1] == '.txt']
    x = open("D:\新建文件夹\大作业2 (2)\笑傲江湖人物.txt",'r',encoding='utf-8')
    names0 = x.readline()
    names = []##构建人物词典索引
    k=''
    for i in names0:
        if i != ' ':
            k=k+i
        else:
            names.append(k)
            k=''
    ##此步骤成功

    f = open('character.csv', 'w', newline='')
    f_csv = csv.DictWriter(f,names)
    f_csv.writeheader()
    for file in files:
        counts = {}
        txt = open(file, 'r', encoding='utf-8').read()
        words = jieba.lcut(txt)
        for name in names :
            counts[name] = 0
        for word in words:
            if len(word) <= 1:
                continue
            if word in names:
                counts[word] = counts.get(word) + 1  ##按照人名字典加入
        print(counts)


        rows = [counts]
        f_csv.writerows(rows)

Step4()















