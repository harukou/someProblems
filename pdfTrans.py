import hashlib #写签名
import random #salt随机数
import urllib.parse #解析url
import requests #向url提出请求
from concurrent import futures #并发
from io import StringIO #在内存中读写str

from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams


def read_from_pdf(file_path):
    '''
    解析pdf文件
    '''
    with open(file_path, 'rb') as file:
        resource_manager = PDFResourceManager() #创建资源管理器
        return_str = StringIO() #建立内存文件对象
        lap_params = LAParams() #参数分析器
        device = TextConverter(
            resource_manager, return_str, laparams=lap_params)
        process_pdf(resource_manager, device, file)
        device.close()
        content = return_str.getvalue()
        return_str.close()
        return content

def create_sign(q, appid, salt, key):
    '''
    制造签名
    '''
    sign = str(appid) + str(q) + str(salt) + str(key)
    md5 = hashlib.md5()
    md5.update(sign.encode('utf-8'))
    return md5.hexdigest()


def create_url(q, url):
    '''
    根据参数构造query字典
    '''
    fro = 'auto'
    to = 'zh'
    salt = random.randint(32768, 65536)
    sign = create_sign(q, appid, salt, key)
    url = url+'?appid='+str(appid)+'&q='+urllib.parse.quote(q)+'&from='+str(fro)+'&to='+str(to)+'&salt='+str(salt)+'&sign='+str(sign)
    return url


def translate(q):
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    url = create_url(q, url)
    r = requests.get(url)
    txt = r.json()
    if txt.get('trans_result', -1) == -1:
        print('程序已经出错，请查看报错信息：\n{}'.format(txt))
        return '这一部分翻译错误\n'
    #print(type(txt['trans_result']))
    return txt['trans_result'][0]['dst'] #trans_result为list，成员为dict


def clean_data(data):
    '''
    将输入的data返回成为段落组成的列表
    '''
    data = data.replace('\n\n', '*#')
    data = data.replace('\n', ' ')
    return data.split('*#')


def _main(pdf_path, txt_path):
    # try:
    data = read_from_pdf(pdf_path)
    data_list = clean_data(data) #分段
    with futures.ThreadPoolExecutor() as excuter:
        zh_txt = excuter.map(translate, data_list)
    #zh_txt = list(zh_txt)
    article = '\n\n'.join(zh_txt)
    #print(article)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(article)



if __name__ == '__main__':
    appid = 20180420000148443    #appid 
    key = 'olczrqHqer4zbZGheVf0'#key 
    pdfname = input('input the pdf name(xxx.pdf):')
    txtname = pdfname.split('.')[0]+'.txt'
    _main(pdfname, txtname)  #填入 pdf 路径与翻译完毕之后的 txt 路径
    print('****That is all !')
