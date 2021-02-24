import requests as rq
from bs4 import BeautifulSoup
import re
from openpyxl import Workbook
from selenium import webdriver

def get_html(url):
    _html = ""
    resp = rq.get(url)
    if resp.status_code == 200:
        _html = resp.text
        return _html
    else:
        return 0


def get_datgel(base_url):
    #file = open('test.txt', 'a', encoding='utf-8')
    review_list = []
    count = 1
    driver = webdriver.Chrome()
    driver.get(base_url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # reviews = soup.find('div', {'class':'commentListInner'}).find_all("div", re.compile("comment"))
    reviews = soup.find_all('span', {'class': 'cmt'})
    # print(reviews)

    for review in reviews:
        review_list.append(review.get_text().strip())
        #print(count, "\b번 댓글 : ", review_list[-1])
        #print("\n")
        count += 1
        #file.write(review_list[-1] + "\n--------------------------------------------------\n")
    #file.close()
    return review_list

login_url = 'http://www.ilbe.com/'

user = ###id
password = ###pwd

# requests.session 메서드는 해당 reqeusts를 사용하는 동안 cookie를 header에 유지하도록 하여
# 세션이 필요한 HTTP 요청에 사용됩니다.
session = rq.session()

params = dict()
params['m_id'] = user
params['m_passwd'] = password

session = rq.session()
# javascrit(jQuery) 코드를 분석해보니, 결국 login_proc.php 를 m_id 와 m_passwd 값과 함께
# POST로 호출하기 때문에 다음과 같이 requests.session.post() 메서드를 활용하였습니다.
# 실제코드: <form name="frm"  id="frm"  action="#" method="post">
res = session.post(login_url, data = params)

# 응답코드가 200 즉, OK가 아닌 경우 에러를 발생시키는 메서드입니다.
res.raise_for_status()

print(res)

base_url = 'http://www.ilbe.com/list/ilbe?'
page_path = 'page=%d'
end_url = '&listStyle=list'

page = 300
count = 0
review_list = []

# 몇페이지 크롤링할지 1이면 첫페이지만
while page >= 1:
    print(page)
    sub_path = page_path % (page)
    html = get_html(base_url + sub_path + end_url)

    if (html == 0):
        break

    soup = BeautifulSoup(html, 'html.parser')
    area = soup.find_all("a", {"class": "subject"})

    for index in area:
        count += 1
        _url = index["href"]
        _url = 'http://www.ilbe.com' + _url
        _text = index.text.split(".")
        # print(count, "\b번 제목 : ", _text[0])

        review_list.append(get_datgel(_url))

    page -= 1

    print("....done")

#print(review_list)

sum1 = 1
i = 0
j = 0

wb = Workbook()

#파일 이름을 정하고, 데이터를 넣을 시트를 활성화합니다.
sheet1 = wb.active
file_name = 'ilbe_Crawling.xlsx'

#시트의 이름을 정합니다.
sheet1.title = 'data'

#Sheet1에다 입력
sheet1 = wb.active
sheet1['A1'] = 'content'
sheet1['B1'] = 'index'

for j in range(0, len(review_list)):
    for i in range(0, len(review_list[j])):
        sum1 = sum1 + 1
        sheet1.cell(row=sum1, column=1).value = review_list[j][i]

wb.save(filename=file_name)
