import pandas as pd
import numpy as np
import torch
from itertools import combinations
from predict_model import Text_CNN

# 초성 리스트. 00 ~ 18 총 19개
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20 총 21개
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음) 총 28개
JONGSUNG_LIST = ['J없음', 'Jㄱ', 'Jㄲ', 'Jㄳ', 'Jㄴ', 'Jㄵ', 'Jㄶ', 'Jㄷ', 'Jㄹ', 'Jㄺ', 'Jㄻ', 'Jㄼ', 'Jㄽ', 'Jㄾ', 'Jㄿ', 'Jㅀ', 'Jㅁ', 'Jㅂ', 'Jㅄ', 'Jㅅ', 'Jㅆ', 'Jㅇ', 'Jㅈ', 'Jㅊ', 'Jㅋ', 'Jㅌ', 'Jㅍ', 'Jㅎ']
#숫자 리스트 00 ~ 09 총 10개
NUMBER_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#알파벳 00 ~ 25 총 25개
ALPHABET_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#특수문자 00 ~ 18 총 19개
SPECIAL_CHAR_LIST = ['!', '\"', '#', '$', '%', '&', '\'', '?', '@', '*', '+', ',', '-', '.', '/', '~', ' ', ':', '^']
#추가 총 4개
SINGLE_CHAR_LIST = ['ㄳ', 'ㄵ', 'ㅄ','ㄺ']
#총 127개

ALL_CHAR = []
ALL_CHAR = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
ALL_CHAR += JUNGSUNG_LIST
ALL_CHAR += JONGSUNG_LIST
ALL_CHAR += NUMBER_LIST
ALL_CHAR += ALPHABET_LIST
ALL_CHAR += SPECIAL_CHAR_LIST
ALL_CHAR += SINGLE_CHAR_LIST

#문자열 데이터가 들어오면 one hot encoding을해서 리스트를 반환.
def onehotencoding(string):
    return_list = []
    tmp_list = []
    check = 0
    for i in range(len(string)):
        for k in range(len(ALL_CHAR)):
            if(ALL_CHAR[k] == string[i]):
                tmp_list.append(1)
                check = 1
            else:
                tmp_list.append(0)
        if(check == 1):
            return_list.append(tmp_list)
            tmp_list = []
            check = 0
        else:
            tmp_list = []
    return return_list


# 자모단위로 분해
def tokenize(word):
    r_lst = []
    for w in list(word.strip()):
        ## 영어인 경우 구분해서 작성함.
        if '가' <= w <= '힣':
            ## 588개 마다 초성이 바뀜.
            ch1 = (ord(w) - ord('가')) // 588
            ## 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2
            r_lst.append(CHOSUNG_LIST[ch1])
            r_lst.append(JUNGSUNG_LIST[ch2])
            if ch3 != 0:
                r_lst.append(JONGSUNG_LIST[ch3])
        elif 'A' <= w <= 'Z':
            ch = (ord(w) - ord('A'))
            r_lst.append(ALPHABET_LIST[ch])
        elif 'a' <= w <= 'z':
            ch = (ord(w) - ord('a'))
            r_lst.append(ALPHABET_LIST[ch])
        else:
            r_lst.append(w)

    return r_lst

#데이터 크기가 n보다 크면 n까지 자르고, n보다 작으면 n크기까지 zero padding을 함.
def max_resize_data(data):
    n = 200
    if(len(data) > n):
        data = data[:n]
    return data

def zero_padding_resize_data(data):
    zero_list = [0 for k in range(127)]
    for i in range(len(data), 200):
        data.append(zero_list)
    return data

def isbadword(s_list):
    li_str = []
    li_num = []
    for i in range(0, len(s_list)):
        a = list(combinations(s_list, i + 1))
        for j in range(0, len(a)):
            str1 = ' '.join(a[j])
            li_str.append(str1)
            li_num.append(make_output(str1))

    li_str.reverse()
    li_num.reverse()
    return li_str, li_num


def choose_one_string(li_str, li_num, s_list):
    for i in range(0, len(li_num)):
        # if( make_output(s)-0.5 > li_num):
        if (0.4 > li_num[i]): #욕일 확률이 40%이하인 것을 찾아서 출력
            a = li_str[i].split()

            for j in range(0, len(s_list)):

                if (a.count(s_list[j]) == 0):
                    len_l = len(s_list[j])
                    s_list.remove(s_list[j])
                    num_s = "*"
                    for k in range(1, len_l):
                        num_s = num_s + "*"
                    s_list.insert(j, num_s)

            return ' '.join(s_list)

    tmp = []
    for i in range(0, len(s_list)):
        num_s = "*"
        for j in range(1, len(s_list[i])):
            num_s = num_s + "*"
        tmp.insert(i, num_s)
    return ' '.join(tmp)

def check_isbad(num):
    if(num > 0.6): #욕일 확률이 60%이상인것만 마스킹 처리함
        return True
    else:
        return False

#모델 불러오기
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_model = Text_CNN().to(device)
#load_model.load_state_dict(torch.load('./text_cnn_pytorch_V2.pth', map_location=lambda storage, loc: storage))
load_model.load_state_dict(torch.load('./text_cnn_pytorch_V4.pth'))

def make_output(string):
    x = zero_padding_resize_data(max_resize_data(onehotencoding(tokenize(string))))
    x = np.asarray(x).astype('float32')
    x = torch.from_numpy(x)
    x = x.reshape(1, 200, 127)
    x = x.permute(0, 2, 1)
    #x = x.to("cpu")
    x = x.to("cuda:0")
    output = load_model.forward(x)
    return output.item()

def aa(s):

    print("욕일 확률 : {}".format(make_output(s)*100))

    print("-------------분석---------------")

    if(check_isbad(make_output(s)) == True):#욕이라고 판단했을때, (기준은 60%이상)
        s_list = []
        s_list = s.split()
        li_str, li_num = isbadword(s_list)
        print_s = choose_one_string(li_str, li_num, s_list)
        for i in range(0, len(li_num)):
            print(li_str[i], "     ", li_num[i])
    else: #욕이 아니라고 판단했을때
        print_s = s

    print("-------------최종---------------")
    print(print_s)

    return print_s

aa("난 시발아 나가 뒤져야하냐")