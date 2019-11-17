import pandas as pd
import numpy as np
import torch

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

class load():

    def __init__(self, data_path):

        self.data_path = data_path
        self.split_train_test(0.3)

    def return_len(self):
        return len(self.train_set), len(self.test_set), len(self.val_set)

    # train데이터와 test데이터를 나눠주는 함수
    def split_train_test(self, test_ratio = 0.3):
        ilbe_data = pd.read_csv(self.data_path)
        shuffled_indices = np.random.permutation(len(ilbe_data))
        test_set_size = int(len(ilbe_data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        train_set = ilbe_data.iloc[train_indices]
        test_set = ilbe_data.iloc[test_indices]
        val_set = test_set[:int(len(test_set) / 2)]
        test_set = test_set[int(len(test_set) / 2):]

        train_set = train_set.values
        val_set = val_set.values
        test_set = test_set.values

        self.train_set = train_set.tolist()
        self.val_set = val_set.tolist()
        self.test_set = test_set.tolist()

    def main_processing(self):

        print("data loading....1/4")

        # train_set, val_set, test_set데이터를 전처리함
        train_x_data, train_y_data = self.dataprocessing(self.train_set)
        val_x_data, val_y_data = self.dataprocessing(self.val_set)
        test_x_data, test_y_data = self.dataprocessing(self.test_set)

        print("data loading....2/4")

        # 데이터 크기를 일정하게 200크기로 맞춤
        train_x_data_resized = self.zero_padding_resize_data(self.max_resize_data(train_x_data))
        val_x_data_resized = self.zero_padding_resize_data(self.max_resize_data(val_x_data))
        test_x_data_resized = self.zero_padding_resize_data(self.max_resize_data(test_x_data))
        train_x_data = 0
        val_x_data = 0
        test_x_data = 0

        print("data loading....3/4")
        print('resize....done')

        # list형태의 자료를 tensor로 바꿈
        x_train_torch = torch.FloatTensor(train_x_data_resized)
        x_val_torch = torch.FloatTensor(val_x_data_resized)
        x_test_torch = torch.FloatTensor(test_x_data_resized)

        y_train_torch = torch.FloatTensor(train_y_data)
        y_val_torch = torch.FloatTensor(val_y_data)
        y_test_torch = torch.FloatTensor(test_y_data)

        print("data loading....4/4")

        #x data reshape
        x_train_torch = x_train_torch.permute(0, 2, 1)
        x_val_torch = x_val_torch.permute(0, 2, 1)
        x_test_torch = x_test_torch.permute(0, 2, 1)

        return x_train_torch, x_val_torch, x_test_torch, y_train_torch, y_val_torch, y_test_torch

    # 문자열 데이터가 들어오면 one hot encoding을해서 리스트를 반환.
    def onehotencoding(self, string):
        return_list = []
        tmp_list = []
        check = 0
        for i in range(len(string)):
            for k in range(len(ALL_CHAR)):
                if (ALL_CHAR[k] == string[i]):
                    tmp_list.append(1)
                    check = 1
                else:
                    tmp_list.append(0)
            if (check == 1):
                return_list.append(tmp_list)
                tmp_list = []
                check = 0
            else:
                tmp_list = []
        return return_list

    # 자모단위로 분해
    def tokenize(self, word):
        r_lst = []
        word = str(word)
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

    # onehotencoding함수와 tokenize함수를 사용해 데이터 배열이 들어오면 자모단위로 분해하고 onehotencoding을 해서 내보냄
    def dataprocessing(self, data_set):
        r_list1 = []
        r_list2 = []
        for i in range(len(data_set)):
            r_list1.append(self.onehotencoding(self.tokenize(data_set[i][0])))
            r_list2.append(data_set[i][1])
        return r_list1, r_list2

    # 데이터 크기가 n보다 크면 n까지 자르고, n보다 작으면 n크기까지 zero padding을 함.
    def max_resize_data(self, data):
        n = 200
        for i in range(0, len(data)):
            if (len(data[i]) > n):
                data[i] = data[i][:n]
        return data

    def zero_padding_resize_data(self, data):
        zero_list = [0 for k in range(127)]
        for i in range(0, len(data)):
            for j in range(len(data[i]), 200):
                data[i].append(zero_list)
        return data
