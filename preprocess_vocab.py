"""
Author: SivilTaram
Desc: this script is to build vocabulary from data file, the generated file is stored in data/vocab.txt.
The format is as [word \t\t frequency]
"""
import string
from collections import Counter
import jieba


def is_all_chinese(word):
    # identify whether all chinese characters
    for _char in word:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def cut_mixed_sentence(text):
    # for chinese, return character; for english, return word;
    jieba_words = list(jieba.cut(text))
    ret_chars = []
    for word in jieba_words:
        if is_all_chinese(word):
            ret_chars.extend(list(word))
        else:
            ret_chars.append(word)
    return ret_chars


def build_vocabulary(data_paths, write_vocab_path):
    words = []
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf8") as data_file:
            lines = data_file.readlines()
            for line in lines:
                # char-level statistic
                cut_words = cut_mixed_sentence(line.strip())
                words.extend([word for word in cut_words
                              if word not in string.whitespace])
    counter = Counter(words)
    vocab = counter.most_common(4000)
    write_f = open(write_vocab_path, "w", encoding="utf8")
    for key, value in vocab:
        write_f.write("{}\t\t{}\n".format(key, value))
    write_f.close()


if __name__ == '__main__':
    build_vocabulary(["data\\train.txt",
                      "data\\dev.txt"], "data\\vocab.txt")
