"""
Post-evaluate on the predicted file
"""
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import nltk
import statistics
import jieba
import json


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
    return ' '.join(ret_chars)


class Scorer(object):

    @staticmethod
    def corpus_bleu_score(references, predictions):
        ref_list = [[ref.split(' ')] for ref in references]
        pred_list = [pred.split(' ') for pred in predictions]
        bleu1s = corpus_bleu(ref_list, pred_list, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2s = corpus_bleu(ref_list, pred_list, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3s = corpus_bleu(ref_list, pred_list, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4s = corpus_bleu(ref_list, pred_list, weights=(0.25, 0.25, 0.25, 0.25))

        return bleu1s, bleu2s, bleu3s, bleu4s

    @staticmethod
    def em_score(references, predictions):
        matches = []
        for ref, cand in zip(references, predictions):
            if ref == cand:
                matches.append(1)
            else:
                matches.append(0)
        return matches

    @staticmethod
    def rouge_score(references, predictions):
        """
        https://github.com/pltrdy/rouge
        :param references: list string
        :param predictions: list string
        :return:
        """
        rouge = Rouge()
        rouge1s = []
        rouge2s = []
        rougels = []
        for ref, cand in zip(references, predictions):
            if cand.strip() == '':
                cand = 'hello'
            rouge_score = rouge.get_scores(cand, ref)
            rouge_1 = rouge_score[0]['rouge-1']['f']
            rouge_2 = rouge_score[0]['rouge-2']['f']
            rouge_l = rouge_score[0]['rouge-l']['f']
            rouge1s.append(rouge_1)
            rouge2s.append(rouge_2)
            rougels.append(rouge_l)
        return rouge1s, rouge2s, rougels


def read_file_and_score(result_file):
    metrics = {
        "EM": 0.0,
        "BLEU1": 0.0,
        "BLEU2": 0.0,
        "BLEU4": 0.0,
        "ROUGE1": 0.0,
        "ROUGE2": 0.0,
        "ROUGEL": 0.0
    }
    with open(result_file, "r", encoding="utf8") as result_f:
        lines = result_f.readlines()
        predictions = []
        references = []
        for line in lines:
            _, reference, prediction = line.strip().split('\t\t')
            # use space to split the sentence
            predictions.append(cut_mixed_sentence(prediction))
            references.append(cut_mixed_sentence(reference))

        metrics['EM'] = Scorer.em_score(references, predictions)
        metrics['BLEU1'], metrics['BLEU2'], _, metrics['BLEU4'] = Scorer.corpus_bleu_score(references, predictions)
        metrics['ROUGE1'], metrics['ROUGE2'], metrics['ROUGEL'] = Scorer.rouge_score(references, predictions)
        for key in metrics.keys():
            if isinstance(metrics[key], list):
                metrics[key] = statistics.mean(metrics[key])
        print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    read_file_and_score("log\\fix_dataset\\result.txt")
