"""
Post-evaluate on the predicted file
"""
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import nltk
import statistics
import jieba

smoothing_function = SmoothingFunction().method2


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
    def bleu_score(references, predictions):
        """
        :return:
        """
        bleu1s = []
        bleu2s = []
        bleu3s = []
        bleu4s = []
        for ref, cand in zip(references, predictions):
            ref_list = [ref.split(' ')]
            cand_list = cand.split(' ')
            bleu1 = sentence_bleu(ref_list, cand_list, weights=(1, 0, 0, 0),
                                  smoothing_function=smoothing_function)
            bleu1s.append(bleu1)
            bleu2 = sentence_bleu(ref_list, cand_list, weights=(0.5, 0.5, 0,
                                                                0), smoothing_function=smoothing_function)
            bleu2s.append(bleu2)
            bleu3 = sentence_bleu(ref_list, cand_list, weights=(0.33, 0.33,
                                                                0.33, 0), smoothing_function=smoothing_function)
            bleu3s.append(bleu3)
            bleu4 = sentence_bleu(ref_list, cand_list, weights=(0.25, 0.25,
                                                                0.25, 0.25), smoothing_function=smoothing_function)
            bleu4s.append(bleu4)
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

    @staticmethod
    def restored_count(references, predictions, currents):

        def score_function(ref_n_gram, pred_n_gram, ref_restore, pred_restore):
            ref_restore = set(ref_restore)
            pred_restore = set(pred_restore)
            ref_n_gram = set([ngram_phrase for ngram_phrase in ref_n_gram if
                              set(ngram_phrase) & ref_restore])
            pred_n_gram = set([ngram_phrase for ngram_phrase in pred_n_gram if
                               set(ngram_phrase) & pred_restore])
            inter_count = len(ref_n_gram & pred_n_gram)
            pred_count = len(pred_n_gram)
            ref_count = len(ref_n_gram)
            return inter_count, pred_count, ref_count

        inter_count_1 = []
        pred_count_1 = []
        ref_count_1 = []

        inter_count_2 = []
        pred_count_2 = []
        ref_count_2 = []

        inter_count_3 = []
        pred_count_3 = []
        ref_count_3 = []

        for ref, cand, cur in zip(references, predictions, currents):
            ref_tokens = ref.split(' ')
            pred_tokens = cand.split(' ')
            cur_tokens = cur.split(' ')
            ref_restore_tokens = [token for token in ref_tokens if token not in
                                  cur_tokens]
            pred_restore_tokens = [token for token in pred_tokens if token not in
                                   cur_tokens]
            if len(ref_restore_tokens) == 0:
                continue
            ref_ngram_1 = list(nltk.ngrams(ref_tokens, n=1))
            pred_ngram_1 = list(nltk.ngrams(pred_tokens, n=1))
            inter_1, pred_1, ref_1 = score_function(ref_ngram_1, pred_ngram_1, ref_restore_tokens, pred_restore_tokens)

            ref_ngram_2 = list(nltk.ngrams(ref_tokens, n=2))
            pred_ngram_2 = list(nltk.ngrams(pred_tokens, n=2))
            inter_2, pred_2, ref_2 = score_function(ref_ngram_2, pred_ngram_2, ref_restore_tokens, pred_restore_tokens)

            ref_ngram_3 = list(nltk.ngrams(ref_tokens, n=3))
            pred_ngram_3 = list(nltk.ngrams(pred_tokens, n=3))
            inter_3, pred_3, ref_3 = score_function(ref_ngram_3, pred_ngram_3, ref_restore_tokens, pred_restore_tokens)

            inter_count_1.append(inter_1)
            pred_count_1.append(pred_1)
            ref_count_1.append(ref_1)
            inter_count_2.append(inter_2)
            pred_count_2.append(pred_2)
            ref_count_2.append(ref_2)
            inter_count_3.append(inter_3)
            pred_count_3.append(pred_3)
            ref_count_3.append(ref_3)

        return (inter_count_1, pred_count_1, ref_count_1,
                inter_count_2, pred_count_2, ref_count_2,
                inter_count_3, pred_count_3, ref_count_3)


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
        metrics['BLEU1'], metrics['BLEU2'], _, metrics['BLEU4'] = Scorer.bleu_score(references, predictions)
        metrics['ROUGE1'], metrics['ROUGE2'], metrics['ROUGEL'] = Scorer.rouge_score(references, predictions)
        for key in metrics.keys():
            if isinstance(metrics[key], list):
                metrics[key] = statistics.mean(metrics[key])
        print(metrics)


if __name__ == '__main__':
    read_file_and_score("log\\fix_dataset\decode_test_50maxenc_4beam_5mindec_30maxdec_ckpt-10017\\result.txt")
