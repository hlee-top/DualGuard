from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy import interpolate
import numpy as np
from utils import read_jsonl, get_classifier, get_toxicity_score, cosine_similarity


def tpr_at_fpr(fpr, tpr, fpr_target):
    fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind='linear')
    return fpr_tpr_interpolation(fpr_target)


def f1_at_fpr(y_true, y_scores, fpr_target):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Finding the threshold for our target FPR
    threshold = thresholds[next(i for i in range(len(fpr)) if fpr[i] > fpr_target) - 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

    # Interpolating to find precision and recall at the target threshold
    precision_interp = interpolate.interp1d(thresholds_pr, precision[:-1], fill_value="extrapolate")
    recall_interp = interpolate.interp1d(thresholds_pr, recall[:-1], fill_value="extrapolate")
    precision_at_threshold = precision_interp(threshold)
    recall_at_threshold = recall_interp(threshold)

    # Calculate F1 score
    f1 = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold)
    return f1


def get_best_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]
    return best_threshold


def auc_score(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def watermark_evaluate(save_result_path, watermark_key_list, unwatermark_key_list, return_result=False):
    all_data = read_jsonl(save_result_path)
    y_true, y_scores = [], []
    watermark_label, original_label = 1, 0
    for data in all_data:
        for watermark_key in watermark_key_list:
            if not np.isnan(data[watermark_key]['score']):
                y_scores.append(data[watermark_key]['score'])
                y_true.append(watermark_label)
            else:
                print("nan")

        for unwatermark_key in unwatermark_key_list:
            if not np.isnan(data[unwatermark_key]['score']):
                y_scores.append(data[unwatermark_key]['score'])
                y_true.append(original_label)
            else:
                print("nan")
    auc = auc_score(y_true, y_scores)
    print("auc", auc)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_threshold = get_best_threshold(y_true, y_scores)
    print("best_threshold", best_threshold)

    print(watermark_key_list, unwatermark_key_list)
    print("auc\tTP@1%\tTP@5%\tTP@10%")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(auc, tpr_at_fpr(fpr, tpr, 0.01),
                                                  tpr_at_fpr(fpr, tpr, 0.05),
                                                  tpr_at_fpr(fpr, tpr, 0.1)))

    print("-" * 30)
    if return_result:
        return fpr, tpr, auc


def detect_evaluate(watermark_algo, save_result_path, modify_text_key_list, original_text_key_list, return_result=False,
                    tokenizer=None, token_num=None):
    classifier_name = "roberta"
    classifier_tokenizer, classifier = get_classifier(classifier_name)

    all_data = read_jsonl(save_result_path)
    y_true, y_scores = [], []
    modify_label, original_label = 1, 0
    modify_num, original_num = 0, 0
    for data in all_data:
        for modify_text_key in modify_text_key_list:
            text = data[modify_text_key]
            if tokenizer is not None and token_num is not None:
                text_id = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                text_id = text_id[:token_num]
                text = tokenizer.decode(text_id)
            if modify_num >= 200:
                continue
            toxic_score = get_toxicity_score(classifier_name, classifier_tokenizer, classifier, text)
            if toxic_score <= 0.5:
                continue
            detect_result = watermark_algo.detect_spoof(text)
            pos_logit = detect_result["pos_logits"]
            neg_logit = detect_result["neg_logits"]

            score_list = []
            for idx in range(len(pos_logit)):
                score_list.append(
                    1 - cosine_similarity(pos_logit[idx], neg_logit[idx]).cpu().detach().numpy().tolist()[0])
            score = np.mean(score_list).tolist()
            if not np.isnan(score):
                y_scores.append(score)
                y_true.append(modify_label)
                modify_num += 1

        for original_text_key in original_text_key_list:
            text = data[original_text_key]
            if tokenizer is not None and token_num is not None:
                text_id = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                text_id = text_id[:token_num]
                text = tokenizer.decode(text_id)
            if original_num >= 200:
                continue
            toxic_score = get_toxicity_score(classifier_name, classifier_tokenizer, classifier, text)
            if toxic_score > 0.5:
                continue
            detect_result = watermark_algo.detect_spoof(text)
            pos_logit = detect_result["pos_logits"]
            neg_logit = detect_result["neg_logits"]
            score_list = []
            for idx in range(len(pos_logit)):
                score_list.append(
                    1 - cosine_similarity(pos_logit[idx], neg_logit[idx]).cpu().detach().numpy().tolist()[0])
            score = np.mean(score_list).tolist()
            if not np.isnan(score):
                y_scores.append(score)
                y_true.append(original_label)
                original_num += 1

    auc = auc_score(y_true, y_scores)
    print("auc", auc)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_threshold = get_best_threshold(y_true, y_scores)
    print("best_threshold", best_threshold)

    print(modify_text_key_list, original_text_key_list)
    print("auc\tTP@1%\tTP@5%\tTP@10%")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(auc, tpr_at_fpr(fpr, tpr, 0.01),
                                                  tpr_at_fpr(fpr, tpr, 0.05),
                                                  tpr_at_fpr(fpr, tpr, 0.1)))
    print("-" * 30)

    if return_result:
        return fpr, tpr, auc


def trace_evaluate(watermark_algo, save_result_path, modify_text_key_list, original_text_key_list, return_result=False,
                   tokenizer=None, token_num=None):
    classifier_name = "roberta"
    classifier_tokenizer, classifier = get_classifier(classifier_name)

    all_data = read_jsonl(save_result_path)
    y_true, y_scores = [], []
    modify_label, original_label = 1, 0
    modify_num, original_num = 0, 0
    for data in all_data:
        watermarked_text = data["watermarked_text"]
        water_score = get_toxicity_score(classifier_name, classifier_tokenizer, classifier, watermarked_text)

        for modify_text_key in modify_text_key_list:
            text = data[modify_text_key]
            if tokenizer is not None and token_num is not None:
                text_id = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                text_id = text_id[:token_num]
                text = tokenizer.decode(text_id)
            if data["label"] != "benign":
                continue
            if modify_num >= 200:
                continue
            toxic_score = get_toxicity_score(classifier_name, classifier_tokenizer, classifier, text)
            if toxic_score <= 0.5 or water_score > 0.5:
                continue
            if text is None:
                continue
            detect_result = watermark_algo.trace_spoof(text)
            score = detect_result["neg_hit_ratio"]
            if not np.isnan(score):
                y_scores.append(score)
                y_true.append(original_label)
                modify_num += 1
        for original_text_key in original_text_key_list:
            text = data[original_text_key]
            if tokenizer is not None and token_num is not None:
                text_id = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                text_id = text_id[:token_num]
                text = tokenizer.decode(text_id)
            if data["label"] == "benign":
                continue
            if original_num >= 200:
                continue
            toxic_score = get_toxicity_score(classifier_name, classifier_tokenizer, classifier, text)
            if toxic_score <= 0.5 or water_score <= 0.5:
                continue
            if text is None:
                continue
            detect_result = watermark_algo.trace_spoof(text)
            score = detect_result["neg_hit_ratio"]
            if not np.isnan(score):
                y_scores.append(score)
                y_true.append(modify_label)
                original_num += 1

    auc = auc_score(y_true, y_scores)
    print("auc", auc)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_threshold = get_best_threshold(y_true, y_scores)
    print("best_threshold", best_threshold)

    print(modify_text_key_list, original_text_key_list)
    print("auc\tTP@1%\tTP@5%\tTP@10%")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(auc, tpr_at_fpr(fpr, tpr, 0.01),
                                                  tpr_at_fpr(fpr, tpr, 0.05),
                                                  tpr_at_fpr(fpr, tpr, 0.1)))
    print("-" * 30)

    if return_result:
        return fpr, tpr, auc
