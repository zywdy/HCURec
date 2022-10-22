from sklearn.metrics import (
    roc_auc_score,
)
import numpy as np


def group_labels(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        all_labels: labels after group.
        all_preds: preds after group.
    """

    all_keys = list(set(group_keys))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, group_keys):
        group_labels[k].append(l)
        group_preds[k].append(p)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_labels, all_preds


def mrr_score(y_true, y_score):
    """Computing mrr score metric.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.

    FIXME:
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            tmp_labels, tmp_preds = [], []
            for l, p in zip(labels, preds):
                tmp_labels += l
                tmp_preds += p
            auc = roc_auc_score(np.asarray(tmp_labels), np.asarray(tmp_preds))
            res["auc"] = round(auc, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric == "group_auc":
            auc_list = []
            for each_labels, each_preds in zip(labels, preds):
                try:
                    x = roc_auc_score(each_labels, each_preds)
                    auc_list.append(x)
                except:
                    print("There are only zero labels")
                    auc_list.append(0.0)
            group_auc = np.mean(
                auc_list
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def ILAD(vecs):
    score = np.dot(vecs, vecs.T)
    score = (score + 1) / 2
    score = score.mean() - 1 / score.shape[0]
    score = float(score)
    return score


def ILMD(vecs):
    score = np.dot(vecs, vecs.T)
    score = (score + 1) / 2
    score = score.min()
    score = float(score)
    return score


def evaluate_density_ILxD(topk, rankings, news_scoring):
    ILADs = []
    ILMDs = []
    for each_preds, each_newsmeb in zip(rankings, news_scoring):
        nv = each_newsmeb

        score = each_preds

        top_docids = np.argsort(score)[-topk:]

        nv = np.array(nv).reshape(np.shape(nv)[0], np.shape(nv)[-1]) / np.sqrt(np.square(nv).sum(axis=-1))

        nv = nv[top_docids]
        ilad = ILAD(nv)
        ilmd = ILMD(nv)

        ILADs.append(ilad)
        ILMDs.append(ilmd)

    ILADs = np.array(ILADs).mean()
    ILMDs = np.array(ILMDs).mean()
    return ILADs, ILMDs


def evaluate_diversity_topic_norm(topk, rankings, co_verts):
    topics = []
    for each_preds, each_coverts in zip(rankings, co_verts):
        score = each_preds
        top_args = np.argsort(score)[-topk:]
        each_coverts = np.array(each_coverts)[top_args]

        s = 0
        for v in each_coverts:
            if v == 1:
                continue
            if v == 0:
                s += 1
        s /= (topk + 0.01)
        topics.append(s)
    topics = np.array(topics).mean()
    return topics


def evaluate_diversity_topic_all(TOP_DIVERSITY_NUM, rankings, co_verts, co_subverts):
    g3 = evaluate_diversity_topic_norm(TOP_DIVERSITY_NUM, rankings, co_verts)
    g4 = evaluate_diversity_topic_norm(TOP_DIVERSITY_NUM, rankings, co_subverts)

    metric = {
        'vert_norm_acc': g3,
        'subvert_norm_acc': g4,
    }

    return metric
