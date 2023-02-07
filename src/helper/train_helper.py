from torch.nn.utils import clip_grad_norm_
import torch
import numpy as np
from sklearn.metrics import f1_score
import copy



def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask
    
def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos


def mean_recall_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def mean_ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    Mean NDCG @k : float
    """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)

def insert_list_in_list(s, t, p):
    return s[:p+1] + t + s[p+1:]
    
def trancate(data, max_len, pad):
    if len(data) >= max_len:
        return data[:max_len]
    else:
        data += [pad for x in range(max_len-len(data))]
        return data

def mask_trancate(data, max_len, pad):
    for i in range(max_len):
        try:
            data[i]+=[pad for x in range(max_len-len(data[i]))]
        except IndexError:
            data.append([pad for x in range(max_len)])
    return data

def delete_pad(data, pad=0):
    for i, b in enumerate(data):
        for j, d in enumerate(b):
            if d == pad:
                data[i] = data[i][:j]
    return data

def create_tgt_level_mask(seq, level):
    index = [x for x in range(len(level))]
    level_copy = copy.copy(level)
    tmp = []
    level_len = len(level)

    while len(level) > 0:
        t = [0 for x in range(level_len)]
        for i in range(len(level)):
            if level[i] == level[i+1] and level[i+1] == level[i+2]:
                level.pop(i)
                level.pop(i)
                level.pop(i)
                t[index.pop(i)] = 1
                t[index.pop(i)] = 1
                t[index.pop(i)] = 1
                break
        tmp.append(t)

    tmp = sorted(tmp, key=lambda x: x.index(1))
    tmp = [np.array(t) for t in tmp]

    result = []
    level = level_copy

    for s, l in zip(seq, level):
        l = l-1
        prev = np.zeros(len(level), dtype=int)
        if s == 2:
            for t in tmp[:l+1]:
                prev += t
            result.append(prev)
            tmp.pop(l)
        else:
            for t in tmp[:l+1]:
                prev += t
            result.append(prev)
    tgt_self_mask = [x.tolist() for x in result]
    return tgt_self_mask


def make_next_level_input(batch, preds, label2child, max_len, next_level):
    new_tgt_input_ids = delete_pad(batch["tgt_input_ids"].tolist())
    new_tgt_level_ids = delete_pad(batch["tgt_level_ids"].tolist())
    old_tgt_position = batch["tgt_position"].tolist()
    new_tgt_position = (batch["tgt_position"] & False).tolist()
    new_tgt_child = []
    new_tgt_child_num = []
    old_tgt_child = batch["tgt_child"].tolist()
    preds = preds.tolist()
    target_idx = -1

    for b_idx, tgt_position in enumerate(old_tgt_position):
        tmp = -1
        for i, position in enumerate(tgt_position):
            if position == False:
                continue
            target_idx += 1
            for j, pred in enumerate(preds[target_idx]):
                if pred == False:
                    continue
                child_idx = old_tgt_child[target_idx][j]
                if child_idx == 0:
                    break
                elif child_idx == 3: # assign [END] token
                    tmp+=1
                    new_tgt_input_ids[b_idx] = insert_list_in_list(new_tgt_input_ids[b_idx], [1, child_idx, 2], i+3*tmp)
                    new_tgt_level_ids[b_idx] = insert_list_in_list(new_tgt_level_ids[b_idx], [next_level, next_level, next_level], i+3*tmp)
                    new_tgt_position[b_idx] = insert_list_in_list(new_tgt_position[b_idx], [False, False, False], i+3*tmp)
                elif child_idx not in label2child: # when leaf node (leaf node of hierarchy dont have child)
                    tmp+=1
                    new_tgt_input_ids[b_idx] = insert_list_in_list(new_tgt_input_ids[b_idx], [1, child_idx, 1, 3, 2, 2], i+3*tmp)
                    new_tgt_level_ids[b_idx] = insert_list_in_list(new_tgt_level_ids[b_idx], [next_level, next_level, next_level+1, next_level+1, next_level+1, next_level], i+3*tmp)
                    new_tgt_position[b_idx] = insert_list_in_list(new_tgt_position[b_idx], [False, False, False, False, False, False], i+3*tmp)
                    tmp+=1
                else:
                    new_childs = label2child[child_idx]
                    new_childs += [0 for _ in range(max_len-len(new_childs))]
                    new_tgt_child.append(new_childs)
                    new_tgt_child_num.append(len(new_childs))
                    tmp+=1
                    new_tgt_input_ids[b_idx] = insert_list_in_list(new_tgt_input_ids[b_idx], [1, child_idx, 2], i+3*tmp)
                    new_tgt_level_ids[b_idx] = insert_list_in_list(new_tgt_level_ids[b_idx], [next_level, next_level, next_level], i+3*tmp)
                    new_tgt_position[b_idx] = insert_list_in_list(new_tgt_position[b_idx], [False, True, False], i+3*tmp)

    max_seq = max([len(s) for s in new_tgt_input_ids])
    try:
        new_tgt_self_level_mask = [mask_trancate(create_tgt_level_mask(i, copy.copy(l)), max_seq, False) for i, l in zip(new_tgt_input_ids, new_tgt_level_ids)]
    except:
        print(new_tgt_level_ids)
        print(new_tgt_input_ids)

    for i in range(len(new_tgt_input_ids)):
        new_tgt_input_ids[i] = trancate(new_tgt_input_ids[i], max_seq, 0)
        new_tgt_level_ids[i] = trancate(new_tgt_level_ids[i], max_seq, 0)
        new_tgt_position[i] = trancate(new_tgt_position[i], max_seq, 0)

    batch["tgt_input_ids"] = torch.LongTensor(new_tgt_input_ids)
    batch["tgt_level_ids"] = torch.LongTensor(new_tgt_level_ids)
    batch["tgt_position"] = torch.BoolTensor(new_tgt_position)
    batch["tgt_child"] = torch.LongTensor(new_tgt_child)
    batch["tgt_child_num"] = torch.LongTensor(new_tgt_child_num)
    batch["tgt_self_level_mask"] = torch.BoolTensor(new_tgt_self_level_mask)
    batch["t_mask"] = create_pad_mask(batch["tgt_input_ids"], 0)
    
    if len(new_tgt_child) == 0:
        flag = False
    else:
        flag = True

    return flag, batch

def extract_fine_grained_labels(sub_hierarchy):
    result = []
    sub_hierarchy = delete_pad(sub_hierarchy.tolist())
    for instance in sub_hierarchy:
        fine_grained_labels = []
        stop = 0
        while 3 in instance:
            for i, ids in enumerate(instance):
                # ( target ( [END] ) ) : a single child [END]
                if instance[i] == 1 and instance[i+2]==1 and instance[i+3]==3 and instance[i+4]==2 and instance[i+5]==2:
                    fine_grained_labels.append(instance[i+1])
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                # ( target ( [END] ) ( Other node ) ) : Multi children with [END]
                elif instance[i] == 1 and instance[i+2]==1 and instance[i+3]==3 and instance[i+4]==2 and instance[i+5]==1:
                    fine_grained_labels.append(instance[i+1])
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
            stop +=1
            if stop >10:
                print(f"error on extract lables : {instance}")
                break
        result.append(fine_grained_labels)
    return result

def extract_labels(sub_hierarchy, root=4):
    result = []
    for instance in sub_hierarchy:
        mask = instance == 0
        mask += instance == 1 
        mask += instance == 2
        mask += instance == 3
        mask += instance == root
        labels = instance[mask==False]
        result.append(labels.tolist())
    return result

def extract_coarse_grained_labels(labels, fine_grained_labels):
    result = []
    for label, fine_grained_label in zip(labels, fine_grained_labels):
        tmp = [i for i in label if i not in fine_grained_label]
        result.append(tmp)
    return result
