import numpy as np

def compute_recall_ap(tps, fps, gt_num):
    '''
    According to COCO evaluation, the function computes instance level recall and average precision.
    tps: T x N
    fps: T x N
    gt_num:
    '''
    tps, fps = np.array(tps), np.array(fps)
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
    T = len(tp_sum)
    recall, precision = np.ones(T), np.ones(T)
    # 100 equal divide for recall 0~1
    recThrs = np.linspace(0, 1, int((1 - 0) / 0.01) + 1)
    R = len(recThrs)
    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
        tp = np.array(tp)
        fp = np.array(fp)
        nd = len(tp)
        rc = tp / gt_num
        pr = tp / (fp + tp + np.spacing(1))
        q = np.zeros((R,))

        if nd:
            recall[t] = rc[-1]
        else:
            recall[t] = 0

        pr = pr.tolist();
        q = q.tolist()

        for i in range(nd - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        inds = np.searchsorted(rc, recThrs, side='left')
        inds[inds == len(pr)] = len(pr) - 1  # for those points whose recall larger than rc[-1]
        try:
            for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
        except:
            pass
        precision[t] = np.mean(np.array(q))

    return recall, precision