import numpy as np

def ACC(ground_truth, predictions):
    
    return np.mean(ground_truth.astype(int) == predictions.astype(int))

def RMSE(ground_truth, predictions):
    
    return np.sqrt(np.mean((ground_truth-predictions)**2))

def SAGR(ground_truth, predictions):
    
    return np.mean(np.sign(ground_truth) == np.sign(predictions))

def PCC(ground_truth, predictions):
    
    return np.corrcoef(ground_truth, predictions)[0,1]

def CCC(ground_truth, predictions):
    
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)

def ICC(labels, predictions):
    
    naus = predictions.shape[1]
    icc = np.zeros(naus)

    n = predictions.shape[0]

    for i in range(0,naus):
        a = np.asmatrix(labels[:,i]).transpose()
        b = np.asmatrix(predictions[:,i]).transpose()
        dat = np.hstack((a, b))
        mpt = np.mean(dat, axis=1)
        mpr = np.mean(dat, axis=0)
        tm  = np.mean(mpt, axis=0)
        BSS = np.sum(np.square(mpt-tm))*2
        BMS = BSS/(n-1)
        RSS = np.sum(np.square(mpr-tm))*n
        tmp = np.square(dat - np.hstack((mpt,mpt)))
        WSS = np.sum(np.sum(tmp, axis=1))
        ESS = WSS - RSS
        EMS = ESS/(n-1)
        icc[i] = (BMS - EMS)/(BMS + EMS)

    return icc
