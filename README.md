# EEG_MATLAB
This repository contains code and related files for EEG spectral analysis, DA, SVM, RF model construction and visualization.
综合性能。“ROC曲线”是由一系列不同的模型分类阈值生成的，对于每一个阈值，模型的真阳性率（TPR）和假阳性率（FPR）都会被计算出来，并且以（FPR，TPR）为坐标画出56。其中真阳性率（TPR）同上述敏感性，假阳性率（FPR）通过下方“公式（6）”计算，假阳性率是指模型错误地将负类样本识别为正类样本的比例。AUC和上述5个指标的取值范围在0到1之间，越接近1表示模型性能越好。值得注意的是，若AUC为0.5，则说明模型的分类能力等同于随机猜测。
