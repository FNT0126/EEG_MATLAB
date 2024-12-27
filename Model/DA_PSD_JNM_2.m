%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Written by Jiayi Fang                                                                     %%%%%
%%%% School of Perfume and Aroma Technology, Shanghai Institute of Technology                  %%%%%
%%%% Date: 2024/03/31                                                                          %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 清除变量
clear;clc;

%% 导入数据
load all_psd.mat

%% 设置字体
zh = '宋体'; % 中文
en = 'Arial'; % 英文

%% 随机划分训练集与测试集
n            = size(all_psd,1); % 总样本数
m            = size(all_psd,2)-1; % 特征数
M            = m/7; % 分析脑部区域个数
tx_psd       = all_psd(1:n/2,:);
fsg_psd      = all_psd(n/2+1:end,:);
random_idx1  = randperm(n/2);
random_idx2  = randperm(n/2);
train_psd1   = tx_psd(random_idx1(1:n*4/5/2),1:end);
test_idx1    = setdiff(1:n/2,random_idx1(1:n*4/5/2));
test_psd1    = tx_psd(test_idx1,1:end);
train_psd2   = fsg_psd(random_idx1(1:n*4/5/2),1:end);
test_idx2    = setdiff(1:n/2,random_idx2(1:n*4/5/2));
test_psd2    = fsg_psd(test_idx2,1:end);
train_psd    = [train_psd1;train_psd2];
test_psd     = [test_psd1;test_psd2];
train_labels = train_psd(:,end);
test_labels  = test_psd(:,end); % labels of all samples: 0 for TX; 1 for FSG
train_data    = train_psd(:,1:end-1);
test_data     = test_psd(:,1:end-1);
all_data     = [train_data;test_data];

%% 数据标准化、归一化
all_data     = zscore(all_data);
% [all_data,~] = mapminmax(all_data');
% all_data     = all_data';

% 整合训练集、测试集样本数及标签
n_train      = size(train_data,1);
n_test       = size(test_data,1);
all_labels   = [train_labels;test_labels];

%% PLSR(偏最小二乘回归)
% 1.1 创建模型
sumpct2 = 0; % 初始化主成分总贡献率
s = size(all_data,2); % 主成分数
[~,~,Xscores,~,~,pctvar,~,~] = plsregress(all_data,all_labels,s);
[pctvar,Ip] = sort(pctvar,2,'descend');
for i = 1:s
    sumpct2 = sumpct2 + pctvar(1,i);
    if sumpct2 > 0.9500
        break
    end
end
idx = find(cumsum(pctvar(1,:))>0.95,1);
disp(['前 ' num2str(idx) ' 个主成分的总体解释方差为 ' num2str(sumpct2*100) '%'])

% 整合降维后所有数据及其正类样本和负类样本
all_data   = Xscores(:,Ip(1:idx));
train_data = all_data(1:n_train,:);
test_data  = all_data(1:n_test,:);

%% K-fold CV on all data(Repeat G times)
K = 5; % K-fold CV
G = 1; % K-fold for G times
for times = 1:G
    indices = crossvalind('Kfold',train_labels,K); % generate indices for CV
    for k = 1:K % K iterations
        cv_test_idx = find(indices == k); % indices for test samples in validation
        cv_train_idx = find(indices ~= k); % indices for training samples in validation
        cv_train = train_data(cv_train_idx,:); % 训练集（交叉验证）
        cv_validation = train_data(cv_test_idx,:); % 验证集（交叉验证）
        cv_TrainLabels = train_labels(cv_train_idx); % 训练集标签（交叉验证）
        cv_TestLabels(:,k,times) = train_labels(cv_test_idx); % 验证集集标签（交叉验证）
        [cv_class1,~,~] = classify(cv_train, cv_train, cv_TrainLabels, ...
            'diagLinear');
        [cv_class2,~,val_scores(:,:,k,times)] = classify(cv_validation, ...
            cv_train, cv_TrainLabels, 'diagLinear');
        cv_acc_train(k,times) = mean(cv_class1 == cv_TrainLabels);
        cv_acc(k,times) = mean(cv_class2 == cv_TestLabels(:,k,times)); % compute accuracy
        TP = sum((cv_class2 == cv_TestLabels(:,k,times)) & (cv_class2 == 1));
        TN = sum((cv_class2 == cv_TestLabels(:,k,times)) & (cv_class2 == 0));
        FP = sum((cv_class2 ~= cv_TestLabels(:,k,times)) & (cv_class2 == 1));
        FN = sum((cv_class2 ~= cv_TestLabels(:,k,times)) & (cv_class2 == 0));
        cv_sensitivity(k,times) = TP/(TP+FN); % compute specificity
        cv_specificity(k,times) = TN/(TN+FP); % compute sensitivity
    end
    TEST_Scores(:,:,:,times) = val_scores(:,:,:,times);
end

PROB = squeeze(TEST_Scores(:,2,:,:));
for times = 1:G
    for k = 1:K
        [x, y, ~,auc(k,times)] = perfcurve(cv_TestLabels(:,k,times), ...
            PROB(:,k,times), 1);
    end
end
auc_avg_LDA2 = mean(auc);
auc_std_LDA2 = std(auc);
acc_std_LDA2 = std(cv_acc);
cv_acc_Train_LDA2  = mean(cv_acc_train);
cv_acc_LDA2  = mean(cv_acc);
cv_sensitivity_LDA2  = mean(cv_sensitivity);
cv_specificity_LDA2  = mean(cv_specificity);
sensitivity_std_LDA2 = std(cv_sensitivity);
specificity_std_LDA2 = std(cv_specificity);
% temp(q) = cv_acc_LDA;
% end
% acc_avg = mean(temp);
save('cv_data2.mat','cv_acc_LDA2','cv_sensitivity_LDA2','cv_specificity_LDA2','cv_acc_Train_LDA2', ...
    'acc_std_LDA2','sensitivity_std_LDA2','specificity_std_LDA2','auc_avg_LDA2','auc_std_LDA2','-append')

%% LDA仿真测试(只要数据集及参数不变，则预测结果不变)
[predict_label_1, ~, ~] = classify(train_data,train_data,train_labels,"diagLinear");
[predict_label_2, err, posterior] = classify(test_data,train_data,train_labels,"diagLinear");
tp = sum((predict_label_2 == test_labels) & (predict_label_2 == 1));
tn = sum((predict_label_2 == test_labels) & (predict_label_2 == 0));
fp = sum((predict_label_2 ~= test_labels) & (predict_label_2 == 1));
fn = sum((predict_label_2 ~= test_labels) & (predict_label_2 == 0));
precision_LDA2   = tp/(tp+fp);
sensitivity_LDA2 = tp/(tp+fn);
specificity_LDA2 = tn/(tn+fp);
train_acc = mean(predict_label_1 == train_labels);
acc_LDA2   = mean(predict_label_2 == test_labels);
F1_score_LDA2  = 2*(precision_LDA2*sensitivity_LDA2/(precision_LDA2+sensitivity_LDA2));

% 预测正类概率
prob = posterior(:,2);

% 计算AUC
[X, Y, ~, auc_LDA2] = perfcurve(test_labels, prob, 1);
disp(['CV_Accuracy_Train = ' num2str(cv_acc_Train_LDA2*100) '%'])
disp(['CV_Accuracy_Test = ' num2str(cv_acc_LDA2*100) '%'])
disp(['Accuracy = ' num2str(train_acc*100) '%' ...
    '（' num2str(sum(predict_label_1 == train_labels)) '/' ...
    num2str(length(train_labels)) '）' '  -->训练集'])
disp(['Accuracy = ' num2str(acc_LDA2*100) '%' ...
    '（' num2str(sum(predict_label_2 == test_labels)) '/' ...
    num2str(length(test_labels)) '）' '  -->测试集'])
disp(['Test_Sensitivity = ' num2str(sensitivity_LDA2*100) '%'])
disp(['Test_Specificity = ' num2str(specificity_LDA2*100) '%'])

save('test_data2.mat','acc_LDA2','sensitivity_LDA2','specificity_LDA2','auc_LDA2','F1_score_LDA2', ...
    'precision_LDA2','-append')

%% 绘图
% figure
% plot(1:length(test_labels), test_labels, '*', 'Color', [0.85,0.33,0.10], LineWidth=1)
% hold on
% plot(1:length(test_labels), predict_label_2, 'o', 'Color', [0 0.4470 0.7410], LineWidth=1)
% yticks([0 1])
% grid on
% lgd = legend('真实类别','预测类别', 'fontsize', 9, 'FontName', zh);
% lgd.LineWidth = 0.75;
% ax = gca;
% ax.LineWidth = 1;
% ax.FontSize  = 12;
% ax.FontName  = en;
% xlabel('测试集样本编号', 'fontsize', 14, 'FontName', zh)
% ylabel('测试集样本类别', 'fontsize', 14, 'FontName', zh)
% string = {'(LDA)测试集预测结果' ['Accuracy = ' num2str(test_acc * 100,'%.2f') '%']};
% title(string, 'fontsize', 14, 'FontName', zh)
% hold off
% 
% 混淆矩阵
% 训练集
figure
cm = confusionchart(train_labels, predict_label_1);
cm.Title = 'Confusion Matrix for Train Set';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.FontSize = 14;
cm.FontName = en;
cm.XLabel   = 'Predicted Class';
cm.YLabel   = 'True Class';

% 测试集
figure
cm = confusionchart(test_labels, predict_label_2);
cm.Title = 'Confusion Matrix for Test Set';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.FontSize = 14;
cm.FontName = en;
cm.XLabel   = 'Predicted Class';
cm.YLabel   = 'True Class';

%% 绘制ROC曲线
figure;
plot(X, Y, '-', 'LineWidth', 1.5,'Color',[0.15,0.5,0.5]);
ax = gca;
ax.LineWidth = 1;
ax.FontSize  = 12;
ax.FontName  = en;
xlabel('False Positive Rate', 'fontsize', 14, 'FontName', en,'Color','k');
ylabel('True Positive Rate', 'fontsize', 14, 'FontName', en,'Color','k');
title('Receiver Operating Characteristic (ROC) Curve', ...
    'fontsize', 14, 'FontName', en);
p = patch([X;1],[Y;0], [0.15,0.5,0.5], 'EdgeColor', 'none', 'linestyle','none','FaceAlpha',0.5);
xlim([-0.05 1.05])
ylim([-0.05 1.05])
xticks(0:0.2:1)
yticks(0:0.2:1)
box on

% 添加对角线
hold on;
plot([0 1], [0 1], '--','Color',[0.4 0.4 0.4],'LineWidth',1);
hold off

% 显示图例和AUC值
lgd = legend('ROC Curve', ['AUC = ' num2str(auc_LDA2) ''], 'Random Guessing', ...
    'fontsize', 9, 'FontName', en,'Location','southeast','Color','w');
lgd.LineWidth = 0.75;

