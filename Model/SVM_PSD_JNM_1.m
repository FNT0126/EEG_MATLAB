%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Written by Jiayi Fang                                                                     %%%%%
%%%% School of Perfume and Aroma Technology, Shanghai Institute of Technology                  %%%%%
%%%% Date: 2024/03/09                                                                          %%%%%
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
all_data          = zscore(all_data);
[all_data,~] = mapminmax(all_data');
all_data   = all_data';

% 整合训练集、测试集样本数及标签
n_train      = size(train_data,1);
n_test       = size(test_data,1);
all_labels = [train_labels;test_labels];

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
all_data = Xscores(:,Ip(1:idx));
train_data = all_data(1:n_train,:);
test_data  = all_data(1:n_test,:);

%% SVM创建/训练(RBF核函数)
% 寻找最佳c/g参数
[cg,Bestacc,Bestc,Bestg] = SVMcgForClass(train_labels,train_data,...
    -8,8,-8,8,5,1,1,4);
% function [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,...
% cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
cmd = ['-b 1', ' -t 2 ', ' -c ', num2str(Bestc), ' -g ', num2str(Bestg)];

%% SVM仿真测试(只要数据集及参数不变，则预测结果不变)
model = libsvmtrain(train_labels, train_data,cmd);
[predict_label_1, accuracy_1, prob_A] = libsvmpredict(...
    train_labels, train_data, model,'-b 1');
[predict_label_2, accuracy_2, prob_B] = libsvmpredict(...
    test_labels, test_data, model,'-b 1');
acc1 = accuracy_1(1); % 训练集
acc2 = accuracy_2(1); % 测试集
acc_SVM = acc2/100;
tp = sum((predict_label_2 == test_labels) & (predict_label_2 == 1));
tn = sum((predict_label_2 == test_labels) & (predict_label_2 == 0));
fp = sum((predict_label_2 ~= test_labels) & (predict_label_2 == 1));
fn = sum((predict_label_2 ~= test_labels) & (predict_label_2 == 0));
precision_SVM   = tp/(tp+fp);
sensitivity_SVM = tp/(tp+fn); % compute specificity
specificity_SVM = tn/(tn+fp); % compute sensitivity
F1_score_SVM  = 2*(precision_SVM*sensitivity_SVM/(precision_SVM+sensitivity_SVM));

% 预测正类概率
prob = prob_B(:,2);

% 计算AUC
[X, Y, ~, auc_SVM] = perfcurve(test_labels, prob, 1);

disp(['Accuracy = ' num2str(acc1) '%' ...
    '（' num2str(sum(predict_label_1 == train_labels)) '/' ...
    num2str(length(train_labels)) '）' '  -->训练集'])
disp(['Accuracy = ' num2str(acc2) '%' ...
    '（' num2str(sum(predict_label_2 == test_labels)) '/' ...
    num2str(length(test_labels)) '）' '  -->测试集'])
disp(['Test_Sensitivity = ' num2str(sensitivity_SVM*100) '%'])
disp(['Test_Specificity = ' num2str(specificity_SVM*100) '%'])

save('test_data.mat','acc_SVM','sensitivity_SVM','specificity_SVM','auc_SVM','F1_score_SVM', ...
    'precision_SVM','-append')

%% 绘制ROC曲线
figure;
plot(X, Y, '-', 'LineWidth', 1.5,'Color',[0.15,0.5,0.5]);
ax = gca;
ax.LineWidth = 1;
ax.FontName  = en;
ax.FontSize  = 12;
xlabel('False Positive Rate', 'fontsize', 14, 'FontName', en,'Color','k');
ylabel('True Positive Rate', 'fontsize', 14, 'FontName', en,'Color','k');
title('Receiver Operating Characteristic (ROC) Curve', ...
    'fontsize', 14, 'FontName', en);
patch([X;1],[Y;0], [0.15,0.5,0.5], 'EdgeColor', 'none', 'linestyle','none','FaceAlpha',0.5);
xlim([-0.05 1.05])
ylim([-0.05 1.05])
xticks(0:0.2:1)
yticks(0:0.2:1)
box on

% 添加对角线
hold on;
plot([0 1], [0 1], '--','Color',[0.4 0.4 0.4],'LineWidth',1);

% 显示图例和AUC值
lgd = legend('ROC Curve', ['AUC = ' num2str(auc_SVM) ''], 'Random Guessing', ...
    'fontsize', 9, 'FontName', en,'Position',[0.63 0.14 0.25 0.12]);
lgd.LineWidth = 0.75;
hold off