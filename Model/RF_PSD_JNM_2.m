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
train_data   = train_psd(:,1:end-1);
test_data    = test_psd(:,1:end-1);
all_data     = [train_data;test_data];

%% 数据标准化、归一化
all_data          = zscore(all_data);
% [all_data,~] = mapminmax(all_data');
% all_data   = all_data';

% 整合训练集、测试集样本数及标签
n_train      = size(train_data,1);
n_test       = size(test_data,1);
all_labels = [train_labels;test_labels];

%% 降维
% PCA
% [~,score,latent1,~,explained] = PCA(all_data);
% idx = find(cumsum(explained)>95,1);
% all_data = score(:,1:idx);

% PLSR
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

%% 选择最佳决策树棵树与最小叶子数
mintree  = 10;  % 决策树最小棵数
maxtree  = 300; % 决策树最大棵数
tgap     = 10;  % 树间距
minleaf  = 1;   % 最小叶子数下限
maxleaf  = 20;  % 最小叶子数上限
NumPrint = 10;  % 模型运行进度
G = 1; % 模型重复运行K-fold CV次数
K = 5; % K-fold CV
Accuracy = zeros(maxleaf,(maxtree - mintree)/tgap + 1);
Test_Scores = zeros(maxleaf,(maxtree - mintree)/tgap + 1,size(train_data,1)/K,2,K,G);

for times = 1:G
    indices = crossvalind('Kfold',train_labels,K); % generate indices for CV
    for i = mintree:tgap:maxtree
        for j = minleaf:maxleaf
            for k = 1:K % K-fold CV
                cv_test_idx = find(indices == k); % indices for test samples in validation
                cv_train_idx = find(indices ~= k); % indices for training sample validation
                cv_train = train_data(cv_train_idx,:); % 训练集（交叉验证）
                cv_validation = train_data(cv_test_idx,:); % 验证集（交叉验证）
                cv_TrainLabels = train_labels(cv_train_idx); % 训练集标签（交叉验证）
                cv_TestLabels(:,k,times) = train_labels(cv_test_idx); % 验证集集标签（交叉验证）

                mdl = TreeBagger(i, cv_train, cv_TrainLabels,'minleaf', j, 'NumPrint', NumPrint);

                %  仿真测试
                [t_sim(:,k,times),val_scores(:,:,k,times)] = predict(mdl, cv_validation);

                % 格式转换
                predict_label(:,k,times) = str2double(t_sim(:,k,times));
                cv_acc(k,times) = mean(predict_label(:,k,times) == cv_TestLabels(:,k,times));
            end
            Accuracy(j,(i - mintree)/tgap + 1) = mean(cv_acc,"all");
            Test_Scores(j,(i - mintree)/tgap + 1,:,:,:,times) = val_scores(:,:,:,times);
        end    
    end
end
% 最小叶子数*决策树棵树*被试*正反例概率*交叉验证折序号*模型重复交叉验证次数
Prob = squeeze(Test_Scores(:,:,:,2,:,:));
[r,c] = size(Accuracy);
x = 1:c;
y = 1:r;
[X,Y] = meshgrid(x,y);
Y = flipud(Y);
AUC = zeros(r,c,K,G);

for times = 1:G
    for t = 1:c
        for p = 1:r
            for k = 1:K
                [~, ~, ~, AUC(p,t,k,times)] = perfcurve(squeeze(cv_TestLabels(:,k,times)),...
                    squeeze(Prob(p,t,:,k,times)), 1);
            end
        end
    end
end
AUC = mean(AUC,[3,4]);

Accuracy = flipud(Accuracy);
AUC = flipud(AUC);
object = AUC + Accuracy;

% 找到目标矩阵的最大值
maxval = max(object,[],"all");

% 找到最大值的行列索引
[rind,cind] = find(object == maxval);
disp(['最佳观测点的最小叶子数为', num2str(maxleaf - rind + 1), ','...
    '决策树颗树为', num2str(cind*tgap)]);

%% 决策树棵树与最小叶子数对性能的影响(气泡矩阵图)
mycolor = othercolor('BrBG4');
% mycolor = othercolor('RdBu12');
% mycolor = slanCM(1);

figure
bc = bubblechart(X(:),Y(:),Accuracy(:),AUC(:));
bc.MarkerFaceAlpha = 1;
bc.MarkerEdgeColor = 'none';
colormap(mycolor)
h = colorbar('Position',[0.757 0.14 0.025 0.5]);
h.Label.String = 'Area under the Curve (AUC)';
h.Label.FontName = en;
h.Label.FontSize = 12;
% clim([0.9 1])
blgd = bubblelegend('Accuracy','Box','off','FontName',en,'FontSize',10);
blgd.Location = "northeastoutside";
% bubblelim([0 1])
bubblesize([5 10])
blgd.FontName = en;
blgd.FontSize = 12;
blgd.LineWidth = 0.75;
ax = gca;
ax.LineWidth = 1;
ax.XLim = [0.5 c+0.5];
ax.XTick = 0.5:c+0.5;
temp1 = c+1;
for ii = 1:c+1
    if ii == temp1
        ax.XTickLabel{ii} = '';
    else
        ax.XTickLabel{ii} = num2str(ii*tgap);
    end
end
ax.YLim = [0.5 r+0.5];
ax.YTick = 0.5:r+0.5;
temp2 = r+1;
for jj = 1:r+1
    if jj == temp2
        ax.YTickLabel{jj} = '';
    else
        ax.YTickLabel{jj} = num2str(jj);
    end
end
ax.GridLineWidth = 1;
% ax.GridAlpha = 0.2;
ax.TickDir = "none";
ax.Color = [0.94 0.94 0.94];
ax.FontName = en;
ax.FontSize = 12;
% ax.XColor = [0.5 0.5 0.5];
% ax.YColor = [0.5 0.5 0.5];
% ax.XAxis.TickLabelColor = [0 0 0] ;
% ax.YAxis.TickLabelColor = [0 0 0];
% ax.XLabel.Color = [0 0 0];
% ax.YLabel.Color = [0 0 0];
xlabel('The number of decision trees in random forest','FontName',en,'FontSize',14);
ylabel('Minimum number of leaves','FontName',en,'FontSize',14);
grid on

%% RF(随机森林)模型最优超参数
trees = 20;                                       % 决策树数目
leaf  = 13;                                        % 最小叶子数
OOBPrediction = 'on';                              % 计算预测概率
OOBPredictorImportance = 'on';                     % 计算特征重要性
Method   = 'classification';                       % 分类/回归
NumPrint = 50;

%% 最优超参数重复运行模型N次
N = 1;
err = zeros(trees,N);
for N = 1:1  
    model = TreeBagger(trees, train_data, train_labels, 'OOBPredictorImportance', ...
        OOBPredictorImportance, 'Method', Method, 'OOBPrediction', OOBPrediction, ...
        'minleaf', leaf,'NumPrint', NumPrint);

    % 仿真测试
    [t_sim1,train_scores(:,:,N)] = predict(model, train_data);
    [t_sim2,test_scores(:,:,N)]  = predict(model, test_data);
    train_votes(:,:,N) = train_scores(:,:,N)*trees;
    test_votes(:,:,N)  = test_scores(:,:,N)*trees;

    % 格式转换
    predict_label_1 = str2double(t_sim1);
    predict_label_2 = str2double(t_sim2);

    % Test on test data
    TP(N)   = sum((predict_label_2 == test_labels) & (predict_label_2 == 1));
    TN(N)   = sum((predict_label_2 == test_labels) & (predict_label_2 == 0));
    FP(N)   = sum((predict_label_2 ~= test_labels) & (predict_label_2 == 1));
    FN(N)   = sum((predict_label_2 ~= test_labels) & (predict_label_2 == 0));
    train_acc(N) = mean(predict_label_1 == train_labels); % 计算训练集准确率
    test_acc(N)  = mean(predict_label_2 == test_labels); % 计算测试集准确率
    test_sensitivity(N) = TP/(TP + FN); % compute specificity
    test_specificity(N) = TN/(TN + FP); % compute sensitivity
    err(:,N) = oobError(model);
end

% 求各个指标的平均值
train_acc_avg    = mean(train_acc);
acc_RF2           = mean(test_acc);
precision_RF2   = TP/(TP+FP);
sensitivity_RF2   = mean(test_sensitivity);
specificity_RF2   = mean(test_specificity);
train_scores_avg = mean(train_scores,3);
train_votes_avg  = round(mean(train_votes,3));
test_scores_avg  = mean(test_scores,3);
test_votes_avg   = round(mean(test_votes,3));
err_avg          = mean(err,2);
F1_score_RF2  = 2*(precision_RF2*sensitivity_RF2/(precision_RF2+sensitivity_RF2));
% 预测正类概率
prob = test_scores_avg(:,2);

% 计算AUC
[XX, YY, ~, auc_RF2] = perfcurve(test_labels, prob, 1);
disp(['Train_acc = ' num2str(train_acc_avg*100) '%' ...
    '（' num2str(sum(predict_label_1 == train_labels)) '/' ...
    num2str(length(train_labels)) '）' '  -->训练集'])
disp(['Test_acc = ' num2str(acc_RF2*100) '%' ...
    '（' num2str(sum(predict_label_2 == test_labels)) '/' ...
    num2str(length(test_labels)) '）' '  -->测试集'])
disp(['Test_Sensitivity = ' num2str(sensitivity_RF2*100) '%'])
disp(['Test_Specificity = ' num2str(specificity_RF2*100) '%'])

save('test_data2.mat','acc_RF2','sensitivity_RF2','specificity_RF2','auc_RF2','F1_score_RF2', ...
    'precision_RF2','-append')

%% 绘制ROC曲线
figure;
plot(XX, YY, '-', 'LineWidth', 1.5,'Color',[0.15,0.5,0.5]);
ax = gca;
ax.LineWidth = 1;
ax.FontSize  = 12;
ax.FontName  = en;
xlabel('False Positive Rate', 'fontsize', 14, 'FontName', en);
ylabel('True Positive Rate', 'fontsize', 14, 'FontName', en);
title('Receiver Operating Characteristic (ROC) Curve', ...
    'fontsize', 14, 'FontName', en);
patch([XX;1],[YY;0], [0.15,0.5,0.5], 'EdgeColor', 'none', 'linestyle','none','FaceAlpha',0.5);
xlim([-0.05 1.05])
ylim([-0.05 1.05])
xticks(0:0.2:1)
yticks(0:0.2:1)
box on

% 添加对角线
hold on;
plot([0 1], [0 1], '--','Color',[0.4 0.4 0.4],'LineWidth',1);

% 显示图例和AUC值
lgd = legend('ROC Curve', ['AUC = ' num2str(auc_RF2) ''], ...
    'Random Guessing', 'fontsize', 9, 'FontName', en,'Location','southeast');
lgd.LineWidth = 0.75;
hold off
% save("RF_Data2.mat")