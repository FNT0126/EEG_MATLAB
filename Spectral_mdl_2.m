%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Optimized by Jiayi Fang                                                                   %%%%%
%%%% School of Perfume and Aroma Technology, Shanghai Institute of Technology                  %%%%%
%%%% Date: 2024/02/27                                                                          %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 清除变量
clear;clc;

%% 设置字体
zh = '宋体'; % 中文
en = 'Times New Roman'; % 英文

%% 声明各组数据所在文件夹
group1_dir = 'D:\file\eegdata\clean_data\1_REST';
group2_dir = 'D:\file\eegdata\clean_data\3_TX';
group3_dir = 'D:\file\eegdata\clean_data\6_FSG';

% 提取各组数据的文件信息
group1_files = dir([group1_dir, filesep, '*.set']);
group2_files = dir([group2_dir, filesep, '*.set']);
group3_files = dir([group3_dir, filesep, '*.set']);

% 提取各组数据的文件名并按数字大小顺序排列
group1_files_name = {group1_files.name};
group2_files_name = {group2_files.name};
group3_files_name = {group3_files.name};

for s = 1:length(group1_files)
    default_idx(s) = str2double(group1_files_name{s}(1:end-9)); % 文件全名1_rest.set为例(end-9)
end

[~,file_name_idx] = sort(default_idx);

for r = 1:length(group1_files)
    Group1_files_name{r} = group1_files_name{file_name_idx(r)};
    Group2_files_name{r} = group2_files_name{file_name_idx(r)};
    Group3_files_name{r} = group3_files_name{file_name_idx(r)};
end

fs    = 512; % sampling rate
L     = 2000; % window length (ms)
dt    = 1000/fs; % time resolution (ms)
t     = 0:dt:L-dt; % time points (ms), starting from 0
tindx = t/dt+1; % time index
NFFT  = pow2(nextpow2(fs))*2;
f     = linspace(0, fs/2, NFFT/2+1); % frequency axis
f_idx = find(f>=1 & f<=30);  % Frequency region of interest
win   = hann(length(t)); % window type: rectwin,hann,hamming
S2    = sum(win.^2); % normalizating factor 2

%% welch法估算PSD
for n = 1:length(group1_files)
    %提取当前被试数据文件名
    subj_fn1 = Group1_files_name{n};
    % 读取当前被试预处理后的数据
    EEG_new1 = pop_loadset('filename',subj_fn1,'filepath',group1_dir);
    for chanindx1 = 1:size(EEG_new1.data, 1) % channel
        [psd1(n,chanindx1,:),f] = pwelch(detrend(EEG_new1.data(chanindx1,:)),win,512,NFFT,fs); 
    end
end
psd1 = double(psd1);

for m = 1:length(group2_files)
    %提取当前被试数据文件名
    subj_fn2 = Group2_files_name{m};
    % 读取当前被试预处理后的数据
    EEG_new2 = pop_loadset('filename',subj_fn2,'filepath',group2_dir);
    for chanindx2 = 1:size(EEG_new2.data, 1) % channel
        [psd2(m,chanindx2,:),f] = pwelch(detrend(EEG_new2.data(chanindx2,:)),win,512,NFFT,fs); 
    end
end
psd2 = double(psd2);

for z = 1:length(group3_files)
    %提取当前被试数据文件名
    subj_fn3 = Group3_files_name{z};
    % 读取当前被试预处理后的数据
    EEG_new3 = pop_loadset('filename',subj_fn3,'filepath',group3_dir);
    for chanindx3 = 1:size(EEG_new3.data, 1) % channel
        [psd3(z,chanindx3,:),f] = pwelch(detrend(EEG_new3.data(chanindx3,:)),win,512,NFFT,fs); 
    end
end
% psd3 = 10*log10(psd3);
psd3 = double(psd3);

%% 导入电极信息
EEG_chanlocs1 = EEG_new1.chanlocs;
EEG_chanlocs2 = EEG_new2.chanlocs;
EEG_chanlocs3 = EEG_new3.chanlocs;

%% 定义感兴趣的频段
delta_idx  = find(f>=1 & f<=4);
theta_idx  = find(f>4 & f<=8);
alpha1_idx = find(f>8 & f<=10);
alpha2_idx = find(f>10 & f<=13);
alpha_idx  = find(f>8 & f<=13);
beta_idx   = find(f>13 & f<=30);
beta1_idx  = find(f>13 & f<=20);
beta2_idx  = find(f>20 & f<=30);
all_idx = {delta_idx,theta_idx,alpha1_idx,alpha2_idx,beta1_idx,beta2_idx};

%% Spectrum
% frequencies to plot
f_plot = f(f_idx);

% 不同频率点的单个通道或脑区平均
% 组1
f_psd1_F_d = squeeze(mean(psd1(:, 1:12, delta_idx), 2));
f_psd1_F_t = squeeze(mean(psd1(:, 1:12, theta_idx), 2));
f_psd1_F_a1 = squeeze(mean(psd1(:, 1:12, alpha1_idx), 2));
f_psd1_F_a2 = squeeze(mean(psd1(:, 1:12, alpha2_idx), 2));
f_psd1_F_b1 = squeeze(mean(psd1(:, 1:12, beta1_idx), 2));
f_psd1_F_b2 = squeeze(mean(psd1(:, 1:12, beta2_idx), 2));

f_psd1_T_d = squeeze(mean(psd1(:, [13 17], delta_idx), 2));
f_psd1_T_t = squeeze(mean(psd1(:, [13 17], theta_idx), 2));
f_psd1_T_a1 = squeeze(mean(psd1(:, [13 17], alpha1_idx), 2));
f_psd1_T_a2 = squeeze(mean(psd1(:, [13 17], alpha2_idx), 2));
f_psd1_T_b1 = squeeze(mean(psd1(:, [13 17], beta1_idx), 2));
f_psd1_T_b2 = squeeze(mean(psd1(:, [13 17], beta2_idx), 2));

f_psd1_C_d = squeeze(mean(psd1(:, [14:16,18:21], delta_idx), 2));
f_psd1_C_t = squeeze(mean(psd1(:, [14:16,18:21], theta_idx), 2));
f_psd1_C_a1 = squeeze(mean(psd1(:, [14:16,18:21], alpha1_idx), 2));
f_psd1_C_a2 = squeeze(mean(psd1(:, [14:16,18:21], alpha2_idx), 2));
f_psd1_C_b1 = squeeze(mean(psd1(:, [14:16,18:21], beta1_idx), 2));
f_psd1_C_b2 = squeeze(mean(psd1(:, [14:16,18:21], beta2_idx), 2));

f_psd1_P_d = squeeze(mean(psd1(:, 22:27, delta_idx), 2));
f_psd1_P_t = squeeze(mean(psd1(:, 22:27, theta_idx), 2));
f_psd1_P_a1 = squeeze(mean(psd1(:, 22:27, alpha1_idx), 2));
f_psd1_P_a2 = squeeze(mean(psd1(:, 22:27, alpha2_idx), 2));
f_psd1_P_b1 = squeeze(mean(psd1(:, 22:27, beta1_idx), 2));
f_psd1_P_b2 = squeeze(mean(psd1(:, 22:27, beta2_idx), 2));

f_psd1_O_d = squeeze(mean(psd1(:, 28:30, delta_idx), 2));
f_psd1_O_t = squeeze(mean(psd1(:, 28:30, theta_idx), 2));
f_psd1_O_a1 = squeeze(mean(psd1(:, 28:30, alpha1_idx), 2));
f_psd1_O_a2 = squeeze(mean(psd1(:, 28:30, alpha2_idx), 2));
f_psd1_O_b1 = squeeze(mean(psd1(:, 28:30, beta1_idx), 2));
f_psd1_O_b2 = squeeze(mean(psd1(:, 28:30, beta2_idx), 2));

f_psd1_d = {f_psd1_F_d,f_psd1_T_d,f_psd1_C_d,f_psd1_P_d,f_psd1_O_d};
f_psd1_t = {f_psd1_F_t,f_psd1_T_t,f_psd1_C_t,f_psd1_P_t,f_psd1_O_t};
f_psd1_a1 = {f_psd1_F_a1,f_psd1_T_a1,f_psd1_C_a1,f_psd1_P_a1,f_psd1_O_a1};
f_psd1_a2 = {f_psd1_F_a2,f_psd1_T_a2,f_psd1_C_a2,f_psd1_P_a2,f_psd1_O_a2};
f_psd1_b1 = {f_psd1_F_b1,f_psd1_T_b1,f_psd1_C_b1,f_psd1_P_b1,f_psd1_O_b1};
f_psd1_b2 = {f_psd1_F_b2,f_psd1_T_b2,f_psd1_C_b2,f_psd1_P_b2,f_psd1_O_b2};

% 组2
f_psd2_F_d = squeeze(mean(psd2(:, 1:12, delta_idx), 2));
f_psd2_F_t = squeeze(mean(psd2(:, 1:12, theta_idx), 2));
f_psd2_F_a1 = squeeze(mean(psd2(:, 1:12, alpha1_idx), 2));
f_psd2_F_a2 = squeeze(mean(psd2(:, 1:12, alpha2_idx), 2));
f_psd2_F_b1 = squeeze(mean(psd2(:, 1:12, beta1_idx), 2));
f_psd2_F_b2 = squeeze(mean(psd2(:, 1:12, beta2_idx), 2));

f_psd2_T_d = squeeze(mean(psd2(:, [13 17], delta_idx), 2));
f_psd2_T_t = squeeze(mean(psd2(:, [13 17], theta_idx), 2));
f_psd2_T_a1 = squeeze(mean(psd2(:, [13 17], alpha1_idx), 2));
f_psd2_T_a2 = squeeze(mean(psd2(:, [13 17], alpha2_idx), 2));
f_psd2_T_b1 = squeeze(mean(psd2(:, [13 17], beta1_idx), 2));
f_psd2_T_b2 = squeeze(mean(psd2(:, [13 17], beta2_idx), 2));

f_psd2_C_d = squeeze(mean(psd2(:, [14:16,18:21], delta_idx), 2));
f_psd2_C_t = squeeze(mean(psd2(:, [14:16,18:21], theta_idx), 2));
f_psd2_C_a1 = squeeze(mean(psd2(:, [14:16,18:21], alpha1_idx), 2));
f_psd2_C_a2 = squeeze(mean(psd2(:, [14:16,18:21], alpha2_idx), 2));
f_psd2_C_b1 = squeeze(mean(psd2(:, [14:16,18:21], beta1_idx), 2));
f_psd2_C_b2 = squeeze(mean(psd2(:, [14:16,18:21], beta2_idx), 2));

f_psd2_P_d = squeeze(mean(psd2(:, 22:27, delta_idx), 2));
f_psd2_P_t = squeeze(mean(psd2(:, 22:27, theta_idx), 2));
f_psd2_P_a1 = squeeze(mean(psd2(:, 22:27, alpha1_idx), 2));
f_psd2_P_a2 = squeeze(mean(psd2(:, 22:27, alpha2_idx), 2));
f_psd2_P_b1 = squeeze(mean(psd2(:, 22:27, beta1_idx), 2));
f_psd2_P_b2 = squeeze(mean(psd2(:, 22:27, beta2_idx), 2));

f_psd2_O_d = squeeze(mean(psd2(:, 28:30, delta_idx), 2));
f_psd2_O_t = squeeze(mean(psd2(:, 28:30, theta_idx), 2));
f_psd2_O_a1 = squeeze(mean(psd2(:, 28:30, alpha1_idx), 2));
f_psd2_O_a2 = squeeze(mean(psd2(:, 28:30, alpha2_idx), 2));
f_psd2_O_b1 = squeeze(mean(psd2(:, 28:30, beta1_idx), 2));
f_psd2_O_b2 = squeeze(mean(psd2(:, 28:30, beta2_idx), 2));

f_psd2_d = {f_psd2_F_d,f_psd2_T_d,f_psd2_C_d,f_psd2_P_d,f_psd2_O_d};
f_psd2_t = {f_psd2_F_t,f_psd2_T_t,f_psd2_C_t,f_psd2_P_t,f_psd2_O_t};
f_psd2_a1 = {f_psd2_F_a1,f_psd2_T_a1,f_psd2_C_a1,f_psd2_P_a1,f_psd2_O_a1};
f_psd2_a2 = {f_psd2_F_a2,f_psd2_T_a2,f_psd2_C_a2,f_psd2_P_a2,f_psd2_O_a2};
f_psd2_b1 = {f_psd2_F_b1,f_psd2_T_b1,f_psd2_C_b1,f_psd2_P_b1,f_psd2_O_b1};
f_psd2_b2 = {f_psd2_F_b2,f_psd2_T_b2,f_psd2_C_b2,f_psd2_P_b2,f_psd2_O_b2};

% 组3
f_psd3_F_d = squeeze(mean(psd3(:, 1:12, delta_idx), 2));
f_psd3_F_t = squeeze(mean(psd3(:, 1:12, theta_idx), 2));
f_psd3_F_a1 = squeeze(mean(psd3(:, 1:12, alpha1_idx), 2));
f_psd3_F_a2 = squeeze(mean(psd3(:, 1:12, alpha2_idx), 2));
f_psd3_F_b1 = squeeze(mean(psd3(:, 1:12, beta1_idx), 2));
f_psd3_F_b2 = squeeze(mean(psd3(:, 1:12, beta2_idx), 2));

f_psd3_T_d = squeeze(mean(psd3(:, [13 17], delta_idx), 2));
f_psd3_T_t = squeeze(mean(psd3(:, [13 17], theta_idx), 2));
f_psd3_T_a1 = squeeze(mean(psd3(:, [13 17], alpha1_idx), 2));
f_psd3_T_a2 = squeeze(mean(psd3(:, [13 17], alpha2_idx), 2));
f_psd3_T_b1 = squeeze(mean(psd3(:, [13 17], beta1_idx), 2));
f_psd3_T_b2 = squeeze(mean(psd3(:, [13 17], beta2_idx), 2));

f_psd3_C_d = squeeze(mean(psd3(:, [14:16,18:21], delta_idx), 2));
f_psd3_C_t = squeeze(mean(psd3(:, [14:16,18:21], theta_idx), 2));
f_psd3_C_a1 = squeeze(mean(psd3(:, [14:16,18:21], alpha1_idx), 2));
f_psd3_C_a2 = squeeze(mean(psd3(:, [14:16,18:21], alpha2_idx), 2));
f_psd3_C_b1 = squeeze(mean(psd3(:, [14:16,18:21], beta1_idx), 2));
f_psd3_C_b2 = squeeze(mean(psd3(:, [14:16,18:21], beta2_idx), 2));

f_psd3_P_d = squeeze(mean(psd3(:, 22:27, delta_idx), 2));
f_psd3_P_t = squeeze(mean(psd3(:, 22:27, theta_idx), 2));
f_psd3_P_a1 = squeeze(mean(psd3(:, 22:27, alpha1_idx), 2));
f_psd3_P_a2 = squeeze(mean(psd3(:, 22:27, alpha2_idx), 2));
f_psd3_P_b1 = squeeze(mean(psd3(:, 22:27, beta1_idx), 2));
f_psd3_P_b2 = squeeze(mean(psd3(:, 22:27, beta2_idx), 2));

f_psd3_O_d = squeeze(mean(psd3(:, 28:30, delta_idx), 2));
f_psd3_O_t = squeeze(mean(psd3(:, 28:30, theta_idx), 2));
f_psd3_O_a1 = squeeze(mean(psd3(:, 28:30, alpha1_idx), 2));
f_psd3_O_a2 = squeeze(mean(psd3(:, 28:30, alpha2_idx), 2));
f_psd3_O_b1 = squeeze(mean(psd3(:, 28:30, beta1_idx), 2));
f_psd3_O_b2 = squeeze(mean(psd3(:, 28:30, beta2_idx), 2));

f_psd3_d = {f_psd3_F_d,f_psd3_T_d,f_psd3_C_d,f_psd3_P_d,f_psd3_O_d};
f_psd3_t = {f_psd3_F_t,f_psd3_T_t,f_psd3_C_t,f_psd3_P_t,f_psd3_O_t};
f_psd3_a1 = {f_psd3_F_a1,f_psd3_T_a1,f_psd3_C_a1,f_psd3_P_a1,f_psd3_O_a1};
f_psd3_a2 = {f_psd3_F_a2,f_psd3_T_a2,f_psd3_C_a2,f_psd3_P_a2,f_psd3_O_a2};
f_psd3_b1 = {f_psd3_F_b1,f_psd3_T_b1,f_psd3_C_b1,f_psd3_P_b1,f_psd3_O_b1};
f_psd3_b2 = {f_psd3_F_b2,f_psd3_T_b2,f_psd3_C_b2,f_psd3_P_b2,f_psd3_O_b2};

% Delta (不同ROIs)
ns  = size(f_psd1_d{1},1);
for h = 1:5
figure
avg1_d(:,h) = mean(f_psd1_d{h},1);
se1_d(:,h)  = std(f_psd1_d{h})/sqrt(ns);

avg2_d(:,h) = mean(f_psd2_d{h},1);
se2_d(:,h)  = std(f_psd2_d{h})/sqrt(ns);

avg3_d(:,h) = mean(f_psd3_d{h},1);
se3_d(:,h)  = std(f_psd3_d{h})/sqrt(ns);

[~,p_d12,~,stat_d12] = ttest(mean(f_psd1_d{h},2),mean(f_psd2_d{h},2),"Tail","left");
pvals_d12(:,h) = p_d12;
tvals_d12(:,h) = stat_d12.tstat;

[~,p_d13,~,stat_d13] = ttest(mean(f_psd1_d{h},2),mean(f_psd3_d{h},2),"Tail","left");
pvals_d13(:,h) = p_d13;
tvals_d13(:,h) = stat_d13.tstat;

plot(f(all_idx{1}), mean(f_psd1_d{h},1), 'linewidth', 1,'Color','#00AEEF');
hold on
plot(f(all_idx{1}), mean(f_psd2_d{h},1), 'linewidth', 1,'Color','#8DC63F');
plot(f(all_idx{1}), mean(f_psd3_d{h},1), 'linewidth', 1,'Color','#FFC20D');
fill([f(all_idx{1})', fliplr(f(all_idx{1})')], ...
    [(avg1_d(:,h)+se1_d(:,h))', fliplr((avg1_d(:,h)-se1_d(:,h))')], ...
    'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{1})', fliplr(f(all_idx{1})')], ...
    [(avg2_d(:,h)+se2_d(:,h))', fliplr((avg2_d(:,h)-se2_d(:,h))')], ...
    'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{1})', fliplr(f(all_idx{1})')], ...
    [(avg3_d(:,h)+se3_d(:,h))', fliplr((avg3_d(:,h)-se3_d(:,h))')], ...
    'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Delta ' num2str(h)])
end

% Theta (不同ROIs)
for h = 1:5
figure
avg1_t(:,h) = mean(f_psd1_t{h},1);
se1_t(:,h)  = std(f_psd1_t{h})/sqrt(ns);

avg2_t(:,h) = mean(f_psd2_t{h},1);
se2_t(:,h)  = std(f_psd2_t{h})/sqrt(ns);

avg3_t(:,h) = mean(f_psd3_t{h},1);
se3_t(:,h)  = std(f_psd3_t{h})/sqrt(ns);

[~,p_t12,~,stat_t12] = ttest(mean(f_psd1_t{h},2),mean(f_psd2_t{h},2),"Tail","left");
pvals_t12(:,h) = p_t12;
tvals_t12(:,h) = stat_t12.tstat;

[~,p_t13,~,stat_t13] = ttest(mean(f_psd1_t{h},2),mean(f_psd3_t{h},2),"Tail","left");
pvals_t13(:,h) = p_t13;
tvals_t13(:,h) = stat_t13.tstat;

plot(f(all_idx{2}), mean(f_psd1_t{h},1), 'linewidth', 1,'Color','#00AEEF');
hold on
plot(f(all_idx{2}), mean(f_psd2_t{h},1), 'linewidth', 1,'Color','#8DC63F');
plot(f(all_idx{2}), mean(f_psd3_t{h},1), 'linewidth', 1,'Color','#FFC20D');
fill([f(all_idx{2})', fliplr(f(all_idx{2})')], ...
    [(avg1_t(:,h)+se1_t(:,h))', fliplr((avg1_t(:,h)-se1_t(:,h))')], ...
    'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{2})', fliplr(f(all_idx{2})')], ...
    [(avg2_t(:,h)+se2_t(:,h))', fliplr((avg2_t(:,h)-se2_t(:,h))')], ...
    'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{2})', fliplr(f(all_idx{2})')], ...
    [(avg3_t(:,h)+se3_t(:,h))', fliplr((avg3_t(:,h)-se3_t(:,h))')], ...
    'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Theta ' num2str(h)])
end

% Alpha1 (不同ROIs)
for h = 1:5
figure
avg1_a1(:,h) = mean(f_psd1_a1{h},1);
se1_a1(:,h)  = std(f_psd1_a1{h})/sqrt(ns);

avg2_a1(:,h) = mean(f_psd2_a1{h},1);
se2_a1(:,h)  = std(f_psd2_a1{h})/sqrt(ns);

avg3_a1(:,h) = mean(f_psd3_a1{h},1);
se3_a1(:,h)  = std(f_psd3_a1{h})/sqrt(ns);

[~,p_a112,~,stat_a112] = ttest(mean(f_psd1_a1{h},2),mean(f_psd2_a1{h},2),"Tail","left");
pvals_a112(:,h) = p_a112;
tvals_a112(:,h) = stat_a112.tstat;

[~,p_a113,~,stat_a113] = ttest(mean(f_psd1_a1{h},2),mean(f_psd3_a1{h},2),"Tail","left");
pvals_a113(:,h) = p_a113;
tvals_a113(:,h) = stat_a113.tstat;

plot(f(all_idx{3}), mean(f_psd1_a1{h},1), 'linewidth', 1,'Color','#00AEEF');
hold on
plot(f(all_idx{3}), mean(f_psd2_a1{h},1), 'linewidth', 1,'Color','#8DC63F');
plot(f(all_idx{3}), mean(f_psd3_a1{h},1), 'linewidth', 1,'Color','#FFC20D');
fill([f(all_idx{3})', fliplr(f(all_idx{3})')], ...
    [(avg1_a1(:,h)+se1_a1(:,h))', fliplr((avg1_a1(:,h)-se1_a1(:,h))')], ...
    'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{3})', fliplr(f(all_idx{3})')], ...
    [(avg2_a1(:,h)+se2_a1(:,h))', fliplr((avg2_a1(:,h)-se2_a1(:,h))')], ...
    'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{3})', fliplr(f(all_idx{3})')], ...
    [(avg3_a1(:,h)+se3_a1(:,h))', fliplr((avg3_a1(:,h)-se3_a1(:,h))')], ...
    'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Alpha1 ' num2str(h)])
end

% Alpha2 (不同ROIs)
for h = 1:5
figure
avg1_a2(:,h) = mean(f_psd1_a2{h},1);
se1_a2(:,h)  = std(f_psd1_a2{h})/sqrt(ns);

avg2_a2(:,h) = mean(f_psd2_a2{h},1);
se2_a2(:,h)  = std(f_psd2_a2{h})/sqrt(ns);

avg3_a2(:,h) = mean(f_psd3_a2{h},1);
se3_a2(:,h)  = std(f_psd3_a2{h})/sqrt(ns);

[~,p_a212,~,stat_a212] = ttest(mean(f_psd1_a2{h},2),mean(f_psd2_a2{h},2),"Tail","left");
pvals_a212(:,h) = p_a212;
tvals_a212(:,h) = stat_a212.tstat;

[~,p_a213,~,stat_a213] = ttest(mean(f_psd1_a2{h},2),mean(f_psd3_a2{h},2),"Tail","left");
pvals_a213(:,h) = p_a213;
tvals_a213(:,h) = stat_a213.tstat;

plot(f(all_idx{4}), mean(f_psd1_a2{h},1), 'linewidth', 1,'Color','#00AEEF');
hold on
plot(f(all_idx{4}), mean(f_psd2_a2{h},1), 'linewidth', 1,'Color','#8DC63F');
plot(f(all_idx{4}), mean(f_psd3_a2{h},1), 'linewidth', 1,'Color','#FFC20D');
fill([f(all_idx{4})', fliplr(f(all_idx{4})')], ...
    [(avg1_a2(:,h)+se1_a2(:,h))', fliplr((avg1_a2(:,h)-se1_a2(:,h))')], ...
    'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{4})', fliplr(f(all_idx{4})')], ...
    [(avg2_a2(:,h)+se2_a2(:,h))', fliplr((avg2_a2(:,h)-se2_a2(:,h))')], ...
    'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{4})', fliplr(f(all_idx{4})')], ...
    [(avg3_a2(:,h)+se3_a2(:,h))', fliplr((avg3_a2(:,h)-se3_a2(:,h))')], ...
    'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Alpha2 ' num2str(h)])
end

% Beta1 (不同ROIs)
for h = 1:5
figure
avg1_b1(:,h) = mean(f_psd1_b1{h},1);
se1_b1(:,h)  = std(f_psd1_b1{h})/sqrt(ns);

avg2_b1(:,h) = mean(f_psd2_b1{h},1);
se2_b1(:,h)  = std(f_psd2_b1{h})/sqrt(ns);

avg3_b1(:,h) = mean(f_psd3_b1{h},1);
se3_b1(:,h)  = std(f_psd3_b1{h})/sqrt(ns);

[~,p_b112,~,stat_b112] = ttest(mean(f_psd1_b1{h},2),mean(f_psd2_b1{h},2),"Tail","left");
pvals_b112(:,h) = p_b112;
tvals_b112(:,h) = stat_b112.tstat;

[~,p_b113,~,stat_b113] = ttest(mean(f_psd1_b1{h},2),mean(f_psd3_b1{h},2),"Tail","left");
pvals_b113(:,h) = p_b113;
tvals_b113(:,h) = stat_b113.tstat;

plot(f(all_idx{5}), mean(f_psd1_b1{h},1), 'linewidth', 1,'Color','#00AEEF');
hold on
plot(f(all_idx{5}), mean(f_psd2_b1{h},1), 'linewidth', 1,'Color','#8DC63F');
plot(f(all_idx{5}), mean(f_psd3_b1{h},1), 'linewidth', 1,'Color','#FFC20D');
fill([f(all_idx{5})', fliplr(f(all_idx{5})')], ...
    [(avg1_b1(:,h)+se1_b1(:,h))', fliplr((avg1_b1(:,h)-se1_b1(:,h))')], ...
    'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{5})', fliplr(f(all_idx{5})')], ...
    [(avg2_b1(:,h)+se2_b1(:,h))', fliplr((avg2_b1(:,h)-se2_b1(:,h))')], ...
    'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{5})', fliplr(f(all_idx{5})')], ...
    [(avg3_b1(:,h)+se3_b1(:,h))', fliplr((avg3_b1(:,h)-se3_b1(:,h))')], ...
    'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Beta1 ' num2str(h)])
end

% Beta2 (不同ROIs)
for h = 1:5
figure
avg1_b2(:,h) = mean(f_psd1_b2{h},1);
se1_b2(:,h)  = std(f_psd1_b2{h})/sqrt(ns);

avg2_b2(:,h) = mean(f_psd2_b2{h},1);
se2_b2(:,h)  = std(f_psd2_b2{h})/sqrt(ns);

avg3_b2(:,h) = mean(f_psd3_b2{h},1);
se3_b2(:,h)  = std(f_psd3_b2{h})/sqrt(ns);

[~,p_b212,~,stat_b212] = ttest(mean(f_psd1_b2{h},2),mean(f_psd2_b2{h},2),"Tail","left");
pvals_b212(:,h) = p_b212;
tvals_b212(:,h) = stat_b212.tstat;

[~,p_b213,~,stat_b213] = ttest(mean(f_psd1_b2{h},2),mean(f_psd3_b2{h},2),"Tail","left");
pvals_b213(:,h) = p_b213;
tvals_b213(:,h) = stat_b213.tstat;

plot(f(all_idx{6}), mean(f_psd1_b2{h},1), 'linewidth', 1,'Color','#00AEEF');
hold on
plot(f(all_idx{6}), mean(f_psd2_b2{h},1), 'linewidth', 1,'Color','#8DC63F');
plot(f(all_idx{6}), mean(f_psd3_b2{h},1), 'linewidth', 1,'Color','#FFC20D');
fill([f(all_idx{6})', fliplr(f(all_idx{6})')], ...
    [(avg1_b2(:,h)+se1_b2(:,h))', fliplr((avg1_b2(:,h)-se1_b2(:,h))')], ...
    'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{6})', fliplr(f(all_idx{6})')], ...
    [(avg2_b2(:,h)+se2_b2(:,h))', fliplr((avg2_b2(:,h)-se2_b2(:,h))')], ...
    'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
fill([f(all_idx{6})', fliplr(f(all_idx{6})')], ...
    [(avg3_b2(:,h)+se3_b2(:,h))', fliplr((avg3_b2(:,h)-se3_b2(:,h))')], ...
    'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Beta2 ' num2str(h)])
end

%% delta波段
% 第一组被试
delta_psd1         = squeeze(mean(psd1(:, :, delta_idx), 3));
F_delta_psd1       = mean(delta_psd1(:,1:12),2); % F区
T_delta_psd1       = mean(delta_psd1(:,[13 17]),2); % T区
C_delta_psd1       = mean(delta_psd1(:,[14:16,18:21]),2); % C区
P_delta_psd1       = mean(delta_psd1(:,22:27),2); % P区
O_delta_psd1       = mean(delta_psd1(:,28:30),2); % O区
delta_psd1_avg     = squeeze(mean(mean(psd1(:,:,delta_idx),3),1));

% 第二组被试
delta_psd2         = squeeze(mean(psd2(:, :, delta_idx), 3));
F_delta_psd2       = mean(delta_psd2(:,1:12),2); % F区
T_delta_psd2       = mean(delta_psd2(:,[13 17]),2); % T区
C_delta_psd2       = mean(delta_psd2(:,[14:16,18:21]),2); % C区
P_delta_psd2       = mean(delta_psd2(:,22:27),2); % P区
O_delta_psd2       = mean(delta_psd2(:,28:30),2); % O区
delta_psd2_avg     = squeeze(mean(mean(psd2(:,:,delta_idx),3),1));

%% theta波段
% 第一组被试
theta_psd1         = squeeze(mean(psd1(:, :, theta_idx), 3));
F_theta_psd1       = mean(theta_psd1(:,1:12),2); % F区
T_theta_psd1       = mean(theta_psd1(:,[13 17]),2); % T区
C_theta_psd1       = mean(theta_psd1(:,[14:16,18:21]),2); % C区
P_theta_psd1       = mean(theta_psd1(:,22:27),2); % P区
O_theta_psd1       = mean(theta_psd1(:,28:30),2); % O区
theta_psd1_avg     = squeeze(mean(mean(psd1(:,:,theta_idx),3),1));

% 第二组被试
theta_psd2         = squeeze(mean(psd2(:, :, theta_idx), 3));
F_theta_psd2       = mean(theta_psd2(:,1:12),2); % F区
T_theta_psd2       = mean(theta_psd2(:,[13 17]),2); % T区
C_theta_psd2       = mean(theta_psd2(:,[14:16,18:21]),2); % C区
P_theta_psd2       = mean(theta_psd2(:,22:27),2); % P区
O_theta_psd2       = mean(theta_psd2(:,28:30),2); % O区
theta_psd2_avg     = squeeze(mean(mean(psd2(:,:,theta_idx),3),1));

%% alpha波段
% 第一组被试
alpha_psd1     = squeeze(mean(psd1(:, :, alpha_idx), 3));
F_alpha_psd1       = mean(alpha_psd1(:,1:12),2); % F区
T_alpha_psd1       = mean(alpha_psd1(:,[13 17]),2); % T区
C_alpha_psd1       = mean(alpha_psd1(:,[14:16,18:21]),2); % C区
P_alpha_psd1       = mean(alpha_psd1(:,22:27),2); % P区
O_alpha_psd1       = mean(alpha_psd1(:,28:30),2); % O区
alpha_psd1_avg = squeeze(mean(mean(psd1(:,:,alpha_idx),3),1));

% 第二组被试
alpha_psd2     = squeeze(mean(psd2(:, :, alpha_idx), 3));
F_alpha_psd2       = mean(alpha_psd2(:,1:12),2); % F区
T_alpha_psd2       = mean(alpha_psd2(:,[13 17]),2); % T区
C_alpha_psd2       = mean(alpha_psd2(:,[14:16,18:21]),2); % C区
P_alpha_psd2       = mean(alpha_psd2(:,22:27),2); % P区
O_alpha_psd2       = mean(alpha_psd2(:,28:30),2); % O区
alpha_psd2_avg = squeeze(mean(mean(psd2(:,:,alpha_idx),3),1));

%% alpha1波段
% 第一组被试
alpha1_psd1        = squeeze(mean(psd1(:, :, alpha1_idx), 3)); % 对α波段做平均
F_alpha1_psd1       = mean(alpha1_psd1(:,1:12),2); % F区
T_alpha1_psd1       = mean(alpha1_psd1(:,[13 17]),2); % T区
C_alpha1_psd1       = mean(alpha1_psd1(:,[14:16,18:21]),2); % C区
P_alpha1_psd1       = mean(alpha1_psd1(:,22:27),2); % P区
O_alpha1_psd1       = mean(alpha1_psd1(:,28:30),2); % O区
alpha1_psd1_avg    = squeeze(mean(mean(psd1(:,:,alpha1_idx),3),1)); % 对α波段和被试做平均

% 第二组被试
alpha1_psd2        = squeeze(mean(psd2(:, :, alpha1_idx), 3));
F_alpha1_psd2       = mean(alpha1_psd2(:,1:12),2); % F区
T_alpha1_psd2       = mean(alpha1_psd2(:,[13 17]),2); % T区
C_alpha1_psd2       = mean(alpha1_psd2(:,[14:16,18:21]),2); % C区
P_alpha1_psd2       = mean(alpha1_psd2(:,22:27),2); % P区
O_alpha1_psd2       = mean(alpha1_psd2(:,28:30),2); % O区
alpha1_psd2_avg    = squeeze(mean(mean(psd2(:,:,alpha1_idx),3),1));

%% alpha2波段
% 第一组被试
alpha2_psd1        = squeeze(mean(psd1(:, :, alpha2_idx), 3));
F_alpha2_psd1       = mean(alpha2_psd1(:,1:12),2); % F区
T_alpha2_psd1       = mean(alpha2_psd1(:,[13 17]),2); % T区
C_alpha2_psd1       = mean(alpha2_psd1(:,[14:16,18:21]),2); % C区
P_alpha2_psd1       = mean(alpha2_psd1(:,22:27),2); % P区
O_alpha2_psd1       = mean(alpha2_psd1(:,28:30),2); % O区
alpha2_psd1_avg    = squeeze(mean(mean(psd1(:,:,alpha2_idx),3),1));

% 第二组被试
alpha2_psd2        = squeeze(mean(psd2(:, :, alpha2_idx), 3));
F_alpha2_psd2       = mean(alpha2_psd2(:,1:12),2); % F区
T_alpha2_psd2       = mean(alpha2_psd2(:,[13 17]),2); % T区
C_alpha2_psd2       = mean(alpha2_psd2(:,[14:16,18:21]),2); % C区
P_alpha2_psd2       = mean(alpha2_psd2(:,22:27),2); % P区
O_alpha2_psd2       = mean(alpha2_psd2(:,28:30),2); % O区
alpha2_psd2_avg    = squeeze(mean(mean(psd2(:,:,alpha2_idx),3),1));

%% beta波段
% 第一组被试
beta_psd1         = squeeze(mean(psd1(:, :, beta_idx), 3));
F_beta_psd1       = mean(beta_psd1(:,1:12),2); % F区
T_beta_psd1       = mean(beta_psd1(:,[13 17]),2); % T区
C_beta_psd1       = mean(beta_psd1(:,[14:16,18:21]),2); % C区
P_beta_psd1       = mean(beta_psd1(:,22:27),2); % P区
O_beta_psd1       = mean(beta_psd1(:,28:30),2); % O区
beta_psd1_avg     = squeeze(mean(mean(psd1(:,:,beta_idx),3),1));

% 第二组被试
beta_psd2         = squeeze(mean(psd2(:, :, beta_idx), 3));
F_beta_psd2       = mean(beta_psd2(:,1:12),2); % F区
T_beta_psd2       = mean(beta_psd2(:,[13 17]),2); % T区
C_beta_psd2       = mean(beta_psd2(:,[14:16,18:21]),2); % C区
P_beta_psd2       = mean(beta_psd2(:,22:27),2); % P区
O_beta_psd2       = mean(beta_psd2(:,28:30),2); % O区
beta_psd2_avg     = squeeze(mean(mean(psd2(:,:,beta_idx),3),1));

%% beta1波段
% 第一组被试
beta1_psd1        = squeeze(mean(psd1(:, :, beta1_idx), 3));
F_beta1_psd1       = mean(beta1_psd1(:,1:12),2); % F区
T_beta1_psd1       = mean(beta1_psd1(:,[13 17]),2); % T区
C_beta1_psd1       = mean(beta1_psd1(:,[14:16,18:21]),2); % C区
P_beta1_psd1       = mean(beta1_psd1(:,22:27),2); % P区
O_beta1_psd1       = mean(beta1_psd1(:,28:30),2); % O区
beta1_psd1_avg    = squeeze(mean(mean(psd1(:,:,beta1_idx),3),1));

% 第二组被试
beta1_psd2        = squeeze(mean(psd2(:, :, beta1_idx), 3));
F_beta1_psd2       = mean(beta1_psd2(:,1:12),2); % F区
T_beta1_psd2       = mean(beta1_psd2(:,[13 17]),2); % T区
C_beta1_psd2       = mean(beta1_psd2(:,[14:16,18:21]),2); % C区
P_beta1_psd2       = mean(beta1_psd2(:,22:27),2); % P区
O_beta1_psd2       = mean(beta1_psd2(:,28:30),2); % O区
beta1_psd2_avg    = squeeze(mean(mean(psd2(:,:,beta1_idx),3),1));

%% beta2波段
% 第一组被试
beta2_psd1        = squeeze(mean(psd1(:, :, beta2_idx), 3));
F_beta2_psd1       = mean(beta2_psd1(:,1:12),2); % F区
T_beta2_psd1       = mean(beta2_psd1(:,[13 17]),2); % T区
C_beta2_psd1       = mean(beta2_psd1(:,[14:16,18:21]),2); % C区
P_beta2_psd1       = mean(beta2_psd1(:,22:27),2); % P区
O_beta2_psd1       = mean(beta2_psd1(:,28:30),2); % O区
beta2_psd1_avg    = squeeze(mean(mean(psd1(:,:,beta2_idx),3),1));

% 第二组被试
beta2_psd2        = squeeze(mean(psd2(:, :, beta2_idx), 3));
F_beta2_psd2       = mean(beta2_psd2(:,1:12),2); % F区
T_beta2_psd2       = mean(beta2_psd2(:,[13 17]),2); % T区
C_beta2_psd2       = mean(beta2_psd2(:,[14:16,18:21]),2); % C区
P_beta2_psd2       = mean(beta2_psd2(:,22:27),2); % P区
O_beta2_psd2       = mean(beta2_psd2(:,28:30),2); % O区
beta2_psd2_avg    = squeeze(mean(mean(psd2(:,:,beta2_idx),3),1));

% β/α
ratio1 = beta_psd1./alpha_psd1;
ratio1_avg = mean(ratio1,1);

F_ratio1       = mean(ratio1(:,1:12),2); % F区
T_ratio1       = mean(ratio1(:,[13 17]),2); % T区
C_ratio1       = mean(ratio1(:,[14:16,18:21]),2); % C区
P_ratio1       = mean(ratio1(:,22:27),2); % P区
O_ratio1       = mean(ratio1(:,28:30),2); % O区

ratio2 = beta_psd2./alpha_psd2;
ratio2_avg = mean(ratio2,1);
F_ratio2       = mean(ratio2(:,1:12),2); % F区
T_ratio2      = mean(ratio2(:,[13 17]),2); % T区
C_ratio2       = mean(ratio2(:,[14:16,18:21]),2); % C区
P_ratio2       = mean(ratio2(:,22:27),2); % P区
O_ratio2       = mean(ratio2(:,28:30),2); % O区

%% 画地形图(逐个画)
mycolor = flipud(othercolor('RdBu10'));

% Delta pre
figure
topoplot(delta_psd1_avg, EEG_chanlocs1,'maplimits',[1 5.5],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
% title('Delta pre', 'fontsize', 14, 'FontName', en);
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Theta pre
figure
topoplot(theta_psd1_avg, EEG_chanlocs1,'maplimits',[0.5 3.2],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
% title('Theta pre', 'fontsize', 14, 'FontName', en);
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Alpha1 pre
figure
topoplot(alpha1_psd1_avg, EEG_chanlocs1,'maplimits',[1 7.5],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
% title('Alpha1 pre', 'fontsize', 14, 'FontName', en);
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Alpha2 pre
figure
topoplot(alpha2_psd1_avg, EEG_chanlocs1,'maplimits',[0.9 5.8],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
% title('Alpha2 pre', 'fontsize', 14, 'FontName', en);
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Beta1 pre
figure
topoplot(beta1_psd1_avg, EEG_chanlocs1,'maplimits',[0.2 0.73],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
% title('Beta2 pre', 'fontsize', 14, 'FontName', en);
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Beta2 pre
figure
topoplot(beta2_psd1_avg, EEG_chanlocs1,'maplimits',[0.1 0.4],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
% title('Beta2 pre', 'fontsize', 14, 'FontName', en);
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% β/α pre
figure
topoplot(ratio1_avg, EEG_chanlocs1,'maplimits',[0.1 0.23],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Delta post
figure
topoplot(delta_psd2_avg, EEG_chanlocs2,'maplimits',[1 5.5],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Theta post
figure
topoplot(theta_psd2_avg, EEG_chanlocs2,'maplimits',[0.5 3.2],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Alpha1 post
figure
topoplot(alpha1_psd2_avg, EEG_chanlocs2,'maplimits',[1 7.5],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Alpha2 post
figure
topoplot(alpha2_psd2_avg, EEG_chanlocs2,'maplimits',[0.9 5.8],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Beta1 post
figure
topoplot(beta1_psd2_avg, EEG_chanlocs2,'maplimits',[0.2 0.73],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% Beta2 post
figure
topoplot(beta2_psd2_avg, EEG_chanlocs2,'maplimits',[0.1 0.4],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

% β/α post
figure
topoplot(ratio2_avg, EEG_chanlocs1,'maplimits',[0.1 0.23],'colormap',mycolor, ...
'plotrad',0.514,'electrodes','off','numcontour',6,'shading','interp');
fig = gcf;
lineobj = findobj(fig,'type','line');
[lineobj.LineWidth] = deal(1);
% [lineobj.MarkerSize] = deal(3);
contourobj = findobj(fig,'type','contour');
[contourobj.LineWidth] = deal(0.5);
% h = colorbar;
% h.TickLabels = '';

%% 比较不同频段差异并绘制拓扑图
% 对于每一个频率点(1~30Hz)

for ii = 1:30
    for jj = 1:6
        % 挑选第一、二组所有被试在第ii个通道，第jj个频段上的psd值
        data1(:,ii,jj) = mean(psd1(:,ii,all_idx{jj}),3); % 被试*通道*频率点
        data2(:,ii,jj) = mean(psd2(:,ii,all_idx{jj}),3); % 被试*通道*频率点
        % 做配对样本t检验
        [~,p,~,stat] = ttest(data1(:,ii,jj),data2(:,ii,jj),"Tail","left");
        % 存储p值、t值
        pvals(ii,jj) = p;
        tvals(ii,jj) = stat.tstat;
    end
end

% 计算β/α指标p值
[~,P,ci,STAT] = ttest(ratio1,ratio2,"Tail","left");
T = STAT.tstat;
tvals(:,end+1) = T;
pvals(:,end+1) = P;
a = ratio1 - ratio2;
b = mean(a);

% p值颜色映射
pcolor = othercolor('RdBu8');
% pcolor = slanCM(111);

% 对于6个频段及β/α (t值)
for k = 1:7
    figure
    topoplot(tvals(:,k),EEG_chanlocs2,'colormap',pcolor,...
    'plotrad',0.514,'electrodes','off','numcontour',0,'shading','interp');
    fig = gcf;
    lineobj = findobj(fig,'type','line');
    [lineobj.LineWidth] = deal(2);
    colorbar
end

% 对于6个频段及β/α (p值)
for k = 1:7
    figure
    topoplot(pvals(:,k), EEG_chanlocs2,'maplimits',[0 0.1],'colormap',pcolor,...
    'plotrad',0.514,'electrodes','off','numcontour',0,'shading','interp');
    fig = gcf;
    lineobj = findobj(fig,'type','line');
    [lineobj.LineWidth] = deal(1.5);
    colorbar
end

%% 储存各频段PSD作为特征
labels0  = zeros(length(group1_files),1);
labels1  = zeros(length(group1_files),1) + 1;

% 细分频段
all_psd1 = [F_delta_psd1,F_theta_psd1,F_alpha1_psd1,F_alpha2_psd1,F_beta1_psd1,F_beta2_psd1,F_ratio1,...
    T_delta_psd1,T_theta_psd1,T_alpha1_psd1,T_alpha2_psd1,T_beta1_psd1,T_beta2_psd1,T_ratio1,...
    C_delta_psd1,C_theta_psd1,C_alpha1_psd1,C_alpha2_psd1,C_beta1_psd1,C_beta2_psd1,C_ratio1,...
    P_delta_psd1,P_theta_psd1,P_alpha1_psd1,P_alpha2_psd1,P_beta1_psd1,P_beta2_psd1,P_ratio1,...
    O_delta_psd1,O_theta_psd1,O_alpha1_psd1,O_alpha2_psd1,O_beta1_psd1,O_beta2_psd1,O_ratio1,...
    labels0]; 

all_psd2 = [F_delta_psd2,F_theta_psd2,F_alpha1_psd2,F_alpha2_psd2,F_beta1_psd2,F_beta2_psd2,F_ratio2,...
    T_delta_psd2,T_theta_psd2,T_alpha1_psd2,T_alpha2_psd2,T_beta1_psd2,T_beta2_psd2,T_ratio2,...
    C_delta_psd2,C_theta_psd2,C_alpha1_psd2,C_alpha2_psd2,C_beta1_psd2,C_beta2_psd2,C_ratio2,...
    P_delta_psd2,P_theta_psd2,P_alpha1_psd2,P_alpha2_psd2,P_beta1_psd2,P_beta2_psd2,P_ratio2,...
    O_delta_psd2,T_theta_psd2,O_alpha1_psd2,O_alpha2_psd2,O_beta1_psd2,O_beta2_psd2,O_ratio2,...
    labels1];
    
all_psd_2  = [all_psd1;all_psd2];
save("all_psd_2.mat","all_psd_2")
