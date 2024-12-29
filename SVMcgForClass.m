function [cg,bestacc,bestc,bestg] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%SVMcg cross validation by faruto

%%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2010.01.17
%Super Moderator @ www.ilovematlab.cn

%% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% about the parameters of SVMcg 
if nargin < 10
    accstep = 4.5;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end

%% X:c Y:g cg:CVaccuracy
[c,g] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(c);
cg = zeros(m,n);

% eps = 10^(-8);

%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^c(i,j) ),' -g ',num2str( basenum^g(i,j) ),...
            ' -t 2'];
        cg(i,j) = libsvmtrain(train_label, train, cmd);
        
        % if cg(i,j) <= 55
        %     continue;
        % end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^c(i,j);
            bestg = basenum^g(i,j);
        end        
        
        % if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
        %     bestacc = cg(i,j);
        %     bestc = basenum^X(i,j);
        %     bestg = basenum^Y(i,j);
        % end        
        
    end
end
%% to draw the acc with different c & g
% figure
% [C,h] = contour(X,Y,cg);
% ax = gca;
% ax.LineWidth = 1;
% ax.FontName = 'Arial';
% ax.FontSize = 12;
% contourobj = findobj(ax,'type','contour');
% contourobj.FaceColor = 'flat';
% % clabel(C,h,'Color','k','fontsize',7, 'FontName', 'Arial');
% xlabel('log_{2}c','FontSize',14, 'FontName', 'Arial', 'Interpreter', 'tex');
% ylabel('log_{2}g','FontSize',14, 'FontName', 'Arial', 'Interpreter', 'tex');
% yticks(-10:5:10)
% firstline = 'SVC参数选择结果图(等高线图)[GridSearchMethod]'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' Accuracy=',num2str(bestacc),'%'];
% title({firstline;secondline},'Fontsize',14, 'FontName', '宋体');
% hold on
% plot(log2(bestc),log2(bestg),'o','MarkerFaceColor',[0.85,0.33,0.10],'MarkerSize',6,...
%     MarkerEdgeColor='none')
% line([log2(bestc), log2(bestc)], [gmin, gmax],...
%     'Color',[0.8500 0.3250 0.0980], 'LineStyle', '--', 'linewidth', 0.75);
% line([cmin, cmax], [log2(bestg), log2(bestg)],...
%     'Color',[0.8500 0.3250 0.0980], 'LineStyle', '--', 'linewidth', 0.75);
% % clim([0, 100]);
% lgd = legend('', 'optimal point', '', '', 'fontsize', 10, 'FontName', 'Arial', ...
%     'Location','Best');
% lgd.LineWidth = 0.75;
% colorbar
% hold off

% figure
% % meshc(X,Y,cg);
% % mesh(X,Y,cg);
% surf(c,g,cg);
% % surfc(X,Y,cg)
% axis([cmin,cmax,gmin,gmax,40,75]);
% ax = gca;
% ax.LineWidth = 0.75;
% ax.FontName = 'Arial';
% ax.FontSize = 12;
% ax.TickLength = [0 0.015];
% ax.Color = [0.93 0.95 0.97];
% surfaceobj = findobj(ax,'type','surface');
% surfaceobj.FaceColor = 'interp';
% % contourobj = findobj(ax,'type','contour');
% % contourobj.FaceColor = 'flat';
% % contourobj.EdgeColor = 'none';
% xlabel('log_{2}c','FontSize',14, 'FontName', 'Arial', 'Interpreter', 'tex');
% ylabel('log_{2}g','FontSize',14, 'FontName', 'Arial', 'Interpreter', 'tex');
% zlabel('Accuracy(%)','FontSize',14, 'FontName', 'Arial');
% firstline = 'SVC参数选择结果图(3D视图)[GridSearchMethod]'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' Accuracy=',num2str(bestacc),'%'];
% title({firstline;secondline},'Fontsize',14, 'FontName', '宋体');
% hold on
% 
% plot3(log2(bestc), log2(bestg), 0, 'o','MarkerFaceColor',[0.85,0.33,0.10], ...
%     'MarkerSize',6,MarkerEdgeColor='none')
% plot3(log2(bestc),log2(bestg),bestacc,'o','MarkerFaceColor',[0.85,0.33,0.10], ...
%     'MarkerSize',6,MarkerEdgeColor='none')
% line([log2(bestc), log2(bestc)], [gmin, gmax], [0 ,0],...
%     'Color',[0.8500 0.3250 0.0980], 'LineStyle', '--', 'linewidth', 0.75)
% line([cmin, cmax], [log2(bestg), log2(bestg)], [0 ,0],...
%     'Color',[0.8500 0.3250 0.0980], 'LineStyle', '--', 'linewidth', 0.75)
% line([log2(bestc), log2(bestc)], [log2(bestg), log2(bestg)], [0, bestacc],...
%     'Color',[0.8500 0.3250 0.0980], 'LineStyle', '--', 'linewidth', 0.75)
% % clim([0, 100]);
% legend('', '', 'optimal point', '', '', '', '', 'fontsize', 10, 'FontName', 'Arial', ...
%     'Location','Best','Color','w')
% box on
% hold off
