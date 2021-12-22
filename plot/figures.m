clear all; close all; clc

%% -------------------------------Figure 8--------------------------------------
% load
filepath = 'figure8.csv';
inputs   = csvread(filepath,0,0);
pred   = inputs(:,1:10);
target = inputs(:,11:20);
filepath = 'figure8_metrics.csv';
metrics = csvread(filepath,0,0);
r2 = metrics(:,1);
rmse = metrics(:,2);

% return size
[~,n] = size(pred);


title = {'Tropical', 'Arid, desert', 'Arid, steppe', 'Temperate, dry summer', ...
    'Temperate, dry winter','Temperate, no dry season', 'Cold, dry summer', ...
    'Cold, dry winter','Cold, no dry season', 'Polar'};

position = {[0.05,0.67,0.29,0.29],[0.35,0.67,0.29,0.29],[0.65,0.67,0.29,0.29], ...
            [0.05,0.36,0.29,0.29],[0.35,0.36,0.29,0.29],[0.65,0.36,0.29,0.29], ...
            [0.05,0.05,0.29,0.29],[0.35,0.05,0.29,0.29],[0.65,0.05,0.29,0.29]};
name = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)'};
% figure
figure
set(gcf,'Units','centimeters','Position',[0.5 0.5 40 36]);
for i = 2:n-1
    
    %subplot(3,3,i-1)
    subplot('position',position{i-1})
    
    Scatplot(pred(:,i), target(:,i),'squares',0.1,100,5,1,3);
    
    Y = pred(:,i);
    X = target(:,i);
    
    axis([0, 0.5, 0, 0.5])
    
    set(gca,'XTick',[0, 0.1,0.3,0.5])
    set(gca,'YTick',[0, 0.1,0.3,0.5])
    
    if i-1 >=7 && i-1<=9
        set(gca,'XTicklabel',[0, 0.1,0.3,0.5])
    else
        set(gca,'XTicklabel',[])
    end
    
    if i-1 ==1 || i-1==4 || i-1==7
        set(gca,'YTicklabel',[0, 0.1,0.3,0.5])
    else
        set(gca,'YTicklabel',[])
    end
    
    if i-1 == 5
        axis([0, 0.7, 0, 0.7])
    
        set(gca,'XTick',[0, 0.1,0.3,0.5,0.7])
        set(gca,'YTick',[0, 0.1,0.3,0.5,0.7])
        set(gca,'XTicklabel',[0, 0.1,0.3,0.5,0.7])
        set(gca,'YTicklabel',[0, 0.1,0.3,0.5,0.7])
    end

    %axis([nanmin(pred(:,i)), nanmax(pred(:,i)), ...
    %      nanmin(pred(:,i)), nanmax(pred(:,i))])

    %set(gca,'XTick',[0, nanmax(pred(:,i))/5, 2*nanmax(pred(:,i))/5, ...
    %           3*nanmax(pred(:,i))/5, 4*nanmax(pred(:,i))/5,nanmax(pred(:,i))])
    %set(gca,'YTick',[0, nanmax(pred(:,i))/5, 2*nanmax(pred(:,i))/5, ...
    %           3*nanmax(pred(:,i))/5, 4*nanmax(pred(:,i))/5,nanmax(pred(:,i))])

    Rsqure = ['R^2 = ',num2str(r2(i-1))];
    RMSE = ['RMSE = ',num2str(rmse(i-1))];

    
    if i-1 == 5
        text(0.02,0.60, Rsqure,'Fontname', ...
            'Times New Roman','fontsize',14,'FontWeight','bold');
        text(0.02,0.56,RMSE,'Fontname', ...
            'Times New Roman','fontsize',14,'FontWeight','bold');
        text(0.02, 0.66, title{i},'Fontname', ...
            'Times New Roman','fontsize',14,'FontWeight','bold');
    else
        text(0.02,0.40, Rsqure,'Fontname', ...
            'Times New Roman','fontsize',14,'FontWeight','bold');
        text(0.02,0.36,RMSE,'Fontname', ...
            'Times New Roman','fontsize',14,'FontWeight','bold');
        text(0.02, 0.46, title{i},'Fontname', ...
            'Times New Roman','fontsize',14,'FontWeight','bold');        
    end
    
    h1=refline(1,0); %¸¨Öú1:1Ïß
    set(h1,'color','red','linewidth',1.1);
    axis square
    %box on
    %daspect([1 1 1]);
    if i-1 == 4
        ylabel('Prediction','Fontname', ...
        'Times New Roman','fontsize',18,'FontWeight','bold')
    end
    
    if i-1 == 8
        xlabel('Ground Truth','Fontname', ...
        'Times New Roman','fontsize',18,'FontWeight','bold')
    end
    if i-1 == 5
        text(0.64,0.05,name{i-1},'Fontname', ...
            'Times New Roman','fontsize',18,'FontWeight','bold')
    else
        text(0.44,0.05*0.5/0.7,name{i-1},'Fontname', ...
            'Times New Roman','fontsize',18,'FontWeight','bold')
    end
end
colorbar('Position',[0.95,0.4,0.02,0.2])

print('/Users/lewlee/Desktop/figure8.pdf','-dpdf')

print('/Users/lewlee/Desktop/figure8.eps','-depsc')
