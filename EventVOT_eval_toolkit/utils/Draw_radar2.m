function Draw_radar2(data,lim,labels,legendlabels)
    hold on;
    linestyles = {'r-o','y-+','b-*','g-x','c-s','m-d','k-^','y:v', 'b:', 'g:'};
    for i=1:size(data,1)
        H = Draw_radar(data(i,:),lim,labels,linestyles{i});
        for j=1:length(H)-1
            set(get(get(H{j},'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
    end
    legend(legendlabels,'Interpreter','latex');
end
