function [H] = Draw_radar(data,lim,labels,linestyle)
%data是要画图的数据（根据数据的个数来确定雷达图的轴数）
%lim是各指标画图上下限范围
%labels是坐标轴名称
    H={};
    n=length(data);%维度
    adj_data=zeros(n,1);
    point=zeros(n,2);
    
    set(gca,'units','normal','pos',[0 0 1 1]);
    axis off
    axis equal
    hold on
    theta_last=pi/2;
    for i=1:n
        theta=2*pi/n*i+pi/2;
        H{length(H) + 1} = plot([0,500*cos(theta)],[0,500*sin(theta)],'k-','linewidth',2);
        for j=1:5
           H{length(H) + 1} = plot([j*100*cos(theta_last),j*100*cos(theta)],[j*100*sin(theta_last),j*100*sin(theta)],'--','linewidth',0.75,'color',[0.5,0.5,0.5]);
        end
        
        theta_last=theta;
        if data(i)<lim(i,1)
            adj_data(i)=0;
        elseif data(i)>lim(i,2)
            adj_data(i)=500;
        else
            adj_data(i)=(data(i)-lim(i,1))/(lim(i,2)-lim(i,1))*500;
        end
        point(i,1:2)=[adj_data(i)*cos(theta);adj_data(i)*sin(theta)];
        text_around(510*cos(theta),510*sin(theta),labels{i},theta);
    end
    
    for i=1:n
        theta=2*pi/n*i+pi/2;
        for j=1:5
            text_around(j*100*cos(theta),j*100*sin(theta),num2str(lim(i,1)+(lim(i,2)-lim(i,1))/5*j),theta+pi/2,7);
        end
    end
    H{length(H) + 1} = plot([point(:,1);point(1,1)],[point(:,2);point(1,2)],linestyle,'linewidth',1.5);%绘制
    %fill(point(:,1),point(:,2),[0.9 0.9 0.7])
    %alpha(0.5);
    texts=findobj(gca,'Type','Text');
    minx=-300;
    maxx=300;
    miny=-300;
    maxy=300;
    for i=1:length(texts)
        rect=get(texts(i),'Extent');
        x=rect(1);
        y=rect(2);
        dx=rect(3);
        dy=rect(4);
        if x<minx
            minx=x;
        elseif x+dx>maxx
            maxx=x+dx;
        end
        if y<miny
            miny=y;
        elseif y+dy>maxy
            maxy=y+dy;
        end
    end
    axis([minx-50,maxx+50,miny-20,maxy+20]);
end
 
function text_around(x,y,txt,theta,fontsize)
    if nargin==4
        fontsize=14;
    end
    section=mod(theta+pi/12,2*pi);
    if section>pi+pi/6
        %上对齐
        if section>1.5*pi+pi/6
            %左对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','cap','HorizontalAlignment','left','Fontsize',fontsize);
        elseif section>1.5*pi
            %中对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','cap','HorizontalAlignment','center','Fontsize',fontsize);
        else
            %右对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','cap','HorizontalAlignment','right','Fontsize',fontsize);
        end
    elseif section>pi
        %中、右对齐
        text(x,y,txt,'Interpreter','latex','VerticalAlignment','middle','HorizontalAlignment','right','Fontsize',fontsize);
    elseif section>pi/6
        %下对齐
        if section>0.5*pi+pi/6
            %右对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','right','Fontsize',fontsize);
        elseif section>0.5*pi
            %中对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','center','Fontsize',fontsize);
        else
            %左对齐
            text(x,y,txt,'Interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','left','Fontsize',fontsize);
        end
    else
        %中、左对齐
        text(x,y,txt,'Interpreter','latex','VerticalAlignment','middle','HorizontalAlignment','left','Fontsize',fontsize);
    end
end