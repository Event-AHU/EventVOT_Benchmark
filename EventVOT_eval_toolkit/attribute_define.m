%% 
clc; close all; clear all; warning off; 

% COESOT_testing_subset = 'C:\Users\wangx\Downloads\VisEvent_SOT_Benchmark-main\sequence_evaluation_config\COESOT_testing_subset.txt';
% fid = fopen(COESOT_testing_subset, 'w'); 
% 
% files = dir('C:\Users\wangx\Downloads\VisEvent_SOT_Benchmark-main\annos\coesot_anno\');
% files = files(3:end); 
% 
% for vid =1:size(files, 1)
%     videoName = files(vid).name; 
%     fprintf(fid, '%s\n', videoName); 
% end 


attributeFile = './attributes_anno_event.xlsx'; 

%写文件;
%files 文件路径;
%A 数据;
%sheet xlsx工作表;
%x1Range 工作表的单元格;
% xlswrite(attributeFile, A, sheet, xlRange); 
%读文件
%ndata 读取的数据部分
%text 读取的文本部分
%alldata 读取ndata+text
%files 文件路径
%sheet 工作表
%x:x 

att_savePath = './annos/att/';

[ndata, text, alldata] = xlsread(attributeFile);

% for vid =1:300 
for vid =1:1141
    videoName = alldata{vid, 1}; 
    
    fid = fopen([att_savePath videoName '.txt'], 'w');  
    
    
    
   for attIDX =1:14
       currentATT = alldata{vid, attIDX+1}; 
       if isnan(currentATT) 
           fprintf(fid, '%s\n', num2str(0)); 
       else
           fprintf(fid, '%s\n', num2str(currentATT)); 
       end 
       
       
   end 
   
   fclose(fid); 

end 


































































