clc; clear all; close all; warning off;

addpath('./utils/')

lim = [0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60;0,60];
% labels = {'Full Occlusion','Deformation', 'Rotation', 'Fast Motion', 'Partial Occlusion', 'Illumination Vartiation','Scale Vartiation' , 'Background Object Motion', 'Motion Blur', 'Aspect Ration Change'};
labels = {'CM ','DEF', 'MOC', 'HOC','FOC', 'LI', 'OV', 'FM', 'BC', 'NM', 'BOM', 'SIO', 'SV', 'ST'};
% legendlabels = {'Ours','ToMP101','OSTrack','AiATrack', 'MixFormer22k', 'TrDimp',  'TransT50','SiamR-CNN', 'KeepTrack', 'STARK-ST101' };
legendlabels = {'Ours', 'OSTrack', 'SimTrack', 'TransT','Mixformer', 'Stark' };

data =[51.6 57.1 62.6 51.1 29.3 46.3 34.8 39.6 57.2 62.7 55.0 56.3 46.1 26.5;        % Ours 
       49.7 53.9 60.9 52.1 26.7 43.5 33.9 39.7 55.7 58.9 53.6 53.0 44.8 26.2;        % OSTrack
       49.5 52.1 59.6 51.9 32.2 45.7 36.7 40.5 55.6 59.2 54.0 50.5 46.0 28.6;        % SimTrack      
       48.4 51.6 57.9 47.9 31.1 43.3 32.5 36.5 54.4 60.4 52.0 50.0 43.2 23.2;        % TransT
       46.5 46.9 52.2 43.7 23.6 41.3 32.2 39.6 50.1 54.7 47.5 40.6 39.4 27.3;        % Mixformer
       38.8 42.6 45.4 35.8 18.2 35.9 27.5 38.8 44.6 46.9 42.3 34.7 35.6 27.5;        % Stark 
];



Draw_radar2(data,lim,labels,legendlabels);
