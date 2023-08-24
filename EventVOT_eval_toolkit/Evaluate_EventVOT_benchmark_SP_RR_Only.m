%% tracker performance evaluation tool for COESOT Benchmark
% (modified based on LaSOT toolkit by Xiao Wang)
% 10/01/2022


clc; clear all; close all; warning off;

addpath('./utils/');
addpath('./sequence_evaluation_config/');

tmp_mat_path  = './tmp_mat/';          % path to save temporary results
path_anno     = './annos/eventvot_anno/';                  % path to annotations
path_att      = './annos/att/';               % path to attribute
rp_all        = './eventvot_tracking_results/';        % path to tracking results
save_fig_path = './res_fig/';                 % path to result figures
save_fig_suf  = 'png';                          % suffix of figures, 'png' or 'eps'

att_name      = {'Camera Motion', 'Deformation', ...
    'Mildly occlusion','Heavily occlusion', 'Full Occlusion', 'Low Illumination', 'Out-of-View',  ...
    'Fast Motion', 'Background Clutter', ...
    'No Motion', 'Background Object Motion',  ...
    'similarly interferential objects', 'scale variation', 'Small target' };

att_fig_name  = {'CM ','DEF', 'MOC', 'HOC','FOC', 'LI', 'OV', 'FM', 'BC', 'NM', 'BOM', 'SIO', 'SV', 'ST'};

% 'test_set' --- evaluation the whole test set
evaluation_dataset_type = 'test_set';

%% 
% use normalization or not   false
norm_dst = false;  % false  true

trackers      = config_tracker();
sequences  = config_sequence(evaluation_dataset_type);
plot_style   = config_plot_style();

num_seq = numel(sequences);
num_tracker = numel(trackers);

% load tracker info
name_tracker_all = cell(num_tracker, 1);
for i = 1:num_tracker
    name_tracker_all{i} = trackers{i}.name;
end

% load sequence info
name_seq_all = cell(num_seq, 1);
for i = 1:num_seq
    name_seq_all{i} = sequences{i};
    
%     seq_att = importdata(fullfile(path_att, [sequences{i} '.txt']));
    
%     if i == 1
%         att_all = zeros(num_seq, numel(seq_att));
%     end
    sequences{i}
%     att_all(i, :) = seq_att;
end

% parameters for evaluation
metric_type_set = {'error', 'overlap'};
eval_type       = 'OPE';
% ranking_type    = 'threshold';
ranking_type    = 'AUC';   % change it to 'AUC' for success plots
rank_num         = 50;
% draw the results under each challenging factors or not?
drawAttributeResults = false; %% true, false

% if strcmp(metric_type, 'error') && strcmp(ranking_type, 'AUC') % for ranking_type = 'AUC'
% if strcmp(metric_type, 'overlap') && strcmp(ranking_type, 'threshold')  % for ranking_type = 'threshold'



threshold_set_error   = 0:50;
if norm_dst
    threshold_set_error = threshold_set_error / 100;
end
threshold_set_overlap = 0:0.05:1;

for i = 1:numel(metric_type_set)
    % error (for distance plots) or overlap (for success plots)
    metric_type = metric_type_set{i};
    switch metric_type
        case 'error'
            threshold_set = threshold_set_error;
            rank_idx      = 21;
            x_label_name  = 'Location error threshold';
            y_label_name  = 'Precision';
        case 'overlap'
            threshold_set = threshold_set_overlap;
            rank_idx      = 11;
            x_label_name  = 'Overlap threshold';
            y_label_name  = 'Success rate';
    end
    
    %     if strcmp(metric_type, 'error') && strcmp(ranking_type, 'AUC') % for ranking_type = 'AUC'
    if strcmp(metric_type, 'overlap') && strcmp(ranking_type, 'threshold')  % for ranking_type = 'threshold'
        continue;
    end
    
    t_num = numel(threshold_set);
    
    % we only use OPE for evaluation
    plot_type = [metric_type '_' eval_type];
    
    switch metric_type
        case 'error'
            title_name = ['Precision plots of ' eval_type];
            if norm_dst
                title_name = ['Normalized ' title_name];
            end
            
            if strcmp(evaluation_dataset_type, 'all')
                title_name = [title_name ' on EventVOT'];
            else
                title_name = [title_name ' on EventVOT Testing Set'];
            end
        case 'overlap'
            title_name = ['Success plots of ' eval_type];
            
            if strcmp(evaluation_dataset_type, 'all')
                title_name = [title_name ' on EventVOT'];
            else
                title_name = [title_name ' on EventVOT Testing Set'];
            end
    end
    
    dataName = [tmp_mat_path 'aveSuccessRatePlot_' num2str(num_tracker) 'alg_'  plot_type '.mat'];
    
    % evaluate tracker performance
    eval_tracker(sequences, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst);
    
    % plot performance
    load(dataName);
    num_tracker = size(ave_success_rate_plot, 1);
    
    if rank_num > num_tracker || rank_num <0
        rank_num = num_tracker;
    end
    
    fig_name= [plot_type '_' ranking_type];
    idx_seq_set = 1:numel(sequences);
    
    
    
    
    %%%%%% draw and save the overall performance plot
    plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, ...
        idx_seq_set, rank_num, ranking_type, rank_idx, ...
        name_tracker_all, threshold_set, title_name, ...
        x_label_name, y_label_name, fig_name, save_fig_path, ...
        save_fig_suf);
    
    
    
    %%%%% draw and save the per-attribute performance plot
    if drawAttributeResults
        att_trld = 0;
        att_num  = size(att_all, 2);
        for att_idx = 1:att_num    % for each attribute
            idx_seq_set = find(att_all(:, att_idx) > att_trld);
            if length(idx_seq_set) < 2
                continue;
            end
            disp([att_name{att_idx} ' ' num2str(length(idx_seq_set))]);
            
            fig_name   = [att_fig_name{att_idx} '_'  plot_type '_' ranking_type];
            title_name = ['Plots of ' eval_type ': ' att_name{att_idx} ' (' num2str(length(idx_seq_set)) ')'];
            
            switch metric_type
                case 'overlap'
                    title_name = ['Success plots of ' eval_type ' - ' att_name{att_idx} ' (' num2str(length(idx_seq_set)) ')'];
                case 'error'
                    title_name = ['Precision plots of ' eval_type ' - ' att_name{att_idx} ' (' num2str(length(idx_seq_set)) ')'];
                    if norm_dst
                        title_name = ['Normalized ' title_name];
                    end
            end
            
            plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, ...
                idx_seq_set, rank_num, ranking_type, rank_idx, ...
                name_tracker_all, threshold_set, title_name, ...
                x_label_name, y_label_name, fig_name, save_fig_path, ...
                save_fig_suf);
        end
        
    end
    
end



