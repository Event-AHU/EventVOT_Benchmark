clc; close all; clear; warning off;

dataset_path = 'D:\samples';
sequence = dir(dataset_path);
sequence = sequence(3:end);

for videoIndex = 1:length(sequence)
    
    videoname = sequence(videoIndex).name;
    disp(['Processing: ', videoname]);

    %% 读取GT
    gt_path = fullfile(dataset_path, videoname, 'groundtruth.txt');
    trackerResult = importdata(gt_path);

    %% 读取图像
    img_dataPath = fullfile(dataset_path, videoname, 'img');
    frames = dir(fullfile(img_dataPath, '*.png'));

    [~, idx] = sort({frames.name});
    frames = frames(idx);

    %% 视频写入（直接写，避免中间存图）
    video_path = fullfile(dataset_path, videoname, [videoname '.avi']);
    writerObject = VideoWriter(video_path);
    writerObject.FrameRate = 30;
    open(writerObject);

    %% 创建不可见figure（加速）
    h = figure('visible','off');

    for frameIndex = 1:length(frames)

        img_path = fullfile(img_dataPath, frames(frameIndex).name);
        im = imread(img_path);

        imshow(im); hold on;

        % 当前bbox
        currentBBox = trackerResult(frameIndex,:);

        % ? bbox合法性检查
        if any(isnan(currentBBox)) || all(currentBBox <= 0)
            % skip
        else
            % 限制范围（防止越界）
            currentBBox(1) = max(1, currentBBox(1));
            currentBBox(2) = max(1, currentBBox(2));

            rectangle('Position', currentBBox, ...
                'LineWidth', 3, ...
                'EdgeColor', 'g');
        end

        % 帧号
        text(10, 30, sprintf('#%d', frameIndex), ...
            'Color','g','FontSize',20,'FontWeight','bold');

        set(gca,'position',[0 0 1 1]);

        % 获取图像
        frame = getframe(gca);
        writeVideo(writerObject, frame);

        hold off;
    end

    close(writerObject);
    close(h);

    disp(['Saved video: ', video_path]);

end