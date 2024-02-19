% Example usage:
directoryPath = '/home/mbzirc/Downloads/AhsanBB/Dehazing/UEIB_Data/Reference_papers/Comparison_Results/U-shape_Transformer_for_Underwater_Image_Enhancement-main/test/Ablation_Output';
meanUCIQE = CalculateMeanUCIQ(directoryPath);
fprintf('Mean UCIQE Score: %.4f\n', meanUCIQE);

function Mean_UCIQE_Score = CalculateMeanUCIQ(directoryPath, Coe_Metric)
    % Check if Coe_Metric is not provided, set default values
    if nargin == 1
        Coe_Metric = [0.4680, 0.2745, 0.2576];
    end

    % List all image files in the directory
    imageFiles = dir(fullfile(directoryPath, '*.png')); % Change the file extension as needed

    % Initialize an array to store UCIQE scores
    uciqeScores = zeros(1, numel(imageFiles));

    % Iterate through the images in the directory
    for i = 1:numel(imageFiles)
%         disp(i)
        % Read the image
        img = imread(fullfile(directoryPath, imageFiles(i).name));

        % Calculate UCIQE score for the current image
        uciqeScores(i) = UCIQE(img, Coe_Metric);
    end

    % Calculate the mean UCIQE score
    Mean_UCIQE_Score = mean(uciqeScores);
end

function Qualty_Val = UCIQE(I, Coe_Metric)
    % Transform to Lab color space
    cform = makecform('srgb2lab');
    Img_lab = applycform(I, cform);

    Img_lum = double(Img_lab(:, :, 1));
    Img_lum = Img_lum / 255 + eps;

    Img_a = double(Img_lab(:, :, 2)) / 255;
    Img_b = double(Img_lab(:, :, 3)) / 255;

    % Chroma
    Img_Chr = sqrt(Img_a(:).^2 + Img_b(:).^2);

    % Saturation
    Img_Sat = Img_Chr ./ sqrt(Img_Chr.^2 + Img_lum(:).^2);

    % Average of saturation
    Aver_Sat = mean(Img_Sat);

    % Average of Chroma
    Aver_Chr = mean(Img_Chr);

    % Variance of Chroma
    Var_Chr = sqrt(mean((abs(1 - (Aver_Chr ./ Img_Chr).^2))));

    % Contrast of luminance
    Tol = stretchlim(Img_lum);
    Con_lum = Tol(2) - Tol(1);

    % Get final quality value
    Qualty_Val = Coe_Metric(1) * Var_Chr + Coe_Metric(2) * Con_lum + Coe_Metric(3) * Aver_Sat;
end


