% Define the directory containing your images
imageDirectory = '/home/mbzirc/Downloads/AhsanBB/Dehazing/UEIB_Data/Reference_papers/Comparison_Results/U-shape_Transformer_for_Underwater_Image_Enhancement-main/test/Ablation_Output';

% List all image files in the directory
imageFiles = dir(fullfile(imageDirectory, '*.png')); % Change the extension if necessary

% Check if there are any image files in the directory
if isempty(imageFiles)
    fprintf('No image files found in the selected directory. Exiting.\n');
    return;
end

% Initialize an array to store NIQE scores
niqeScores = zeros(1, numel(imageFiles));

% Loop through each image file
for i = 1:numel(imageFiles)
    % Read the image
    image = imread(fullfile(imageDirectory, imageFiles(i).name));
    
    % Calculate the NIQE score for the current image
    score = niqe(image);
    
    % Store the NIQE score in the array
    niqeScores(i) = score;
end

% Calculate the mean NIQE score
meanNIQE = mean(niqeScores);

% Display the mean NIQE score
fprintf('Mean NIQE Score: %.4f\n', meanNIQE);
