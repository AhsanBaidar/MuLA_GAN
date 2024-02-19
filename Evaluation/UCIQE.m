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
