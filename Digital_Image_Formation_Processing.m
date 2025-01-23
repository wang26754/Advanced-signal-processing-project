clc;
clear;

BlockSize = [8, 8];

% 1. Load the images in Matlab and convert them to double.
img = im2double(imread('outoffocus.tiff'));
img_n = im2double(imread('natural.tiff'));

% 2. Visualize the images and illustrate the Bayer mosaic arrays.
figure;
subplot(2,1,1);
title('Out-of-focus Image');
imshow(img, []);
subplot(2,1,2);
title('Natural Image');
imshow(img_n, []);

% 3. Separate the images into subchannels
R = img(1:2:end, 1:2:end);
G1 = img(1:2:end, 2:2:end);
G2 = img(2:2:end, 1:2:end);
B = img(2:2:end, 2:2:end);

R_n = img_n(1:2:end, 1:2:end);
G1_n = img_n(1:2:end, 2:2:end);
G2_n = img_n(2:2:end, 1:2:end);
B_n = img_n(2:2:end, 2:2:end);

% Plot
figure;
subplot(2,2,1); imshow(R, []); title('Out-of-focus Image for Red channel');
subplot(2,2,2); imshow(G1, []); title('Out-of-focus Image for Green 1 channel');
subplot(2,2,3); imshow(G2, []); title('Out-of-focus Image for Green 2 channel');
subplot(2,2,4); imshow(B, []); title('Out-of-focus Image for Blue channel');

figure;
subplot(2,2,1); imshow(R_n, []); title('Natural Image for Red channel');
subplot(2,2,2); imshow(G1_n, []); title('Natural Image for Green 1 channel');
subplot(2,2,3); imshow(G2_n, []); title('Natural Image for Green 2 channel');
subplot(2,2,4); imshow(B_n, []); title('Natural Image for Blue channel');

% 4. Each subchannel with a sliding window operator
[R_mean, R_variance] = calculateScatterPlot(R);
[G1_mean, G1_variance] = calculateScatterPlot(G1);
[G2_mean, G2_variance] = calculateScatterPlot(G2);
[B_mean, B_variance] = calculateScatterPlot(B);

% 5. Fit straight lines to the mean-variance scatterplots and plot
R_fit = polyfit(R_mean, R_variance,1);
G1_fit = polyfit(G1_mean, G1_variance,1);
G2_fit = polyfit(G2_mean, G2_variance,1);
B_fit = polyfit(B_mean, B_variance,1);

figure;
subplot(2,2,1);
scatter(R_mean, R_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Red Channel Mean-Variance Scatterplot');
hold on;
plot(R_mean, polyval(R_fit, R_mean));
hold off;

subplot(2,2,2);
scatter(G1_mean,G1_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Green Channel 1 Mean-Variance Scatterplot');
hold on;
plot(G1_mean, polyval(G1_fit, G1_mean));
hold off;

subplot(2,2,3);
scatter(G2_mean,G2_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Green Channel 2 Mean-Variance Scatterplot');
hold on;
plot(G2_mean, polyval(G2_fit, G2_mean));
hold off;

subplot(2,2,4);
scatter(B_mean,B_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Blue Channel Mean-Variance Scatterplot');
hold on;
plot(B_mean, polyval(B_fit, B_mean));
hold off;
%%
% 6. Apply the transformation to each subchannel
R_trans = blockproc(R, BlockSize, @(block) Transform(block, R_fit(1), R_fit(2)));
G1_trans = blockproc(G1, BlockSize, @(block) Transform(block, G1_fit(1), G1_fit(2)));
G2_trans = blockproc(G2, BlockSize, @(block) Transform(block, G2_fit(1), G2_fit(2)));
B_trans = blockproc(B, BlockSize, @(block) Transform(block, B_fit(1), B_fit(2)));

R_trans_n = blockproc(R_n, BlockSize, @(block) Transform(block, R_fit(1), R_fit(2)));
G1_trans_n = blockproc(G1_n, BlockSize, @(block) Transform(block, G1_fit(1), G1_fit(2)));
G2_trans_n = blockproc(G2_n, BlockSize, @(block) Transform(block, G2_fit(1), G2_fit(2)));
B_trans_n = blockproc(B_n, BlockSize, @(block) Transform(block, B_fit(1), B_fit(2)));

% 7. Compute the mean-variance scatterplots and plot.
[R_trans_mean, R_trans_variance] = calculateScatterPlot(R_trans);
[G1_trans_mean, G1_trans_variance] = calculateScatterPlot(G1_trans);
[G2_trans_mean, G2_trans_variance] = calculateScatterPlot(G2_trans);
[B_trans_mean, B_trans_variance] = calculateScatterPlot(B_trans);

RT_fit = polyfit(R_trans_mean, R_trans_variance,1); 
G1T_fit = polyfit(G1_trans_mean, G1_trans_variance,1);
G2T_fit = polyfit(G2_trans_mean, G2_trans_variance,1);
BT_fit = polyfit(B_trans_mean, B_trans_variance,1);

figure;
subplot(2,2,1);
scatter(R_trans_mean, R_trans_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Red Channel Mean-Variance Scatterplot after the transformation');
hold on;
plot(R_trans_mean, polyval(RT_fit, R_trans_mean));
hold off;

subplot(2,2,2);
scatter(G1_trans_mean,G1_trans_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Green Channel 1 Mean-Variance Scatterplot after the transformation');
hold on;
plot(G1_trans_mean, polyval(G1T_fit, G1_trans_mean));
hold off;

subplot(2,2,3);
scatter(G2_trans_mean,G2_trans_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Green Channel 2 Mean-Variance Scatterplot after the transformation');
hold on;
plot(G2_trans_mean, polyval(G2T_fit, G2_trans_mean));
hold off;

subplot(2,2,4);
scatter(B_trans_mean,B_trans_variance);
xlabel('Sample Mean');
ylabel('Sample Variance');
title('Blue Channel Mean-Variance Scatterplot after the transformation');
hold on;
plot(B_trans_mean, polyval(BT_fit, B_trans_mean));
hold off;

% 8. Implement in Matlab a sliding-DCT filter
lambda = 0.1;
f = @(block) idct2(Threshold(block.data, lambda));
RT_f = blockproc(R_trans, BlockSize, f);
G1T_f = blockproc(G1_trans, BlockSize, f);
G2T_f = blockproc(G2_trans, BlockSize, f);
BT_f = blockproc(B_trans, BlockSize, f);

% 9. Apply the inverse  transformation to each subchannel
R_inver = Transform_inver(RT_f, R_fit(1), R_fit(2));
G1_inver = Transform_inver(G1T_f, G1_fit(1), G1_fit(2));
G2_inver = Transform_inver(G2T_f, G2_fit(1), G2_fit(2));
B_inver = Transform_inver(BT_f, B_fit(1), B_fit(2));

% 10. Compare the filtered natural image and the one obtained by the inverse transformation of
% the filtered transformed natural image
figure;
subplot(4,2,1); imshow(RT_f, []); title('Filtered out-of-focus image for Red channel');
subplot(4,2,2); imshow(R_inver, []); title('Inversed out-of-focus Transformation for Red channel');
subplot(4,2,3); imshow(G1T_f, []); title('Filtered out-of-focus image for Green 1');
subplot(4,2,4); imshow(G1_inver, []); title('Inversed out-of-focus Transformation for Green 1 channel');
subplot(4,2,5); imshow(G2T_f, []); title('Filtered out-of-focus image for Green 2 channel');
subplot(4,2,6); imshow(G2_inver, []); title('Inversed out-of-focus Transformation for Green 2 channel');
subplot(4,2,7); imshow(BT_f, []); title('Filtered out-of-focus image for Blue channel');
subplot(4,2,8); imshow(B_inver, []); title('Inversed out-of-focus Transformation for Blue channel');

% 8. Implement in Matlab a sliding-DCT filter
f = @(block) idct2(Threshold(block.data, lambda));
RT_f_n = blockproc(R_trans_n, BlockSize, f);
G1T_f_n = blockproc(G1_trans_n, BlockSize, f);
G2T_f_n = blockproc(G2_trans_n, BlockSize, f);
BT_f_n = blockproc(B_trans_n, BlockSize, f);

% 9. Apply the inverse  transformation to each subchannel
R_inver_n = Transform_inver(RT_f_n, R_fit(1), R_fit(2));
G1_inver_n = Transform_inver(G1T_f_n, G1_fit(1), G1_fit(2));
G2_inver_n = Transform_inver(G2T_f_n, G2_fit(1), G2_fit(2));
B_inver_n = Transform_inver(BT_f_n, B_fit(1), B_fit(2));

% 10. Compare the filtered natural image and the one obtained by the inverse transformation of
% the filtered transformed natural image
figure;
subplot(4,2,1); imshow(RT_f_n, []); title('Filtered natural image for Red channel');
subplot(4,2,2); imshow(R_inver_n, []); title('Inversed natural Transformation for Red channel');
subplot(4,2,3); imshow(G1T_f_n, []); title('Filtered natural image for Green 1');
subplot(4,2,4); imshow(G1_inver_n, []); title('Inversed natural Transformation for Green 1 channel');
subplot(4,2,5); imshow(G2T_f_n, []); title('Filtered natural image for Green 2 channel');
subplot(4,2,6); imshow(G2_inver_n, []); title('Inversed natural Transformation for Green 2 channel');
subplot(4,2,7); imshow(BT_f_n, []); title('Filtered natural image for Blue channel');
subplot(4,2,8); imshow(B_inver_n, []); title('Inversed natural Transformation for Blue channel');
%%
% 11. Perform simple demosaicking and white balance
[M, N] = size(zeros(2*size(R_inver)));
[X_g, Y_g] = meshgrid(1:N, 1:M);
[X_R, Y_R] = meshgrid(1:2:N, 1:2:M);
[X_G1, Y_G1] = meshgrid(1:2:N, 2:2:M);
[X_G2, Y_G2] = meshgrid(2:2:N, 1:2:M);
[X_B, Y_B] = meshgrid(2:2:N, 2:2:M);
interp_R = interp2(X_R, Y_R, R_inver, X_g, Y_g, 'nearest', 0);
interp_G1 = interp2(X_G1, Y_G1, G1_inver, X_g, Y_g, 'nearest', 0);
interp_G2 = interp2(X_G2, Y_G2, G2_inver, X_g, Y_g, 'nearest', 0);
interp_B = interp2(X_B, Y_B, B_inver, X_g, Y_g, 'nearest', 0);
interp_G = (interp_G1 + interp_G2)/2;
Demosaic_img = cat(3, interp_R, interp_G, interp_B);

% Perform median filtering on the saturation channel to reduce green noise
image_hsv = rgb2hsv(Demosaic_img);
image_hsv(:, :, 3) = medfilt2(image_hsv(:, :, 3), [8 8]);
Demosaic_img = hsv2rgb(image_hsv);

im_hsv = rgb2hsv(Demosaic_img);
max_pix = max(im_hsv(:,:,3), [], 'all');
[v1,v2] = find(im_hsv(:,:,3)==max_pix, 1, 'first');

balanced_image(:,:,1) = Demosaic_img(:,:,1) ./ Demosaic_img(v1,v2,1);
balanced_image(:,:,2) = Demosaic_img(:,:,2) ./ Demosaic_img(v1,v2,2);
balanced_image(:,:,3) = Demosaic_img(:,:,3) ./ Demosaic_img(v1,v2,3);

figure;
imshow(balanced_image, []);
title('White Balanced Out-of-focus Image');

[M, N] = size(zeros(2*size(R_inver_n)));
[X_g, Y_g] = meshgrid(1:N, 1:M);
[X_R, Y_R] = meshgrid(1:2:N, 1:2:M);
[X_G1, Y_G1] = meshgrid(1:2:N, 2:2:M);
[X_G2, Y_G2] = meshgrid(2:2:N, 1:2:M);
[X_B, Y_B] = meshgrid(2:2:N, 2:2:M);
interp_R_n = interp2(X_R, Y_R, R_inver_n, X_g, Y_g, 'nearest', 0);
interp_G1_n = interp2(X_G1, Y_G1, G1_inver_n, X_g, Y_g, 'nearest', 0);
interp_G2_n = interp2(X_G2, Y_G2, G2_inver_n, X_g, Y_g, 'nearest', 0);
interp_B_n = interp2(X_B, Y_B, B_inver_n, X_g, Y_g, 'nearest', 0);
interp_G_n = (interp_G1_n + interp_G2_n)/2;
Demosaic_img_n = cat(3, interp_R_n, interp_G_n, interp_B_n);


% Perform median filtering on the saturation channel to reduce green noise
image_hsv = rgb2hsv(Demosaic_img_n);
image_hsv(:, :, 3) = medfilt2(image_hsv(:, :, 3), [8 8]);
Demosaic_img_n = hsv2rgb(image_hsv);

im_hsv = rgb2hsv(Demosaic_img_n);
max_pix = max(im_hsv(:,:,3), [], 'all');
[v1,v2] = find(im_hsv(:,:,3)==max_pix, 1, 'first');

balanced_image_n(:,:,1) = Demosaic_img_n(:,:,1) ./ Demosaic_img_n(v1,v2,1);
balanced_image_n(:,:,2) = Demosaic_img_n(:,:,2) ./ Demosaic_img_n(v1,v2,2);
balanced_image_n(:,:,3) = Demosaic_img_n(:,:,3) ./ Demosaic_img_n(v1,v2,3);

figure;
imshow(balanced_image_n, []);
title('White Balanced Natural Image');
%%
% 12. Perform contrast and saturation correctionn
im_hsv = rgb2hsv(balanced_image);
saturationRange = [0.1, 0.9]; 
outputRange = [0, 1]; 
im_hsv(:,:,2) = imadjust(im_hsv(:,:,2), saturationRange, outputRange);
im_hsv(:,:,3) = histeq(im_hsv(:,:,3));
gammaValue = 0.4; 
im_hsv(:,:,3) = imadjust(im_hsv(:,:,3), [], [], gammaValue);
corrected_img = hsv2rgb(im_hsv);
figure; imshow(corrected_img); title('Contrast and Saturation Corrected Out-of-Focus Image');

im_hsv = rgb2hsv(balanced_image_n);
saturationRange = [0.1, 0.9]; 
outputRange = [0, 1]; 
im_hsv(:,:,2) = imadjust(im_hsv(:,:,2), saturationRange, outputRange);
im_hsv(:,:,3) = histeq(im_hsv(:,:,3));
gammaValue = 0.9; 
im_hsv(:,:,3) = imadjust(im_hsv(:,:,3), [], [], gammaValue);
corrected_img = hsv2rgb(im_hsv);
figure; imshow(corrected_img); title('Contrast and Saturation Corrected Natural Image');

function [output_mean, output_variance] = calculateScatterPlot(channel)
    MeanVar_block = @(block) [mean(block.data(:)), var(block.data(:))];
    variance_mean = blockproc(channel, [8, 8], MeanVar_block);
    output_mean = variance_mean(:, 1);
    output_variance = variance_mean(:, 2);
    output_mean = output_mean(:);
    output_variance = output_variance(:);
end

function t = Threshold(data, lambda)
    Block = dct2(data);
    Block(abs(Block) < lambda) = 0;
    t = Block;
end

function result = Transform(block, ac, bc)
    result = (block.data / ac) + (3/8) + (bc / (ac^2));
    result(result < 0) = 1;
    result = 2 * sqrt(result);
end

%  Inverse transformation function
function inverse_result = Transform_inver(pixel_value, ac, bc)
    result_1 = 0.25 * sqrt(3/2) * pixel_value.^(-1);
    result_2 = (5/8) * sqrt(3/2) * pixel_value.^(-3);
    inverse_result = ac * (0.25 * pixel_value.^2 + result_1 - (11/8)*(pixel_value.^(-2)) + result_2 - 1/8 - bc/(ac^2));
end