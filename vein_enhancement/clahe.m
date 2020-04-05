% CLAHE (Contrast Limited Adaptive Histogram Equalization) filtering

% clahe applied on ground truth band
I1 = imread('./images/ground_truth1.png');
J1 = adapthisteq(I1(:,:,1));

% clahe applied on reconstructed band
I2 = imread('./images/reconstructed1.png');
J2 = adapthisteq(I2(:,:,1));

imshowpair(J1,J2,'montage');
