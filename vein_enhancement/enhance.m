
% clahe enhancement
I1 = imread('./images/ground_truth1.png');
J1 = clahe(I1(:,:,1));

I2 = imread('./images/reconstructed1.png');
J2 = clahe(I2(:,:,1));
 
subplot 211
imshowpair(J1,J2,'montage');

% Homomorphic filtering
order=2;

% filter ground truth band
gt=double(imread('./images/ground_truth1.png'));
gt_filter = homo_filter(gt(:,:,1),10,size(gt,1),size(gt,2),order);

% filter reconstructed band
rec=double(imread('./images/reconstructed1.png'));
rec_filter = homo_filter(rec(:,:,1),10,size(rec,1),size(rec,2),order);

subplot 212
imshowpair(gt_filter,rec_filter,'montage');