% Homomorphic filtering, Using Butterworth High Pass Filter.

order=2;

% filter ground truth band
gt=double(imread('./images/ground_truth1.png'));
gt_filter = homo_filter(gt(:,:,1),10,size(gt,1),size(gt,2),order);

% filter reconstructed band
rec=double(imread('./images/reconstructed1.png'));
rec_filter = homo_filter(rec(:,:,1),10,size(rec,1),size(rec,2),order);

imshowpair(gt_filter,rec_filter,'montage');


function filter_image = homo_filter(image,d,row,col,order)
%Butterworth high pass filter
A=zeros(row,col);
for i=1:row
    for j=1:col
        A(i,j)=(((i-row/2).^2+(j-col/2).^2)).^(.5);
        H(i,j)=1/(1+((d/A(i,j))^(2*order)));
    end
end

alphaL=.0999;
aplhaH=2.51;
H=((aplhaH-alphaL).*H)+alphaL;
H=1-H;

%log of image
filter_image=log2(1+image);

%DFT of logged image
filter_image=fft2(filter_image);

%Filter Applying DFT image
filter_image=H.*filter_image;

%Inverse DFT of filtered image
filter_image=abs(ifft2(filter_image));

%Inverse log 
filter_image=exp(filter_image);
end
