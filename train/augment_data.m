hyper_dir = '../dataset/veins_t34bands/train_data/mat/';
dat=dir(fullfile(hyper_dir,'*.mat'));
order= randperm(size(dat,1));


rgb_dir = '../dataset/veins_t34bands/train_data/rgb/';


num = 247;

for i = 1:size(dat,1)
    
    % load mat file
    mat = [hyper_dir dat(order(i)).name];
    rad = load(mat,'rad');
    rad = cell2mat(struct2cell(rad));
    % load rgb image
    png=[rgb_dir 'REFLECTANCE_' dat(order(i)).name(1:end-4) '.png'];
    im = imread(png);
    
    % flip original image
    temp_rad = rad;
    rad = flip(rad);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');

    temp_im = im;
    im = flip(im);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);
    
    num = num+1;
    rad = temp_rad;
    im = temp_im;

    % rotate 90
    rad = rot90(rad,1);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');
    
    
    im= imrotate(im,90);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);
    
    num = num+1;
    
    % flip rotate 90 image
    temp_rad = rad;
    rad = flip(rad);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');

    temp_im = im;
    im = flip(im);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);

    num = num+1;
    rad = temp_rad;
    im = temp_im;

    
    % rotate 180
    rad = rot90(rad,1);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');
    
    
    im= imrotate(im,90);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);
    
    num = num+1;
    
    % flip rotate 180 image
    temp_rad = rad;
    rad = flip(rad);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');

    temp_im = im;
    im = flip(im);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);

    num = num+1;
    rad = temp_rad;
    im = temp_im;

    
    % rotate 270
    rad = rot90(rad,1);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');
    
    
    im= imrotate(im,90);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);
    
    num = num+1;
    
    % flip rotate 270 image
    temp_rad = rad;
    rad = flip(rad);
    mat1 = [hyper_dir num2str(num)];
    save(mat1,'rad');

    temp_im = im;
    im = flip(im);
    png1 = [rgb_dir 'REFLECTANCE_' num2str(num) '.png'];
    imwrite(im,png1);

    num = num+1;
    rad = temp_rad;
    im = temp_im;
    
end
