global FILE_COUNT;
global TOTALCT;
global CREATED_FLAG;

%string='train';
string='valid';
if strcmp(string, 'train') == 1

    hyper_dir = '../dataset/veins_t34bands/train_data/mat/';
    label=dir(fullfile(hyper_dir,'*.mat'));

    rgb_dir = '../dataset/veins_t34bands/train_data/rgb/';
    order= randperm(size(label,1));
   
else

    hyper_dir = '../dataset/veins_t34bands/valid_data/mat/';
    label=dir(fullfile(hyper_dir,'*.mat'));
    
    rgb_dir = '../dataset/veins_t34bands/valid_data/rgb/';
    order= randperm(size(label,1));
    
end  

%% Initialization the patch and stride
size_input=50;
size_label=50;
label_dimension=34;
data_dimension=3;
stride=50;


%% Initialization the hdf5 parameters
prefix=[string '_t34bands'];
chunksz=64;
TOTALCT=0;
FILE_COUNT=0;
amount_hd5_image=50000;
CREATED_FLAG=false;


%% For loop  RGB-HS-HD5  
for i=1:size(label,1)
     if mod(i,amount_hd5_image)==1     
         filename=getFilename(label(order(i)).name,prefix,hyper_dir);
     end
    name_label=strcat(hyper_dir,label(order(i)).name); 
    a_temp=struct2cell(load(name_label,'rad'));
    hs_label=cell2mat(a_temp);
    hs_label=hs_label/(2^12-1);

    rgb_name=[ rgb_dir 'REFLECTANCE_' label(order(i)).name(1:end-4) '.png'];
    
    rgb_data_uint=imread(rgb_name);
    rgb_data=im2double(rgb_data_uint);
       
    for j=1:label_dimension
           ConvertHStoNbands(rgb_data,hs_label(:,:,j),size_input,size_label,1,data_dimension,stride,chunksz,amount_hd5_image,filename)
    end
end       
 

function filename_change=getFilename(filename,prefix,folder_label)
       filename_change=filename;
       filename_change=[prefix filename_change];
       filename_change=filename_change(1:end-4);
       filename_change=strcat(filename_change,'.h5');
end

