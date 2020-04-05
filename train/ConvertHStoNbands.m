 function   ConvertHStoNbands(local_HS_data,local_HS_label,local_size_data,local_size_label,label_dimension,data_dimension,local_stride,chunksz,local_amount_hd5_image,filename)
%% global variable to determine the numbers of images included by each hdf5 file
global FILE_COUNT;
global TOTALCT;
global CREATED_FLAG;

%% initialization
data = zeros(local_size_data,local_size_data, data_dimension, 1);                    
label = zeros(local_size_data, local_size_data, label_dimension, 1);
padding = abs(local_size_data - local_size_label)/2;
count = 0;

% local_HS_label=local_HS_label/max(local_HS_label(:));                                 %  normalize HS_Label
FILE_COUNT=FILE_COUNT+1;
if FILE_COUNT >local_amount_hd5_image 
    FILE_COUNT=1;
    CREATED_FLAG = false;
    TOTALCT=0;
end



%% loading all the .mat will need a for loop   
[img_width,img_height,img_channel] = size(local_HS_label);                 % the choosed label and data are same size


   for x = 1 : local_stride : img_width-local_size_data+1
          for y = 1 :local_stride : img_height-local_size_data+1
            
             subim_input = local_HS_data(x : x+local_size_data-1, y : y+local_size_data-1,:);
             
             subim_label = local_HS_label(x+padding : x+padding+local_size_label-1, y+padding : y+padding+local_size_label-1,:);

             count=count+1;
             data(:, :,:,count) = subim_input;          
             label(:, :,:,count) = subim_label;
           end
   end  
   
order = randperm(count);                               
data = data(:, :, :, order);
label = label(:, :, :, order);   

 clear subim_input;
 clear subim_label;
        
 %% writing to HDF5
             for batchno = 1:floor(count/chunksz)
                 last_read=(batchno-1)*chunksz;
                 batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
                 batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

                 startloc = struct('dat',[1,1,1,TOTALCT+1], 'lab', [1,1,1,TOTALCT+1]);
                 curr_dat_sz = store2hdf5(filename, batchdata, batchlabs, ~CREATED_FLAG, startloc, chunksz);   % the flag affects whether append to the previous file
                 CREATED_FLAG = true;                                                                          
                 TOTALCT = curr_dat_sz(end);
                               
             end
    
             h5disp(filename);
             






