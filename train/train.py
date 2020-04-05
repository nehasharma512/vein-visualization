from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import  os
import time
import scipy.io as sio

from dataset import DatasetFromHdf5
from resblock import resblock,conv_bn_relu_res_block
from utils import AverageMeter,initialize_logger,save_checkpoint,record_loss
from loss import rrmse_loss

def main():
    
    cudnn.benchmark = True
    # Dataset
    train_data = DatasetFromHdf5('./train_t34bands.h5')
    print(len(train_data))
    val_data = DatasetFromHdf5('./valid_t34bands.h5')
    print(len(val_data))

    # Data Loader (Input Pipeline)
    train_data_loader = DataLoader(dataset=train_data, 
                                   num_workers=1,  
                                   batch_size=64,
                                   shuffle=True,
                                   pin_memory=True)
    val_loader = DataLoader(dataset=val_data,
                            num_workers=1, 
                            batch_size=1,
                            shuffle=False,
                           pin_memory=True)

    # Model   

    model = resblock(conv_bn_relu_res_block,10,3,1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
  
    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 21
    init_lr = 0.0001
    iteration = 0
    record_test_loss = 1000
    criterion = rrmse_loss
    optimizer=torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    
    model_path = './models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss.csv'), 'w+')
    
    log_dir = os.path.join(model_path,'train.log')
    logger = initialize_logger(log_dir)
    
    # Resume
    resume_file = '' 
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
       
    for epoch in range(start_epoch+1, end_epoch):
        
        start_time = time.time()         
        train_loss, iteration, lr = train(train_data_loader, model, criterion, optimizer, iteration, init_lr, end_epoch)
        test_loss = validate(val_loader, model, criterion)
        
 
        
        # Save model
        #if test_loss < record_test_loss:
        #    record_test_loss = test_loss
        #    save_checkpoint(model_path, epoch, iteration, model, optimizer)
        save_checkpoint(model_path, epoch, iteration, model, optimizer)        
        # print loss 
        end_time = time.time()
        epoch_time = end_time - start_time
        print ("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, test_loss))
        # save loss
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss)     
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    
# Training 
def train(train_data_loader, model, criterion, optimizer, iteration, init_lr ,end_epoch):
    losses = AverageMeter()
    for i, (images, labels) in enumerate(train_data_loader):
        labels = labels.cuda(async=True)
        images = images.cuda(async=True)
        images = Variable(images)
        labels = Variable(labels)    
        lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5) 
        iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        #  record loss
        losses.update(loss.item())
            
    return losses.avg, iteration, lr

# Validate
def validate(val_loader, model, criterion):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        with torch.no_grad():
          input_var = torch.autograd.Variable(input)
          target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)      
        loss = criterion(output, target_var)

        #  record loss
        losses.update(loss.item())

    return losses.avg

# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
