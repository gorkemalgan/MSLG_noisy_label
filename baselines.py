import os, sys
import time
import numpy as np
import math, random
import datetime
from collections import OrderedDict
from pathlib import Path

from dataset import get_data, get_dataloader, get_cm, DATASETS_BIG, DATASETS_SMALL
from model_getter import get_model
from utils import *

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def cross_entropy(criterion):   
    test_acc_best = 0
    val_acc_best = 0
    epoch_best = 0

    for epoch in range(PARAMS[dataset]['epochs']):
        start_epoch = time.time()
        # Reset the metrics at the start of the next epoch
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()
        net.train()

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler(epoch)

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            start = time.time()
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running accuracy and loss
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
            train_loss.update(loss.item())

            if verbose == 2:
                sys.stdout.write("Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                                .format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()
                
        # evaluate on validation and test data
        val_accuracy, val_loss = evaluate(net, val_dataloader, criterion)
        test_accuracy, test_loss = evaluate(net, test_dataloader, criterion)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            epoch_best = epoch

        summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
        summary_writer.add_scalar('test_loss', test_loss, epoch)
        summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
        summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
        summary_writer.add_scalar('val_loss', val_loss, epoch)
        summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
        summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
        summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)

        save_model(net, epoch)
        if verbose != 0:
            template = 'Epoch {}, Loss: {:7.4f}, Accuracy: {:5.3f}, Val Loss: {:7.4f}, Val Accuracy: {:5.3f}, Test Loss: {:7.4f}, Test Accuracy: {:5.3f}, lr: {:7.6f} Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, train_loss.avg, train_accuracy.percentage, val_loss, val_accuracy, test_loss, test_accuracy, lr_scheduler(epoch), time.time()-start_epoch, (time.time()-start_epoch)/3600))
        
    print('Train acc: {:5.3f}, Val acc: {:5.3f}-{:5.3f}, Test acc: {:5.3f}-{:5.3f} / Train loss: {:7.4f}, Val loss: {:7.4f}, Test loss: {:7.4f} / Best epoch: {}'.format(
        train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, train_loss.avg, val_loss, test_loss, epoch_best
    ))
    summary_writer.close()
    torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def symmetric_crossentropy():
    """
    2019 - ICCV - Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    github repo: https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
    """
    def criterion(y_pred, y_true):
        y_true_1 = nn.functional.one_hot(y_true ,num_classes)
        y_pred_1 = nn.functional.softmax(y_pred, dim=1)

        y_true_2 = nn.functional.one_hot(y_true ,num_classes)
        y_pred_2 = nn.functional.softmax(y_pred, dim=1)

        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)
        loss1 = -torch.sum(y_true_1 * nn.functional.log_softmax(y_pred_1, dim=1), axis=1)
        loss2 = -torch.sum(y_pred_2 * nn.functional.log_softmax(y_true_2.type(torch.float), dim=1), axis=1)
        per_example_loss = alpha*loss1 + beta*loss2
        return torch.mean(per_example_loss)

    alpha=0.1
    beta=1.0
    num_classes = PARAMS[dataset]['num_classes']
    cross_entropy(criterion)

def generalized_crossentropy():
    """
    2018 - NIPS - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    def criterion(y_pred, y_true):
        q = 0.7
        ytrue_tmp = nn.functional.one_hot(y_true, num_classes)
        ypred_tmp = nn.functional.softmax(y_pred, dim=1)
        t_loss = (1 - torch.pow(torch.sum(ytrue_tmp*ypred_tmp, axis=1), q)) / q
        return torch.mean(t_loss)

    num_classes = PARAMS[dataset]['num_classes']
    cross_entropy(criterion)

def bootstrap_soft():
    """
    2015 - ICLR - Training deep neural networks on noisy labels with bootstrapping.
    github repo: https://github.com/dwright04/Noisy-Labels-with-Bootstrapping
    """
    def criterion(y_pred,y_true):
        y_pred_softmax = nn.functional.softmax(y_pred, dim=1)
        y_true_onehot = nn.functional.one_hot(y_true ,num_classes)
        y_true_modified = beta * y_true_onehot + (1. - beta) * y_pred_softmax
        loss = -torch.sum(y_true_modified * nn.functional.log_softmax(y_pred, dim=1), axis=1)
        return torch.mean(loss)
    
    beta = 0.95
    num_classes = PARAMS[dataset]['num_classes']
    cross_entropy(criterion)

def forwardloss(P):
    """
    2017 - CVPR - Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    github repo: https://github.com/giorgiop/loss-correction
    """
    def criterion(y_pred, y_true):
        y_p = nn.functional.softmax(torch.mm(y_pred, P), dim=1)
        loss = loss_object(y_p, y_true)
        return torch.mean(loss)

    P = torch.tensor(P, dtype=torch.float).to(device)
    loss_object = nn.CrossEntropyLoss()

    cross_entropy(criterion)

def joint_optimization():
    """
    2018 - CVPR - Joint optimization framework for learning with noisy labels.
    github repo: https://github.com/DaikiTanaka-UT/JointOptimization
    """
    sparse_categorical_crossentropy = nn.CrossEntropyLoss()
    def criterion(y_pred, y_true):
        ypred_tmp = nn.functional.softmax(y_pred, dim=1)
        y_pred_avg = torch.mean(ypred_tmp, axis=0)
        l_p = -torch.sum(torch.log(y_pred_avg) * p)
        l_e = -torch.sum(ypred_tmp * nn.functional.log_softmax(ypred_tmp, dim=1), axis=1)
        per_example_loss = sparse_categorical_crossentropy(y_pred,y_true) + 1.2 * l_p + 0.8 * l_e
        return torch.mean(per_example_loss)

    num_classes = PARAMS[dataset]['num_classes']
    p = np.ones(num_classes, dtype=np.float32) / float(num_classes)
    p = torch.tensor(p, dtype=torch.float).to(device)
    cross_entropy(criterion)

def pencil(criterion, alpha, beta, stage1, stage2, stage3, type_lr, lambda1, lambda2, k):
    '''
    2019 - CVPR - Probabilistic End-to-end Noise Correction for Learning with Noisy Labels
    github repo: https://github.com/yikun2019/PENCIL
    '''
    PARAMS_PENCIL = {'mnist_fashion':{'alpha':0.1, 'beta':0.8, 'stage1':3, 'stage2':20, 'stage3':25, 'type_lr':'constant', 'lambda1':400, 'lambda2':0, 'k':10},
                     'cifar10'      :{'alpha':0.1, 'beta':0.8, 'stage1':21,'stage2':80, 'stage3':120, 'type_lr':'constant', 'lambda1':400, 'lambda2':0, 'k':10},
                     'cifar100'     :{'alpha':0.1, 'beta':0.8, 'stage1':21,'stage2':80, 'stage3':120, 'type_lr':'constant', 'lambda1':400, 'lambda2':0, 'k':10},
                     'clothing1M'   :{'alpha':0.08,'beta':0.8, 'stage1':5, 'stage2':10, 'stage3':20, 'type_lr':'two_phase','lambda1':3000, 'lambda2':500, 'k':10},
                     'clothing1M50k':{'alpha':0.08,'beta':0.8, 'stage1':5, 'stage2':10, 'stage3':20, 'type_lr':'two_phase','lambda1':3000, 'lambda2':500, 'k':10},
                     'food101N'     :{'alpha':0.08,'beta':0.8, 'stage1':5, 'stage2':10, 'stage3':20, 'type_lr':'two_phase','lambda1':3000, 'lambda2':500, 'k':10}}
    #set default variables if they are not given from the command line
    if alpha   == None: alpha   = PARAMS_PENCIL[dataset]['alpha']
    if beta    == None: beta    = PARAMS_PENCIL[dataset]['beta']
    if lambda1 == None: lambda1 = PARAMS_PENCIL[dataset]['lambda1']
    if lambda2 == None: lambda2 = PARAMS_PENCIL[dataset]['lambda2']
    if type_lr == None: type_lr = PARAMS_PENCIL[dataset]['type_lr']
    if stage1  == None: stage1  = PARAMS_PENCIL[dataset]['stage1']
    if stage2  == None: stage2  = PARAMS_PENCIL[dataset]['stage2']
    if stage3  == None: stage3  = PARAMS_PENCIL[dataset]['stage3']
    if k       == None: k       = PARAMS_PENCIL[dataset]['k']

    print('alpha: {}, beta:{}, stage1:{}, stage2:{}, stage3:{}, type_lr:{}, lambda1:{}, lambda2:{}, k:{}'.format(alpha, beta, stage1, stage2, stage3, type_lr, lambda1, lambda2, k))
    
    y_file = log_dir + "y.npy"
    meta_lr_scheduler = get_meta_lr_scheduler(type_lr,stage1,stage2,lambda1,lambda2)
    grads_yy_log = None

    test_acc_best = 0
    val_acc_best = 0
    epoch_best = 0

    for epoch in range(stage3): 
        start_epoch = time.time()
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()

         # set learning rate
        meta_lr = meta_lr_scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler(epoch)

        if os.path.isfile(y_file):
            y = np.load(y_file)
        else:
            y = []

        # switch to train mode
        net.train()
        # new y is y_tilde after updating
        new_y = np.zeros([NUM_TRAINDATA,PARAMS[dataset]['num_classes']])

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            start = time.time()
            images, labels = images.to(device), labels.to(device)
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))

            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(labels)

            # compute output
            output = net(input_var)

            logsoftmax = nn.LogSoftmax(dim=1).to(device)
            softmax = nn.Softmax(dim=1).to(device)
            if epoch < stage1:
                # lc is classification loss
                lc = criterion(output, target_var)
                # init y_tilde, let softmax(y_tilde) is noisy labels
                onehot = torch.zeros(labels.size(0), PARAMS[dataset]['num_classes']).to(device).scatter_(1, labels.view(-1, 1), k)
                onehot = onehot.cpu().numpy()
                new_y[index, :] = onehot
            else:
                yy = y
                yy = yy[index,:]
                yy = torch.FloatTensor(yy)
                yy = yy.to(device)
                yy = torch.autograd.Variable(yy,requires_grad = True)
                # obtain label distributions (y_hat)
                last_y_var = softmax(yy)
                lc = torch.mean(softmax(output)*(logsoftmax(output)-torch.log((last_y_var))))
                # lo is compatibility loss
                lo = criterion(last_y_var, target_var)
            # le is entropy loss
            le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

            if epoch < stage1:
                loss = lc
            elif epoch < stage2:
                loss = lc + alpha * lo + beta * le
            else:
                loss = lc

            _, predicted = torch.max(output.data, 1)
            train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
            train_loss.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch >= stage1 and epoch < stage2:
                # update y_tilde by back-propagation
                yy.data.sub_(meta_lr*yy.grad.data)
                new_y[index,:] = yy.data.cpu().numpy()
                if grads_yy_log == None:
                    grads_yy_log = meta_lr*yy.grad.data
                else:
                    grads_yy_log[:len(meta_lr*yy.grad.data)] += meta_lr*yy.grad.data

            if verbose == 2:
                sys.stdout.write("Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                                .format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()

        # save y_tilde
        if epoch < stage2:
            y = new_y
            np.save(y_file,y)

        # evaluate on validation and test data
        val_accuracy, val_loss = evaluate(net, val_dataloader, criterion)
        test_accuracy, test_loss = evaluate(net, test_dataloader, criterion)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            epoch_best = epoch

        summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
        summary_writer.add_scalar('test_loss', test_loss, epoch)
        summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
        summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
        summary_writer.add_scalar('val_loss', val_loss, epoch)
        summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
        summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
        summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)
        if grads_yy_log != None:
            summary_writer.add_histogram('grads_yy', grads_yy_log, epoch)

        if verbose != 0:
            template = 'Epoch {}, Loss: {:7.4f}, Accuracy: {:5.3f}, Val Loss: {:7.4f}, Val Accuracy: {:5.3f}, Test Loss: {:7.4f}, Test Accuracy: {:5.3f}, lr: {:7.6f} Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, train_loss.avg, train_accuracy.percentage, val_loss, val_accuracy, test_loss, test_accuracy, lr_scheduler(epoch), time.time()-start_epoch, (time.time()-start_epoch)/3600))
        save_model(net, epoch)

    print('Train acc: {:5.3f}, Val acc: {:5.3f}-{:5.3f}, Test acc: {:5.3f}-{:5.3f} / Train loss: {:7.4f}, Val loss: {:7.4f}, Test loss: {:7.4f} / Best epoch: {}'.format(
        train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, train_loss.avg, val_loss, test_loss, epoch_best
    ))
    summary_writer.close()
    torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def coteaching(criterion):
    '''
    2018 - NIPS - Co-teaching: Robust training of deep neural networks with extremely noisy labels
    github repo: https://github.com/bhanML/Co-teaching
    '''
    # get model 
    net1 = get_model(dataset,framework).to(device)
    net2 = get_model(dataset,framework).to(device)
    optimizer1 = optim.SGD(net1.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)

    train_loss1 = AverageMeter()
    train_accuracy1 = AverageMeter()
    val_loss1 = AverageMeter()
    val_accuracy1 = AverageMeter()
    test_loss1 = AverageMeter()
    test_accuracy1 = AverageMeter()

    train_loss2 = AverageMeter()
    train_accuracy2 = AverageMeter()
    val_loss2 = AverageMeter()
    val_accuracy2 = AverageMeter()
    test_loss2 = AverageMeter()
    test_accuracy2 = AverageMeter()

    # calculate forget rates for each epoch (from origianl code)
    forget_rate=0.2
    num_graduals=10
    exponent=0.2

    forget_rates = np.ones(PARAMS[dataset]['epochs'])*forget_rate
    forget_rates[:num_graduals] = np.linspace(0, forget_rate**exponent, num_graduals)

    test_acc_best = 0
    val_acc_best = 0
    epoch_best = 0

    for epoch in range(PARAMS[dataset]['epochs']):
        start_epoch = time.time()
        # Reset the metrics at the start of the next epoch
        train_loss1.reset()
        train_accuracy1.reset()
        train_loss2.reset()
        train_accuracy2.reset()
        remember_rate = 1 - forget_rates[epoch]

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr_scheduler(epoch)
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr_scheduler(epoch)

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            start = time.time()
            images, labels = images.to(device), labels.to(device)
            num_remember = int(remember_rate * BATCH_SIZE)
            
            with torch.no_grad():
                # select samples based on model 1
                net1.eval()
                y_pred1 = F.softmax(net1(images))
                cross_entropy = F.cross_entropy(y_pred1, labels, reduce=False)
                batch_idx1= np.argsort(cross_entropy.cpu().numpy())[:num_remember]
                # select samples based on model 2
                net2.eval()
                y_pred2 = F.softmax(net2(images))
                cross_entropy = F.cross_entropy(y_pred2, labels, reduce=False)
                batch_idx2 = np.argsort(cross_entropy.cpu().numpy())[:num_remember]

            # train net1
            net1.train()
            optimizer1.zero_grad()
            outputs = net1(images[batch_idx2,:])
            loss1 = criterion(outputs, labels[batch_idx2])
            loss1.backward()
            optimizer1.step()
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy1.update(predicted.eq(labels[batch_idx2].data).cpu().sum().item(), labels.size(0)) 
            train_loss1.update(loss1.item(), images.size(0))
            # train net2
            net2.train()
            optimizer2.zero_grad()
            outputs = net2(images[batch_idx1,:])
            loss2 = criterion(outputs, labels[batch_idx1])
            loss2.backward()
            optimizer2.step()
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy2.update(predicted.eq(labels[batch_idx1].data).cpu().sum().item(), labels.size(0)) 
            train_loss2.update(loss2.item(), images.size(0))

            if verbose == 2:
                sys.stdout.write("Progress: {:6.5f}, Accuracy1: {:5.4f}, Loss1: {:5.4f}, Accuracy2: {:5.4f}, Loss2: {:5.4f}, Process time:{:5.4f}   \r"
                             .format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, 
                             train_accuracy1.avg, train_loss1.avg, train_accuracy2.avg, train_loss2.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()

        # evaluate on validation and test data
        val_accuracy1, val_loss1 = evaluate(net1, val_dataloader, criterion)
        test_accuracy1, test_loss1 = evaluate(net1, test_dataloader, criterion)
        # evaluate on validation and test data
        val_accuracy2, val_loss2 = evaluate(net2, val_dataloader, criterion)
        test_accuracy2, test_loss2 = evaluate(net2, test_dataloader, criterion)

        if max(val_accuracy1, val_accuracy2) > val_acc_best: 
            val_acc_best = max(val_accuracy1, val_accuracy2)
            test_acc_best = max(test_accuracy1, test_accuracy2)
            epoch_best = epoch

        summary_writer.add_scalar('train_loss1', train_loss1.avg, epoch)
        summary_writer.add_scalar('train_accuracy1', train_accuracy1.percentage, epoch)
        summary_writer.add_scalar('val_loss1', val_loss1, epoch)
        summary_writer.add_scalar('val_accuracy1', val_accuracy1, epoch)
        summary_writer.add_scalar('test_loss1', test_loss1, epoch)
        summary_writer.add_scalar('test_accuracy1', test_accuracy1, epoch)
        summary_writer.add_scalar('train_loss2', train_loss2.avg, epoch)
        summary_writer.add_scalar('train_accuracy2', train_accuracy2.percentage, epoch)
        summary_writer.add_scalar('val_loss2', val_loss2, epoch)
        summary_writer.add_scalar('val_accuracy2', val_accuracy2, epoch)
        summary_writer.add_scalar('test_loss2', test_loss2, epoch)
        summary_writer.add_scalar('test_accuracy2', test_accuracy2, epoch)

        summary_writer.add_scalar('train_loss', min(train_loss1.avg, train_loss2.avg), epoch)
        summary_writer.add_scalar('train_accuracy', max(train_accuracy1.percentage, train_accuracy2.percentage), epoch)
        summary_writer.add_scalar('val_loss', min(val_loss1, val_loss2), epoch)
        summary_writer.add_scalar('val_accuracy', max(val_accuracy1, val_accuracy2), epoch)
        summary_writer.add_scalar('test_loss', min(test_loss1, test_loss2), epoch)
        summary_writer.add_scalar('test_accuracy', max(test_accuracy1, test_accuracy2), epoch)
        summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
        summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)

        if verbose != 0:
            end = time.time()
            template = 'Model1 - Epoch {}, Loss: {:7.4f}, Accuracy: {:5.3f}, Val Loss: {:7.4f}, Val Accuracy: {:5.3f}, Test Loss: {:7.4f}, Test Accuracy: {:5.3f}, lr: {:7.6f} Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1,
                                    train_loss1.avg,
                                    train_accuracy1.avg,
                                    val_loss1,
                                    val_accuracy1,
                                    test_loss1,
                                    test_accuracy1,
                                    lr_scheduler(epoch),
                                    end-start_epoch, (end-start_epoch)/3600))
            template = 'Model2 - Epoch {}, Loss: {:7.4f}, Accuracy: {:5.3f}, Val Loss: {:7.4f}, Val Accuracy: {:5.3f}, Test Loss: {:7.4f}, Test Accuracy: {:5.3f}, lr: {:7.6f} Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1,
                                train_loss2.avg,
                                train_accuracy2.avg,
                                val_loss2,
                                val_accuracy2,
                                test_loss2,
                                test_accuracy2,
                                lr_scheduler(epoch),
                                end-start_epoch, (end-start_epoch)/3600))
        save_model(net1, epoch, 'model1')
        save_model(net2, epoch, 'model2')

    print('Train acc: {:5.3f}, Val acc: {:5.3f}-{:5.3f}, Test acc: {:5.3f}-{:5.3f} / Train loss: {:7.4f}, Val loss: {:7.4f}, Test loss: {:7.4f} / Best epoch: {}'.format(
        max(train_accuracy1.percentage, train_accuracy2.percentage), max(val_accuracy1, val_accuracy2), val_acc_best, max(test_accuracy1, test_accuracy2), test_acc_best, 
        min(train_loss1.avg, train_loss2.avg),  min(val_loss1, val_loss2), min(test_loss1, test_loss2), epoch_best
    ))
    torch.save(net1.state_dict(), os.path.join(log_dir, 'saved_model1.pt'))
    torch.save(net2.state_dict(), os.path.join(log_dir, 'saved_model2.pt'))

def metaweightnet():
    '''
    2019 - NIPS - Meta-weight-net: Learning an explicit mapping for sample weighting
    github repo: https://github.com/xjtushujun/meta-weight-net
    '''
    train_meta_loader = val_dataloader

    class VNet(nn.Module):
        def __init__(self, input, hidden, output):
            super(VNet, self).__init__()
            self.linear1 = nn.Linear(input, hidden)
            self.relu1 = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(hidden, output)

        def forward(self, x, weights=None):
            if weights == None:
                x = self.linear1(x)
                x = self.relu1(x)
                out = self.linear2(x)
                return torch.sigmoid(out)
            else:
                x = F.linear(x, weights['fc1.weight'], weights['fc1.bias'])   
                feat = F.threshold(x, 0, 0, inplace=True)
                x = F.linear(feat, weights['fc2.weight'], weights['fc2.bias'])
                return torch.sigmoid(out)
    vnet = VNet(1, 100, 1).to(device)
    optimizer_vnet = torch.optim.Adam(vnet.parameters(), 1e-3, weight_decay=1e-4)

    test_acc_best = 0
    val_acc_best = 0
    epoch_best = 0

    for epoch in range(PARAMS[dataset]['epochs']): 
        start_epoch = time.time()
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()
        meta_loss = AverageMeter()

         # set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler(epoch)

        train_meta_loader_iter = iter(train_meta_loader)
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            start = time.time()

            net.train()
            inputs, targets = inputs.to(device), targets.to(device)
            meta_model = get_model(dataset,framework).to(device)
            meta_model.load_state_dict(net.state_dict())
            outputs = meta_model(inputs)

            cost = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda = vnet(cost_v.data)
            l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)
            meta_model.zero_grad()
            
            #grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True)
            #meta_model.update_params(lr_inner=lr_scheduler(epoch), source_params=grads)
            grads = torch.autograd.grad(l_f_meta, meta_model.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            fast_weights = OrderedDict((name, param - lr_scheduler(epoch)*grad) for ((name, param), grad) in zip(meta_model.named_parameters(), grads))

            try:
                inputs_val, targets_val = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                inputs_val, targets_val = next(train_meta_loader_iter)
            inputs_val, targets_val = inputs_val.to(device), targets_val.type(torch.long).to(device)
            #y_g_hat = meta_model(inputs_val)
            y_g_hat = meta_model.forward(inputs_val,fast_weights)  
            l_g_meta = F.cross_entropy(y_g_hat, targets_val)

            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()

            outputs = net(inputs)
            cost_w = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))

            with torch.no_grad():
                w_new = vnet(cost_v)

            loss = torch.sum(cost_v * w_new)/len(cost_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del grads

            _, predicted = torch.max(outputs.data, 1)
            train_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
            train_loss.update(loss.item())
            meta_loss.update(l_g_meta.item(), targets.size(0))

            if verbose == 2:
                sys.stdout.write("Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                                .format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()

        # evaluate on validation and test data
        val_accuracy, val_loss = evaluate(net, val_dataloader, F.cross_entropy)
        test_accuracy, test_loss = evaluate(net, test_dataloader, F.cross_entropy)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            epoch_best = epoch

        summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
        summary_writer.add_scalar('test_loss', test_loss, epoch)
        summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
        summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
        summary_writer.add_scalar('val_loss', val_loss, epoch)
        summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
        summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
        summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)

        if verbose != 0:
            template = 'Epoch {}, Loss: {:7.4f}, Accuracy: {:5.3f}, Val Loss: {:7.4f}, Val Accuracy: {:5.3f}, Test Loss: {:7.4f}, Test Accuracy: {:5.3f}, lr: {:7.6f} Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, train_loss.avg, train_accuracy.percentage, val_loss, val_accuracy, test_loss, test_accuracy, lr_scheduler(epoch), time.time()-start_epoch, (time.time()-start_epoch)/3600))
        save_model(net, epoch)

    print('Train acc: {:5.3f}, Val acc: {:5.3f}-{:5.3f}, Test acc: {:5.3f}-{:5.3f} / Train loss: {:7.4f}, Val loss: {:7.4f}, Test loss: {:7.4f} / Best epoch: {}'.format(
        train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, train_loss.avg, val_loss, test_loss, epoch_best
    ))
    summary_writer.close()
    torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def mlnt(criterion, consistent_criterion, start_iter=500, mid_iter = 2000, eps=0.99, args_alpha=1,num_fast=10,perturb_ratio=0.5,meta_lr=0.2):
    '''
    2019 - CVPR - Learning to Learn from Noisy Labeled Data
    github repo: https://github.com/LiJunnan1992/MLNT
    '''
    # get model 
    path = Path(log_dir)
    tch_net = get_model(dataset,framework).to(device)
    pretrain_net = get_model(dataset,framework).to(device)

    ce_base_folder = os.path.join(path.parent.parent, 'cross_entropy')
    for f in os.listdir(ce_base_folder):
        ce_path = os.path.join(ce_base_folder, f, 'saved_model.pt')
        if os.path.exists(ce_path):
            print('Loading base model from: {}'.format(ce_path))
            pretrain_net.load_state_dict(torch.load(ce_path, map_location=device))
            break

    # tensorboard
    summary_writer = SummaryWriter(log_dir)
    init = True

    test_acc_best = 0
    val_acc_best = 0
    epoch_best = 0

    for epoch in range(PARAMS[dataset]['epochs']): 
        start_epoch = time.time()
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()

        net.train()
        tch_net.train()
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler(epoch)
        
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            start = time.time()
            inputs, targets = inputs.to(device), targets.to(device) 
            optimizer.zero_grad()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = net(inputs)               # Forward Propagation        
            
            class_loss = criterion(outputs, targets)  # Loss
            class_loss.backward(retain_graph=True)  

            if batch_idx>start_iter or epoch>1:
                if batch_idx>mid_iter or epoch>1:
                    eps=0.999
                    alpha = args_alpha
                else:
                    u = (batch_idx-start_iter)/(mid_iter-start_iter)
                    alpha = args_alpha*math.exp(-5*(1-u)**2)          
            
                if init:
                    init = False
                    for param,param_tch in zip(net.parameters(),tch_net.parameters()): 
                        param_tch.data.copy_(param.data)                    
                else:
                    for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                        param_tch.data.mul_(eps).add_((1-eps), param.data)   
                
                _,feats = pretrain_net(inputs,get_feat=True)
                tch_outputs = tch_net(inputs,get_feat=False)
                p_tch = F.softmax(tch_outputs,dim=1)
                p_tch.detach_()
                
                for i in range(num_fast):
                    targets_fast = targets.clone()
                    randidx = torch.randperm(targets.size(0))
                    for n in range(int(targets.size(0)*perturb_ratio)):
                        num_neighbor = 10
                        idx = randidx[n]
                        feat = feats[idx]
                        feat.view(1,feat.size(0))
                        feat.data = feat.data.expand(targets.size(0),feat.size(0))
                        dist = torch.sum((feat-feats)**2,dim=1)
                        _, neighbor = torch.topk(dist.data,num_neighbor+1,largest=False)
                        targets_fast[idx] = targets[neighbor[random.randint(1,num_neighbor)]]
                        
                    fast_loss = criterion(outputs,targets_fast)

                    grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                    for grad in grads:
                        grad.detach()
                        #grad.detach_()
                        #grad.requires_grad = False  
    
                    fast_weights = OrderedDict((name, param - meta_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                    
                    fast_out = net.forward(inputs,fast_weights)  
        
                    logp_fast = F.log_softmax(fast_out,dim=1)                
                    consistent_loss = consistent_criterion(logp_fast,p_tch)
                    consistent_loss = consistent_loss*alpha/num_fast 
                    consistent_loss.backward(retain_graph=True)
                    del grads, fast_weights
                    
            optimizer.step() # Optimizer update 

            _, predicted = torch.max(outputs.data, 1)
            train_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
            train_loss.update(class_loss.item())

            if verbose == 2:
                sys.stdout.write("Progress: {:6.5f}, Accuracy: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                                .format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()
                
        # evaluate on validation and test data
        val_accuracy, val_loss = evaluate(net, val_dataloader, criterion)
        test_accuracy, test_loss = evaluate(net, test_dataloader, criterion)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            epoch_best = epoch

        summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
        summary_writer.add_scalar('test_loss', test_loss, epoch)
        summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
        summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
        summary_writer.add_scalar('val_loss', val_loss, epoch)
        summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
        summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
        summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)

        if verbose != 0:
            template = 'Epoch {}, Loss: {:7.4f}, Accuracy: {:5.3f}, Val Loss: {:7.4f}, Val Accuracy: {:5.3f}, Test Loss: {:7.4f}, Test Accuracy: {:5.3f}, lr: {:7.6f} Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, train_loss.avg, train_accuracy.percentage, val_loss, val_accuracy, test_loss, test_accuracy, lr_scheduler(epoch), time.time()-start_epoch, (time.time()-start_epoch)/3600))
        save_model(net, epoch)

    print('Train acc: {:5.3f}, Val acc: {:5.3f}-{:5.3f}, Test acc: {:5.3f}-{:5.3f} / Train loss: {:7.4f}, Val loss: {:7.4f}, Test loss: {:7.4f} / Best epoch: {}'.format(
        train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, train_loss.avg, val_loss, test_loss, epoch_best
    ))
    summary_writer.close()
    torch.save(tch_net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def evaluate(net, dataloader, criterion):
    eval_accuracy = AverageMeter()
    eval_loss = AverageMeter()

    net.eval()
    if dataloader:
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs) 
                loss = criterion(outputs, targets) 
                _, predicted = torch.max(outputs.data, 1) 
                eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
                eval_loss.update(loss.item())
    return eval_accuracy.percentage, eval_loss.avg

def save_model(model, epoch, model_name='model'):
    if dataset in DATASETS_BIG:
        save_path = log_dir+'models/'
        save_name = model_name + '_e{}.pt'.format(epoch)
        create_folder(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, save_name))

def main(args):
    start_train = time.time()
    # create necessary folders
    create_folder('{}/dataset'.format(dataset))
    create_folder(log_dir)

    criterion = nn.CrossEntropyLoss().to(device)

    if model_name == 'cross_entropy':
        cross_entropy(criterion)
    elif model_name == 'symmetric_crossentropy':
        symmetric_crossentropy()
    elif model_name == 'generalized_crossentropy':
        generalized_crossentropy()
    elif model_name == 'bootstrap_soft':
        bootstrap_soft()
    elif model_name == 'forwardloss':
        P = get_cm(dataset, noise_type, noise_ratio,framework)
        forwardloss(P)
    elif model_name == 'joint_optimization':
        joint_optimization()
    elif model_name == 'pencil':
        pencil(criterion, args.alpha, args.beta, args.stage1, args.stage2, args.stage3, args.type_lr, args.lambda1, args.lambda2, args.k)
    elif model_name == 'coteaching':
        coteaching(criterion)
    elif model_name == 'mwnet':
        metaweightnet()
    elif model_name == 'mlnt':
        consistent_criterion = nn.KLDivLoss()
        mlnt(criterion, consistent_criterion)

    print('Totail training duration: {:3.2f}h'.format((time.time()-start_train)/3600))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=False, type=str, default='cifar10',
        help="Dataset to use; either 'mnist_fashion', 'cifar10', 'cifar100', 'food101N', 'clothing1M'")
    parser.add_argument('-m', '--model_name', required=False, type=str, default='cross_entropy',
        help="""Model name: 'cross_entropy', 
                            'symmetric_crossentropy', 
                            'generalized_crossentropy', 
                            'bootstrap_soft', 
                            'forwardloss', 
                            'joint_optimization', 
                            'pencil',
                            'coteaching',
                            'mwnet',
                            'mlnt'""")
    parser.add_argument('-n', '--noise_type', required=False, type=str, default='symmetric',
        help="Noise type for cifar10: 'feature-dependent', 'symmetric'")
    parser.add_argument('-r', '--noise_ratio', required=False, type=int, default=35,
        help="Synthetic noise ratio in percentage between 0-100")
    parser.add_argument('-s', '--batch_size', required=False, type=int,
        help="Number of gpus to be used")
    parser.add_argument('-i', '--gpu_ids', required=False, type=int, nargs='+', action='append',
        help="GPU ids to be used")
    parser.add_argument('-f', '--folder_log', required=False, type=str,
        help="Folder name for logs")
    parser.add_argument('-v', '--verbose', required=False, type=int, default=0,
        help="Details of prints: 0(silent), 1(not silent)")
    parser.add_argument('-w', '--num_workers', required=False, type=int,
        help="Number of parallel workers to parse dataset")
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")

    # PENCIL parameters
    parser.add_argument('-a', '--alpha', required=False, type=float,
        help="Alpha parameter")
    parser.add_argument('-b', '--beta', required=False, type=float,
        help="Beta paramter")
    parser.add_argument('-l1', '--lambda1', required=False, type=int,
        help="Learning rate for meta learning phase")
    parser.add_argument('-l2', '--lambda2', required=False, type=int,
        help="Learning rate2 for meta learning phase. Used only when --type is two_phase")
    parser.add_argument('-t', '--type_lr', required=False, type=str,
        help='''Meta learning rate scheduler type: 
                constant - metalearning rate is constant throughout stage2, 
                linear_decrease - meta learning rate is decreased to 0 throughout stage2
                two_phase - in first half of stage2 l1 is used and for second half l2''')
    parser.add_argument('-s1', '--stage1', required=False, type=int,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int,
        help="Epoch num to end stage2 (meta training)")
    parser.add_argument('-s3', '--stage3', required=False, type=int,
        help="Epoch num to end stage3 and full training (finetuning)")
    parser.add_argument('-k', required=False, type=int, default=10,
        help="")

    args = parser.parse_args()
    # configuration variables
    framework = 'pytorch'
    dataset = args.dataset
    model_name = args.model_name
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio/100
    BATCH_SIZE = args.batch_size if args.batch_size != None else PARAMS[dataset]['batch_size']
    verbose = args.verbose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_ids is None:
        ngpu = torch.cuda.device_count() if device.type == 'cuda' else 0  
        gpu_ids = list(range(ngpu)) 
    else:
        gpu_ids = args.gpu_ids[0]
        ngpu = len(gpu_ids)
        if ngpu == 1: 
            device = torch.device("cuda:{}".format(gpu_ids[0]))
    if args.num_workers is None:
        num_workers = 2 if ngpu < 2 else ngpu*2
    else:
        num_workers = args.num_workers

    base_folder = model_name if dataset in DATASETS_BIG else noise_type + '/' + str(args.noise_ratio) + '/' + model_name
    log_folder = args.folder_log if args.folder_log else current_time
    log_base = '{}/logs/{}/'.format(dataset, base_folder)
    log_dir = log_base + log_folder + '/'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Dataset: {}, Model: {}, Device: {}, Batch size: {}, #GPUS to run: {}".format(dataset, model_name, device, BATCH_SIZE, ngpu))
    if dataset in DATASETS_SMALL:
        print("Noise type: {}, Noise ratio: {}".format(noise_type, noise_ratio))

    # global variables
    net = get_model(dataset,framework).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    lr_scheduler = get_lr_scheduler(dataset)
    optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)
    summary_writer = SummaryWriter(log_dir)
    train_dataloader, val_dataloader, test_dataloader, class_names = get_dataloader(dataset, BATCH_SIZE,framework,noise_type,noise_ratio,args.seed,num_workers)
    NUM_TRAINDATA = len(train_dataloader.dataset)

    main(args)