import os, sys
import time
import numpy as np
import math, random
import datetime
from collections import OrderedDict
import itertools
import gc

from dataset import get_data, get_dataloader, get_synthetic_idx, DATASETS_BIG, DATASETS_SMALL
from model_getter import get_model
from utils import *

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PARAMS_META = {'mnist_fashion'     :{'alpha':0.5, 'beta':4000, 'gamma':1, 'stage1':1, 'stage2':20, 'k':10},
               'cifar10'           :{'alpha':0.5, 'beta':4000, 'gamma':1, 'stage1':44,'stage2':120,'k':10},
               'cifar100'          :{'alpha':0.5, 'beta':4000, 'gamma':1, 'stage1':21,'stage2':60, 'k':10},
               'clothing1M'        :{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':10},
               'clothing1M50k'     :{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':10},
               'clothing1Mbalanced':{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':10},
               'food101N'          :{'alpha':0.5, 'beta':1500, 'gamma':1, 'stage1':1, 'stage2':10, 'k':10}}

def metapencil(alpha, beta, gamma, stage1, stage2, K):
    def warmup_training():
        loss = criterion_cce(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def meta_training():
        # meta training for predicted labels
        y_softmaxed = softmax(yy)
        lc = criterion_meta(output, y_softmaxed)                                            # classification loss
        # train for classification loss with meta-learning
        net.zero_grad()
        grads = torch.autograd.grad(lc, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads:
            grad.detach()
        fast_weights = OrderedDict((name, param - alpha*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))  
        fast_out = net.forward(images_meta,fast_weights)   
        loss_meta = criterion_cce(fast_out, labels_meta)
        grads_yy = torch.autograd.grad(loss_meta, yy, create_graph=False, retain_graph=True, only_inputs=True)
        for grad in grads_yy:
            grad.detach()
        meta_grads = beta*grads_yy[0]
        # update labels
        yy.data.sub_(meta_grads)
        new_y[index,:] = yy.data.cpu().numpy()
        del grads, grads_yy

        # training base network
        y_softmaxed = softmax(yy)
        lc = criterion_meta(output, y_softmaxed)                        # classification loss
        le = -torch.mean(torch.mul(softmax(output), logsoftmax(output)))# entropy loss
        loss = lc + gamma*le                                            # overall loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss, meta_grads

    print('use_clean:{}, alpha:{}, beta:{}, gamma:{}, stage1:{}, stage2:{}, K:{}'.format(use_clean_data, alpha, beta, gamma, stage1, stage2, K))

    NUM_TRAINDATA = len(train_dataset)
    t_dataset, m_dataset, t_dataloader, m_dataloader = train_dataset, meta_dataset, train_dataloader, meta_dataloader
    # loss functions
    criterion_cce = nn.CrossEntropyLoss()
    criterion_meta = lambda output, labels: torch.mean(softmax(output)*(logsoftmax(output+1e-10)-torch.log(labels+1e-10)))

    # initialize predicted labels with given labels multiplied with a constant
    y_init_path = '{}/y_{}_{}_{}_{}.npy'.format(dataset,noise_type,noise_ratio,NUM_TRAINDATA,stage1)
    labels_yy = np.zeros(NUM_TRAINDATA)
    if not os.path.exists(y_init_path):
        new_y = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
        for batch_idx, (images, labels) in enumerate(t_dataloader):
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), K).cpu().numpy()
            new_y[index, :] = onehot
        if not os.path.exists(y_init_path):
            np.save(y_init_path,new_y)
    else:
        new_y = np.load(y_init_path)
    test_acc_best = 0
    val_acc_best = 0
    top5_acc_best = 0
    top1_acc_best = 0
    epoch_best = 0

    model_s1_path = '{}/{}_{}_{}_{}_{}.pt'.format(dataset,dataset,noise_type,noise_ratio,NUM_TRAINDATA,stage1)
    if os.path.exists(model_s1_path):
        net.load_state_dict(torch.load(model_s1_path, map_location=device))

    for epoch in range(stage2): 
        start_epoch = time.time()
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()
        train_accuracy_meta = AverageMeter()
        label_similarity = AverageMeter()

        lr = lr_scheduler(epoch)
        set_learningrate(optimizer, lr)
        net.train()
        grads_dict = OrderedDict((name, 0) for (name, param) in net.named_parameters()) 

        # skip straightforward training if there is a pretrained model already
        if os.path.exists(model_s1_path) and epoch < stage1:
            continue
        # if no use clean data, extract reliable data for meta subset
        if epoch == stage1:
            if not os.path.exists(model_s1_path):
                torch.save(net.cpu().state_dict(), model_s1_path)
                net.to(device)            
            if use_clean_data == 0:
                t_dataset, m_dataset, t_dataloader, m_dataloader = get_dataloaders_meta()
                NUM_TRAINDATA = len(t_dataset)
                labels_yy = np.zeros(NUM_TRAINDATA)
                new_y = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
                for batch_idx, (images, labels) in enumerate(t_dataloader):
                    index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
                    onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), K).cpu().numpy()
                    new_y[index, :] = onehot
            t_meta_loader_iter = iter(m_dataloader)
        y_hat = new_y.copy()
        meta_grads_yy_log = np.zeros((NUM_TRAINDATA,NUM_CLASSES))

        for batch_idx, (images, labels) in enumerate(t_dataloader):
            start = time.time()
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
            
            # training images and labels
            images, labels = images.to(device), labels.to(device)
            images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
            # predicted underlying true label distribution
            yy = torch.FloatTensor(y_hat[index,:]).to(device)
            yy = torch.autograd.Variable(yy,requires_grad = True)

            # compute output
            output = net(images)

            if epoch < stage1:
                loss = warmup_training()
            else:
                # meta training images and labels
                try:
                    images_meta, labels_meta = next(t_meta_loader_iter)
                except StopIteration:
                    t_meta_loader_iter = iter(m_dataloader)
                    images_meta, labels_meta = next(t_meta_loader_iter)
                    images_meta, labels_meta = images_meta[:labels.size(0)], labels_meta[:labels.size(0)]
                #labels_meta = torch.zeros(labels_meta.size(0), NUM_CLASSES).scatter_(1, labels_meta.view(-1, 1), 1)
                images_meta, labels_meta = images_meta.to(device), labels_meta.to(device)
                images_meta, labels_meta = torch.autograd.Variable(images_meta), torch.autograd.Variable(labels_meta)
                
                #with torch.autograd.detect_anomaly():
                loss, meta_grads_yy = meta_training()
                meta_grads_yy_log[index] = meta_grads_yy.cpu().detach().numpy()

            _, labels_yy[index] = torch.max(yy.cpu(), 1)
            _, predicted = torch.max(output.data, 1)
            train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) 
            train_loss.update(loss.item())
            train_accuracy_meta.update(predicted.eq(torch.tensor(labels_yy[index]).to(device)).cpu().sum().item(), predicted.size(0)) 
            label_similarity.update(labels.eq(torch.tensor(labels_yy[index]).to(device)).cpu().sum().item(), labels.size(0))
            # keep log of gradients
            for tag, parm in net.named_parameters():
                grads_dict[tag] += parm.grad.data.cpu().numpy() * lr
            del yy

            if verbose == 2:
                template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Accuracy Meta: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_accuracy_meta.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()           

        if SAVE_LOGS == 1:
            np.save(log_dir + "y.npy", new_y)
        # evaluate on validation and test data
        val_accuracy, val_loss, _, _ = evaluate(net, m_dataloader, criterion_cce)
        test_accuracy, test_loss, idx_top5, accs_top5 = evaluate(net, test_dataloader, criterion_cce)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            top5_acc_best = accs_top5.mean()
            top1_acc_best = accs_top5.max()
            epoch_best = epoch

        if SAVE_LOGS == 1:
            summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
            summary_writer.add_scalar('test_loss', test_loss, epoch)
            summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
            summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
            summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
            summary_writer.add_scalar('val_loss', val_loss, epoch)
            summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
            summary_writer.add_scalar('top5_accuracy', accs_top5.mean(), epoch)
            summary_writer.add_scalar('top1_accuracy', accs_top5.max(), epoch)
            summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)
            summary_writer.add_scalar('label_similarity', label_similarity.percentage, epoch)
            summary_writer.add_figure('confusion_matrix', plot_confusion_matrix(net, test_dataloader), epoch)
            if np.count_nonzero(meta_grads_yy_log) > 0:
                summary_writer.add_histogram('meta_labels', labels_yy, epoch)
                summary_writer.add_histogram('grads_yy_meta', get_topk(meta_grads_yy_log), epoch)
                idx_max_grad = np.argsort(meta_grads_yy_log.max(axis=1))[-25:]
                idx_min_grad = np.argsort(meta_grads_yy_log.max(axis=1))[:25]
                summary_writer.add_figure('max_grads', image_grid(idx_max_grad, t_dataset, meta_grads_yy_log), epoch)
                summary_writer.add_figure('min_grads', image_grid(idx_min_grad, t_dataset, meta_grads_yy_log), epoch)
            #for tag, parm in net.named_parameters():
            #    summary_writer.add_histogram('grads_'+tag, grads_dict[tag], epoch)
            if not (noisy_idx is None) and epoch >= stage1 and epoch < stage2:
                num_noisy = np.sum(noisy_idx)
                max_grad_idx = np.argsort(meta_grads_yy_log.max(axis=1))[-num_noisy:]
                num_match = np.sum(noisy_idx[max_grad_idx] == 1)
                similarity = (num_match / num_noisy)*100

                grad_directions = np.argmax(meta_grads_yy_log, axis=1)
                num_true_direction = np.sum(grad_directions == clean_labels)
                true_similarity = (num_true_direction / clean_labels.shape[0])*100

                hard_labels = np.argmax(new_y, axis=1)
                num_true_pred = np.sum(hard_labels == clean_labels)
                pred_similarity = (num_true_pred / clean_labels.shape[0])*100

                #print("Gradients in compliance with synthetic noisy data and true data: {:4.1f}%-{:4.1f}%".format(similarity,true_similarity))
                #print("Correct label percentage: {:4.1f}%".format(pred_similarity))
                summary_writer.add_scalar('compliance_grad_noisydata', similarity, epoch)
                summary_writer.add_scalar('compliance_grad_cleandata', true_similarity, epoch)
                summary_writer.add_scalar('compliance_pred_cleandata', pred_similarity, epoch)

        if verbose > 0:
            template = 'Epoch {}, Accuracy(train,meta_train,val,test): {:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}, Accuracy(top5,top1): {:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f}, Label similarity: {:6.3f}, Learning rate(lr,yy): {}/{}, Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, 
                                train_accuracy.percentage, train_accuracy_meta.percentage, val_accuracy, test_accuracy,
                                accs_top5.mean(), accs_top5.max(),
                                train_loss.avg, val_loss, test_loss,  
                                label_similarity.percentage, lr, int(beta),
                                time.time()-start_epoch, (time.time()-start_epoch)/3600))

    print('{}({}): Train acc: {:3.1f}, Validation acc: {:3.1f}-{:3.1f}, Test acc: {:3.1f}-{:3.1f}, Top5 acc: {:3.1f}-{:3.1f}, Top1 acc: {:3.1f}-{:3.1f}, Best epoch: {}, Num meta-data: {}'.format(
        noise_type, noise_ratio, train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, accs_top5.mean(), top5_acc_best, accs_top5.max(), top1_acc_best, epoch_best, NUM_METADATA))
    if SAVE_LOGS == 1:
        summary_writer.close()
        # write log for hyperparameters
        hp_writer.add_hparams({'alpha':alpha, 'beta': beta, 'gamma':gamma, 'stage1':stage1, 'K':K, 'use_clean':use_clean_data, 'num_meta':NUM_METADATA}, 
                              {'val_accuracy': val_acc_best, 'test_accuracy': test_acc_best, 'top5_acc':top5_acc_best, 'top1_acc':top1_acc_best, 'epoch_best':epoch_best})
        hp_writer.close()
        torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def get_dataloaders_meta():
    NUM_TRAINDATA = len(train_dataset)
    num_meta_data_per_class = int(NUM_METADATA/NUM_CLASSES)
    idx_meta = None
    
    loss_values = np.zeros(NUM_TRAINDATA)
    label_values = np.zeros(NUM_TRAINDATA)
    
    c = nn.CrossEntropyLoss(reduction='none').to(device)
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
        output = net(images)
        loss = c(output, labels)
        loss_values[index] = loss.detach().cpu().numpy()
        label_values[index] = labels.cpu().numpy()
    for i in range(NUM_CLASSES):
        idx_i = label_values == i
        idx_i = np.where(idx_i == True)
        loss_values_i = loss_values[idx_i]
        sorted_idx = np.argsort(loss_values_i)
        anchor_idx_i = np.take(idx_i, sorted_idx[:num_meta_data_per_class])
        if idx_meta is None:
            idx_meta = anchor_idx_i
        else:
            idx_meta = np.concatenate((idx_meta,anchor_idx_i))
    idx_train = np.setdiff1d(np.arange(NUM_TRAINDATA),np.array(idx_meta))

    t_dataset = torch.utils.data.Subset(train_dataset, idx_train)
    m_dataset = torch.utils.data.Subset(train_dataset, idx_meta)
    t_dataloader = torch.utils.data.DataLoader(t_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers)
    m_dataloader = torch.utils.data.DataLoader(m_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers, drop_last=True)
    return t_dataset, m_dataset, t_dataloader, m_dataloader

def get_topk(arr, percent=0.01):
    arr_flat = arr.flatten()
    arr_len = int(len(arr_flat)*percent)
    idx = np.argsort(np.absolute(arr_flat))[-arr_len:]
    return arr_flat[idx]

def image_grid(idx, train_dataset, grads):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(15,15))
    for i in range(25):
        index = idx[i]
        img,l = train_dataset.__getitem__(index)
        # if clean labels are known, print them as well
        if clean_labels is None:
            title = '{}/{}\n{:6.3f}'.format(class_names[l], class_names[np.argmax(grads[index])], grads[index].max())
            color = 'black'
        else:
            title = '{}/{}/{}\n{:6.3f}'.format(class_names[clean_labels[index]], class_names[l], class_names[np.argmax(grads[index])], grads[index].max())
            # if gradient direction is correct
            if clean_labels[index] == np.argmax(grads[index]):
                color = 'g'
            # if gradient direction is wrong
            else:
                color = 'r'
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=title)
        ax = plt.gca()
        ax.set_title(title, color=color)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if dataset == 'mnist_fashion':
            plt.imshow(np.squeeze(img), cmap=plt.cm.binary)
        else:
            img = np.moveaxis(img.numpy(),0,-1)
            plt.imshow(np.clip(img,0,1))
    return figure

def set_learningrate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(net, dataloader, criterion):
    eval_accuracy = AverageMeter()
    eval_loss = AverageMeter()

    topks = {}
    for i in range(NUM_CLASSES):
        topks[i] = AverageMeter()

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            loss = criterion(outputs, targets) 
            _, predicted = torch.max(outputs.data, 1) 
            eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
            eval_loss.update(loss.item())
            for i in range(NUM_CLASSES):
                idx = targets == i
                topks[i].update(predicted[idx].eq(targets[idx].data).cpu().sum().item(), idx.sum().item())  
    # get best 10 accuracies
    accs_per_class = np.array([topks[i].percentage for i in range(NUM_CLASSES)])
    idx = np.argsort(np.absolute(accs_per_class))[-5:]
    return eval_accuracy.percentage, eval_loss.avg, idx, accs_per_class[idx]

def plot_confusion_matrix(net, dataloader):
    net.eval()
    labels, preds = np.zeros(len(dataloader)*BATCH_SIZE), np.zeros(len(dataloader)*BATCH_SIZE)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+targets.size(0))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            _, predicted = torch.max(outputs.data, 1) 

            labels[index] = targets.cpu().numpy()
            preds[index] = predicted.cpu().numpy()

    cm = confusion_matrix(labels, preds)
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=False, type=str, default='cifar10',
        help="Dataset to use; either 'mnist_fashion', 'cifar10', 'cifar100', 'food101N', 'clothing1M'")
    parser.add_argument('-n', '--noise_type', required=False, type=str, default='feature-dependent',
        help="Noise type for cifar10: 'feature-dependent', 'symmetric'")
    parser.add_argument('-r', '--noise_ratio', required=False, type=int, default=40,
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
    parser.add_argument('--save_logs', required=False, type=int, default=1,
        help="Either to save log files (1) or not (0)")
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")
    
    parser.add_argument('-c', '--clean_data', required=False, type=int, default=1,
        help="Either to use available clean data (1) or not (0)")
    parser.add_argument('-m', '--metadata_num', required=False, type=int, default=4000,
        help="Number of samples to be used as meta-data")

    parser.add_argument('-a', '--alpha', required=False, type=float,
        help="Learning rate for meta iteration")
    parser.add_argument('-b', '--beta', required=False, type=float,
        help="Beta paramter")
    parser.add_argument('-g', '--gamma', required=False, type=float,
        help="Gamma paramter")
    parser.add_argument('-s1', '--stage1', required=False, type=int,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int,
        help="Epoch num to end stage2 (meta training)")
    parser.add_argument('-k', required=False, type=int, default=10,
        help="")

    args = parser.parse_args()
    #set default variables if they are not given from the command line
    if args.alpha == None: args.alpha = PARAMS_META[args.dataset]['alpha']
    if args.beta == None: args.beta = PARAMS_META[args.dataset]['beta']
    if args.gamma == None: args.gamma = PARAMS_META[args.dataset]['gamma']
    if args.stage1 == None: args.stage1 = PARAMS_META[args.dataset]['stage1']
    if args.stage2 == None: args.stage2 = PARAMS_META[args.dataset]['stage2']
    if args.k == None: args.k = PARAMS_META[args.dataset]['k']
    # configuration variables
    framework = 'pytorch'
    dataset = args.dataset
    model_name = 'MLNC'
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio/100
    BATCH_SIZE = args.batch_size if args.batch_size != None else PARAMS[dataset]['batch_size']
    NUM_CLASSES = PARAMS[dataset]['num_classes']
    SAVE_LOGS = args.save_logs
    use_clean_data = args.clean_data
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # create necessary folders
    create_folder('{}/dataset'.format(dataset))
    # global variables
    train_dataset, val_dataset, test_dataset, meta_dataset, class_names = get_data(dataset,framework,noise_type,noise_ratio,args.seed,args.metadata_num,0)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    meta_dataloader = torch.utils.data.DataLoader(meta_dataset,batch_size=BATCH_SIZE,shuffle=False, drop_last=True)
    NUM_METADATA = len(meta_dataset)
    noisy_idx, clean_labels = get_synthetic_idx(dataset,args.seed,args.metadata_num,0,noise_type,noise_ratio,)
    net = get_model(dataset,framework).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    lr_scheduler = get_lr_scheduler(dataset)
    optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)
    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    softmax = nn.Softmax(dim=1).to(device)
  
    print("Dataset: {}, Model: {}, Device: {}, Batch size: {}, #GPUS to run: {}".format(dataset, model_name, device, BATCH_SIZE, ngpu))
    if dataset in DATASETS_SMALL:
        print("Noise type: {}, Noise ratio: {}".format(noise_type, noise_ratio))

    # if logging
    if SAVE_LOGS == 1:
        base_folder = model_name if dataset in DATASETS_BIG else noise_type + '/' + str(args.noise_ratio) + '/' + model_name
        log_folder = args.folder_log if args.folder_log else 'c{}_a{}_b{}_g{}_s{}_m{}_{}'.format(use_clean_data, args.alpha, args.beta, args.gamma, args.stage1, NUM_METADATA, current_time)
        log_base = '{}/logs/{}/'.format(dataset, base_folder)
        log_dir = log_base + log_folder + '/'
        log_dir_hp = '{}/logs_hp/{}/'.format(dataset, base_folder)
        create_folder(log_dir)
        summary_writer = SummaryWriter(log_dir)
        create_folder(log_dir_hp)
        hp_writer = SummaryWriter(log_dir_hp)
    
    start_train = time.time()
    metapencil(args.alpha, args.beta, args.gamma, args.stage1, args.stage2, args.k)
    print('Total training duration: {:3.2f}h'.format((time.time()-start_train)/3600))