def get_framework(framework=None):
    # get the installed framework. either tensorflow or pytorch
    if framework == None:
        try:
            import tensorflow as tf
            framework = 'tensorflow'
        except ImportError:
            try:
                import torch
                framework = 'pytorch'
            except ImportError:
                raise ImportError('Either tensorflow or pytorch should be installed!')
    assert framework in ['tensorflow', 'pytorch'], 'Framework should be either tensorflow or pytorch!'
    return framework

def get_model(dataset, framework='pytorch'):
    if dataset == 'mnist_fashion':
        return model_mnistfashion(framework)
    elif dataset == 'cifar10':
        return model_cifar10(framework)
    elif dataset == 'cifar100':
        return resnet34(framework, num_classes = 100, input_shape=(32,32,3), dataset=dataset)
    elif dataset == 'food101N':
        return resnet50(framework, num_classes = 101, input_shape=(224,224,3), dataset=dataset)
    elif dataset == 'clothing1M' or dataset == 'clothing1M50k' or dataset == 'clothing1Mbalanced':
        return resnet50(framework, num_classes = 14, input_shape=(224,224,3), dataset=dataset)

def model_mnistfashion(framework):
    if framework == 'pytorch':
        import torch.nn as nn
        import torch.nn.functional as F

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.dropout1 = nn.Dropout(p=0.25)
                self.fc1 = nn.Linear(64 * 12 * 12, 128)
                self.dropout2 = nn.Dropout(p=0.5)
                self.fc2 = nn.Linear(128, 10)
                
            def forward(self, x, weights=None, get_feat=False):
                if weights==None:
                    x = F.relu(self.conv1(x))
                    x = self.pool(F.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 12 * 12)
                    x = self.dropout1(x)
                    x = F.relu(self.fc1(x))
                    feat = self.dropout2(x)
                    x = self.fc2(feat)
                    if get_feat:
                        return x,feat
                    else:
                        return x
                else:
                    x = F.conv2d(x, weights['conv1.weight'], weights['conv1.bias'], padding=0, stride=1)
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.conv2d(x, weights['conv2.weight'], weights['conv2.bias'], padding=0, stride=1)
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                    x = x.view(-1, 64 * 12 * 12)
                    x = F.dropout(x, p=0.25)
                    x = F.linear(x, weights['fc1.weight'], weights['fc1.bias'])  
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.dropout(x, p=0.5)
                    x = F.linear(x, weights['fc2.weight'], weights['fc2.bias'])  
                return x
        return Net()
    elif framework == 'tensorflow':
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout

        # architecture from: https://keras.io/examples/mnist_cnn/
        img_input = Input(shape=(28,28,1))  
        x = Conv2D(32, kernel_size=(3, 3))(img_input)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(10, name='features')(x)
        #x = Activation('softmax')(x)
        # Create model
        return Model(img_input, x) 

def resnet34(framework, num_classes, input_shape, dataset):
    import os
    if framework == 'pytorch':
        import torch
        from torch import nn
        from resnet_torch import resnet34
        try:
            net = resnet34(pretrained=True)
        except:
            net = resnet34(pretrained=False)
            #net.load_state_dict(torch.load('resnet50.pt'))
        net.fc = nn.Linear(2048,num_classes)
        return net
    elif framework == 'tensorflow':
        return None

def resnet50(framework, num_classes, input_shape, dataset):
    import os
    if framework == 'pytorch':
        import torch
        from torch import nn
        from resnet_torch import resnet50
        try:
            net = resnet50(pretrained=True)
        except:
            net = resnet50(pretrained=False)
            net.load_state_dict(torch.load('resnet50.pt'))
        net.fc = nn.Linear(2048,num_classes)
        return net
    elif framework == 'tensorflow':
        import tensorflow as tf
        #from resnet50 import ResNet50
        model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
        output = tf.keras.layers.Dense(num_classes)(model.layers[-1].output)
        return tf.keras.models.Model(model.input, outputs=output)

def model_cifar10(framework):
    num_classes = 10
    input_shape=(32,32,3)

    if framework == 'pytorch':
        import torch.nn as nn
        import torch.nn.functional as F

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # block1
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                # block2
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
                self.bn4 = nn.BatchNorm2d(128)
                # block3
                self.conv5 = nn.Conv2d(128, 196, 3, padding=1)
                self.bn5 = nn.BatchNorm2d(196)
                self.conv6 = nn.Conv2d(196, 196, 3, padding=1)
                self.bn6 = nn.BatchNorm2d(196)

                self.pool = nn.MaxPool2d(2, 2)

                self.fc1 = nn.Linear(196 * 4 * 4, 256)
                self.bn7 = nn.BatchNorm1d(256)
                self.fc2 = nn.Linear(256, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                
            def forward(self, x, weights=None, get_feat=False):
                if weights==None:
                    # block1
                    x = F.relu(self.bn1(self.conv1(x)))
                    x = F.relu(self.bn2(self.conv2(x)))
                    x = self.pool(x)
                    # block2
                    x = F.relu(self.bn3(self.conv3(x)))
                    x = F.relu(self.bn4(self.conv4(x)))
                    x = self.pool(x)
                    # block3
                    x = F.relu(self.bn5(self.conv5(x)))
                    x = F.relu(self.bn6(self.conv6(x)))
                    x = self.pool(x)

                    x = x.view(-1, 196 * 4 * 4)
                    feat = F.relu(self.bn7(self.fc1(x)))
                    x = self.fc2(feat)
                    if get_feat:
                        return x,feat
                    else:
                        return x
                else:
                    list_weights = list(weights.items())
                    count = 0
                    #block1
                    x = F.conv2d(x, list_weights[count][1], list_weights[count+1][1], padding=1, stride=1)
                    count += 2
                    x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.conv2d(x, list_weights[count][1], list_weights[count+1][1], padding=1, stride=1)
                    count += 2
                    x = F.batch_norm(x, self.bn2.running_mean, self.bn2.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                    #block2
                    x = F.conv2d(x, list_weights[count][1], list_weights[count+1][1], padding=1, stride=1)
                    count += 2
                    x = F.batch_norm(x, self.bn3.running_mean, self.bn3.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.conv2d(x, list_weights[count][1], list_weights[count+1][1], padding=1, stride=1)
                    count += 2
                    x = F.batch_norm(x, self.bn4.running_mean, self.bn4.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                    #block3
                    x = F.conv2d(x, list_weights[count][1], list_weights[count+1][1], padding=1, stride=1)
                    count += 2
                    x = F.batch_norm(x, self.bn5.running_mean, self.bn5.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.conv2d(x, list_weights[count][1], list_weights[count+1][1], padding=1, stride=1)
                    count += 2
                    x = F.batch_norm(x, self.bn6.running_mean, self.bn6.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    x = F.threshold(x, 0, 0, inplace=True)
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                    
                    x = x.view(-1, 196 * 4 * 4)
                    x = F.linear(x, list_weights[count][1], list_weights[count+1][1])
                    count += 2
                    x = F.batch_norm(x, self.bn7.running_mean, self.bn7.running_var, list_weights[count][1], list_weights[count+1][1],training=True) 
                    count += 2
                    feat = F.threshold(x, 0, 0, inplace=True)
                    x = F.linear(feat, list_weights[count][1], list_weights[count+1][1]) 
                    return x  

        return Net()
    elif framework == 'tensorflow':
        from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        img_input = Input(shape=input_shape)
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)

        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='features')(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        #x = Activation(tf.nn.softmax)(x)

        # Create model.
        return Model(img_input, x)

if __name__ == '__main__':
    get_model('clothing1M')