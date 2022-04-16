import time

from src.config import *
from src.model.multi_scale_ori import MSResNet


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2))

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

        self.net = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
        )

        self.fc = nn.Linear(128, opt.n_classes)
        self.activation = nn.Sigmoid()

        self.init_bias()

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        out = self.net(x)
        out = out.view(x.shape[0], out.size(1) * out.size(2))

        logit = self.fc(out)
        return logit


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation acc ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc


class Classifier:
    def __init__(self):
        self.net = CNN1d()
        # self.net = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=opt.n_classes)
        if opt.ngpu > 1:
            self.net = torch.nn.DataParallel(self.net).cuda()
        else:
            self.net = self.net.cuda()
        self.net = self.net.cuda()

    def fit(self, train_set, train_labels, test_set, test_labels):

        train_set = torch.tensor(train_set).type(torch.FloatTensor).cuda()
        val_set = torch.tensor(test_set).type(torch.FloatTensor).cuda()

        train_labels = torch.Tensor(train_labels).type(torch.LongTensor).cuda()
        val_labels = torch.Tensor(test_labels).type(torch.LongTensor).cuda()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.9, 0.999))

        # optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-08)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3, mode='min', min_lr=1e-05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)

        loss_func = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(train_set, train_labels)
        val_dataset = TensorDataset(val_set, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=-1, verbose=True)


        with torch.autograd.set_detect_anomaly(True):
            for t in range(opt.dl_n_epochs):
                train_avg_loss = 0
                val_avg_loss = 0
                train_preds = np.zeros((len(train_dataset), 1))

                for idx, data in enumerate(train_loader):
                    train_x_batch = data[:-1][0]
                    train_y_batch = data[-1]
                    train_y_pred = self.net(train_x_batch.unsqueeze(1))
                    optimizer.zero_grad()
                    loss = loss_func(train_y_pred, train_y_batch)
                    loss.backward()
                    optimizer.step()
                    train_avg_loss += loss.item() / len(train_loader)
                    # print('train_y_pred.data: ', train_y_pred.data)

                    _, predicted = torch.max(train_y_pred.data, 1)
                    predicted = predicted.detach().cpu().numpy()
                    predicted = predicted[..., np.newaxis]
                    train_preds[idx * opt.batch_size:(idx + 1) * opt.batch_size, :] = predicted

                if (t+1) % 1 == 0:
                    val_preds = np.zeros((len(val_dataset), 1))
                    with torch.no_grad():
                        self.net.eval()
                        for idx, data in enumerate(val_loader):
                            val_x_batch = data[:-1][0]
                            val_y_batch = data[-1]
                            start_time = time.time()
                            val_y_pred = self.net(val_x_batch.unsqueeze(1))
                            end_time = time.time()
                            # print(f'\n ======= time cost to predict 1 spectra: {(end_time - start_time) / len(val_y_pred) * 500} ======== \n')
                            val_loss = loss_func(val_y_pred, val_y_batch)
                            val_avg_loss += val_loss.item() / len(val_loader)
                            # val_y_pred = torch.mean(val_y_pred, dim=1).unsqueeze(1).detach().cpu().numpy()
                            _, predicted = torch.max(val_y_pred.data, 1)
                            predicted = predicted.detach().cpu().numpy()
                            predicted = predicted[..., np.newaxis]
                            val_preds[idx * opt.batch_size:(idx + 1) * opt.batch_size, :] = predicted
                        # print('*** checking the length of val_preds: ', len(val_preds))

                        val_acc = accuracy_score(y_pred=val_preds, y_true=val_labels.data.cpu().numpy())

                        print(f'\nEpoch={t+1}, Val Acc={val_acc}\n')

                        self.net.train()

                        # scheduler.step()

                        # for param_group in optimizer.param_groups:
                        #     old_lr = param_group['lr']
                        #     print(param_group['lr'])
                        #
                        # scheduler.step(val_acc)
                        # for param_group in optimizer.param_groups:
                        #     new_lr = param_group['lr']
                        #     if new_lr < old_lr:
                        #         print("change learning rate at epoch {} from {} to {}".format(t, old_lr, new_lr))

                        # early_stopping(val_acc, self.net)
                        #
                        # if early_stopping.early_stop:
                        #     print("Early stopping")
                        #     break
        return loss

    def predict(self, test_set):
        test_set = torch.tensor(test_set).type(torch.FloatTensor).cuda()

        test_dataset = TensorDataset(test_set)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
        test_preds = np.zeros((len(test_dataset), 1))
        with torch.no_grad():
            self.net.eval()
            for idx, data in enumerate(test_loader):
                test_x_batch = data[0]
                test_y_pred = self.net(test_x_batch.unsqueeze(1))
                _, predicted = torch.max(test_y_pred.data, 1)
                # print('test_y_pred.data: ', test_y_pred.data)

                predicted = predicted.detach().cpu().numpy()
                predicted = predicted[..., np.newaxis]
                test_preds[idx * opt.batch_size:(idx + 1) * opt.batch_size, :] = predicted

        return test_preds



