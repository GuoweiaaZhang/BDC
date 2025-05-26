import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from module import *
from construct_loader import *
torch.cuda.set_device(0)
from load_data import *
from torch import optim
from torch.autograd import Variable
import numpy as np
from torch.cuda.amp import autocast, GradScaler
# =================== parameter set- ===================
epoch = 100
lr = 0.0005
weight_decay = 0.00005
batchsize = 40
n_classes = 10
# ===================================== Data loading ====================================
# PU data format is sample size * 1 * 2049, The 2049 dimension contains 2048 dimensions of sample points and 1 dimension of labels.
PU_0900_1000_07_path = r'E:\Code\Data\PU_0900_1000_07.mat'
x_train_Data_PU_0900_1000_07, x_train_y_Data_PU_0900_1000_07, x_test_Data_PU_0900_1000_07, x_test_y_Data_PU_0900_1000_07 = load_data(PU_0900_1000_07_path,
                                                                                             'PU_0900_1000_07', 10, 100,
                                                                                             [])  # Data segmentation and data fft transformation

PU_1500_0400_07_path = r'E:\Code\Data\PU_1500_0400_07.mat'
x_train_Data_PU_1500_0400_07, x_train_y_Data_PU_1500_0400_07, x_test_Data_PU_1500_0400_07, x_test_y_Data_PU_1500_0400_07 = load_data(PU_1500_0400_07_path,
                                                                                             'PU_1500_0400_07', 10, 100, [])

PU_1500_1000_01_path = r'E:\Code\DataPU_1500_1000_01.mat'
x_train_Data_PU_1500_1000_01, x_train_y_Data_PU_1500_1000_01, x_test_Data_PU_1500_1000_01, x_test_y_Data_PU_1500_1000_01 = load_data(PU_1500_1000_01_path,
                                                                                             'PU_1500_1000_01', 10, 100, [])

# =======Single Domain Generalized Task Setting. Setting up different generalization tasks by adjusting the index======================
Task_data = [x_train_Data_PU_0900_1000_07.cuda(), x_train_y_Data_PU_0900_1000_07.view(-1),
             x_train_Data_PU_1500_0400_07.cuda(), x_train_y_Data_PU_1500_0400_07.view(-1),
             x_train_Data_PU_1500_1000_01.cuda(), x_train_y_Data_PU_1500_1000_01.view(-1)]

Train_X1 = Task_data[0]  # Source Domain Data
Train_Y1 = Task_data[1]  # Source Domain labels
Test_X = Task_data[2]  # Target Domain Data
Test_Y = Task_data[3]  # Target Domain labels

# Construct training data loaders
Train_loader = construct_loader([Train_X1, Train_Y1], batch_size=batchsize)
TR_dataloader = Train_loader
# ==================Initialize models =====================
if torch.cuda.is_available():
    # Instantiate models
    CNN_st = CNN().cuda()
    CNN_te = CNN_Tea().cuda()
    classifier = Classifier(n_classes).cuda()
    classifier_t = Classifier_te(n_classes).cuda()
    # Define loss functions
    criterion = nn.CrossEntropyLoss()
    amp_grad_scaler = GradScaler()
    E_dis = nn.MSELoss()
    Con = SupConLoss()
    start_time = time.time()
    # Initialize data placeholders
    Acc = []
    s1_y = []
    s1_x = []
    Recall = []
    F1 = []
    loss_1_list = list()
    loss_2_list = list()
    loss_1 = None
    loss_2 = None
    # Define optimizers for each model component
    optimizer_ST = optim.Adam(CNN_st.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_TE = optim.Adam(CNN_te.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_LD = optim.Adam(CNN_te.lsp_1.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier_t = optim.Adam(classifier_t.parameters(), lr=lr, weight_decay=weight_decay)
    # ======================== Model Training ============================
    for i in range(epoch):
        # Setting the model to training mode
        CNN_st.train()
        CNN_te.train()
        CNN_te.lsp_1.train()
        classifier.train()
        classifier_t.train()
        # Get Data Loader
        iter_source1 = iter(TR_dataloader)
        # Small batch data loops
        for step in range(len(TR_dataloader)):
            s1_data, s1_label = iter_source1.__next__()
            if torch.cuda.is_available():
                s1_x = Variable(s1_data).type(dtype=torch.float32).cuda()
                s1_y = Variable(s1_label).type(dtype=torch.LongTensor).cuda()
            # -----Optimizer gradient clearing-------
            optimizer_ST.zero_grad()
            optimizer_TE.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier_t.zero_grad()
            # -----Training consistency-guided covariance and semantic alignment modules-------
            with autocast():
                features_st, st = CNN_st(s1_x)
                pre_st = classifier(features_st)
                loss_cls_st = criterion(pre_st, s1_y.long())
                # CNN_te as a network with learnable perturbations
                features_te, te = CNN_te(s1_x, perturb=True)
                pre_te = classifier_t(features_te)
                loss_cls_te = criterion(pre_te, s1_y.long())
                # Covariance alignment loss
                st2_cm = get_covariance_matrix(st)[0]
                te2_cm = get_covariance_matrix(te)[0]
                consis_cml = F.l1_loss(st2_cm, te2_cm, reduction='mean')
                consis_ccl = cross_whitening_loss(st, te)
                # Semantic alignment loss
                emb_src = F.normalize(pre_st).unsqueeze(1)
                emb_aug = F.normalize(pre_te).unsqueeze(1)
                con = Con(torch.cat([emb_src, emb_aug], dim=1), s1_y.long())
                #  loss 1
                Loss_1 = loss_cls_st + loss_cls_te + 10 * consis_cml + 1 * consis_ccl + 0.1 * con

            amp_grad_scaler.scale(Loss_1).backward()
            amp_grad_scaler.unscale_(optimizer_ST)
            amp_grad_scaler.step(optimizer_ST)
            amp_grad_scaler.unscale_(optimizer_classifier)
            amp_grad_scaler.step(optimizer_classifier)
            amp_grad_scaler.unscale_(optimizer_classifier_t)
            amp_grad_scaler.step(optimizer_classifier_t)
            amp_grad_scaler.unscale_(optimizer_TE)
            amp_grad_scaler.step(optimizer_TE)
            amp_grad_scaler.update()
            # -----Training discrepancy-guided learnable feature statistics modules-------
            optimizer_LD.zero_grad()
            with autocast():

                features_st, _ = CNN_st(s1_x)
                features_te, _ = CNN_te(s1_x, perturb=True)
                l0, l1, l2 = CNN_te.l0, CNN_te.l1, CNN_te.l2
                l0_st, l1_st, l2_st = CNN_st.l0, CNN_st.l1, CNN_st.l2
                #  loss coral
                coral_0 = CORAL(l0_st.view(40, -1), l0.view(40, -1))
                coral_1 = CORAL(l1_st.view(40, -1), l1.view(40, -1))
                coral_2 = CORAL(l2_st.view(40, -1), l2.view(40, -1))
                loss_2 = - 0.1 * (coral_0)

            amp_grad_scaler.scale(loss_2).backward()
            amp_grad_scaler.unscale_(optimizer_LD)
            amp_grad_scaler.step(optimizer_LD)
            amp_grad_scaler.update()

        if loss_1 is not None:
            loss_1_list.append(loss_1.detach().cpu().numpy())
            mean_loss_1 = np.mean(loss_1_list)
        else:
            mean_loss_1 = 0

        if loss_2 is not None:
            loss_2_list.append(loss_2.detach().cpu().numpy())
            mean_loss_2 = np.mean(loss_2_list)
        else:
            mean_loss_2 = 0

        end_time = time.time()
        time_len = end_time - start_time

# ========================= Model Testing ========================
        with torch.no_grad():
            CNN_te.eval()
            CNN_st.eval()
            classifier.eval()
            classifier_t.eval()

            features = CNN_te(Train_X1.cuda(), perturb=False)[0]
            scores_train = classifier_t(features)
            pre_label = torch.max(torch.softmax(scores_train, 1), 1)[1].cpu().numpy()
            train_acc = accuracy_score(pre_label, Train_Y1.cpu().numpy())

            features_test = CNN_te(Test_X.cuda(), perturb=False)[0]
            scores_test = classifier_t(features_test)
            test_label = torch.max(torch.softmax(scores_test, 1), 1)[1].cpu().numpy()
            test_acc = accuracy_score(test_label, Test_Y.cpu().numpy())

            Acc.append(test_acc)
            recall_ave_class = recall_score(test_label, Test_Y.cpu().numpy(), average='macro')
            Recall.append(recall_ave_class)
            f1 = f1_score(test_label, Test_Y.cpu().numpy(), average='macro')
            F1.append(f1)

        if i == 0 or (i + 1) % 1 == 0:
            print(
                '==> Epoch: {}/{}, Loss_1: {:.5f}, Loss_2: {:.5f}, Test_Acc: {:.5f}, Recall: {:.4f}, F1: {:.4f}, Running time is {:.5f} s'.format(
                    i + 1, epoch, mean_loss_1, mean_loss_2, test_acc, recall_ave_class, f1, time_len))
