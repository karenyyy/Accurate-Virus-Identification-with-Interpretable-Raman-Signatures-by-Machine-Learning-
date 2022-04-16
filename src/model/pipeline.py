import time

from imblearn.under_sampling import RandomUnderSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.model.fullgrad import *
from src.model.nn import *
from src.utils.util import *


class VirusClassifier():

    def __init__(self):
        self.csv_path = FEATURESET_FILE

        self.cnn_testset_predicted = []
        self.rf_testset_predicted = []
        self.svm_testset_predicted = []
        self.mlp_testset_predicted = []
        self.gnb_testset_predicted = []

    def get_features(self, csv_path):
        data_file = pd.read_csv(csv_path, index_col=False)
        self.data_file = data_file

        train_val_set = data_file

        test_set = data_file.loc[data_file['label'].isin([opt.blind_sample])]
        train_val_set = train_val_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        self.test_set = test_set

        self.total_labels = np.array(data_file['label'])
        self.labels = np.array([i for i in train_val_set['label']])

        print('test set sample: ', [CLASSES[i] for i in np.unique(test_set['label'])])

        train_val_features = train_val_set.drop('label', axis=1) \
            .drop('pos1', axis=1)\
            .drop('pos2', axis=1)
        test_features = test_set.drop('label', axis=1) \
            .drop('pos1', axis=1) \
            .drop('pos2', axis=1)
        data_file2 = data_file.drop('label', axis=1) \
            .drop('pos1', axis=1) \
            .drop('pos2', axis=1)

        self.feature_list = [int(i) for i in list(train_val_features.columns)]

        print('how many train_val_features?', len(self.feature_list))

        if opt.back_sub:
            for idx, row in train_val_features.iterrows():
                after_backsub, background = bfg_sub(list(row.values))
                train_val_features.iloc[idx, :] = after_backsub

        self.total_features = np.array(data_file2)
        self.train_val_features = normalize(np.array(train_val_features), copy=False, norm='l2')
        self.test_features = normalize(np.array(test_features), copy=False, norm='l2')

        print('feature set shape: ', self.train_val_features.shape)

        # self.plot_tsne(np.array(self.total_features), self.total_labels, [], data_file)

    def plot_tsne(self, features, label1, label2, df):
        plt.figure(figsize=(10, 10))
        palette = sns.color_palette("bright", len(np.unique(label1)))
        # label1 = [CLASSES[int(l.split('_')[0])] + l.split('_')[1] for l in label1]
        # label1 = [CLASSES[l] for l in label1]

        tsne = TSNE(n_components=2, random_state=100, perplexity=50)
        X_embedded = tsne.fit_transform(normalize(features, copy=False, norm='l2'))

        # pca = PCA(n_components=2)
        # X_embedded = pca.fit_transform(normalize(features, copy=False, norm='l2'))

        # ax = sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1],
        #                 hue=label1, style=label2, legend='full', palette=palette, s=100)
        ax = sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1],
                             hue=label1, legend='full', palette=palette, s=100)

        plt.setp(ax.get_legend().get_texts(), fontsize='15')

        plt.axis('off')
        plt.title('{} hours; Strains: ({})'.format(
            opt.virus_type,
            ', '.join(CLASSES)
        ), fontsize=16)

        # add index as annotation
        for i in range(len(df)):
            plt.text(X_embedded[i, 0] - 0.1,
                    X_embedded[i, 1] - 0.2,
                    str(df.index[i]))

        plt.show()

        df = pd.DataFrame(
            {
                'x': X_embedded[:, 0],
                'y': X_embedded[:, 1],
                'label': label1
            }
        )
        df.to_csv(os.path.join(RESULT_PATH,  f"tsne_{FEATURESET_FILE.replace('.csv', '')}.csv"), index=False)

    def predict_(self, validate_set, predicted, validate_labels):
        validate_set = torch.Tensor(validate_set).cuda()

        grad_cams = [FullGrad(model=self.cnn1d.net, im_size=(1, len(self.feature_list)))]

        masks = defaultdict(list)
        samples = defaultdict(list)
        count = defaultdict(int)
        for idx in range(validate_set.size(0)):
            sample = validate_set[idx, :]
            sample = sample.unsqueeze(0).unsqueeze(1)

            label = validate_labels[idx].item()

            grad_masks = [grad_cam.saliency(sample, label)[0] for grad_cam in grad_cams]
            grad_masks[0] = grad_masks[0][~np.isnan(grad_masks[0])]

            if len(grad_masks[0]) > 1:
                if label not in masks.keys():
                    masks[label] = grad_masks
                    count[label] = 1
                else:
                    masks[label] = np.array(
                        [np.array(list(map(add, masks[label][i], grad_masks[i]))) for i in range(len(grad_masks))])

                    count[label] += 1

            samples[label] = validate_set[np.random.choice(validate_set.size(0), 1)[0], :].data.cpu().numpy()

        return predicted, masks, samples

    def eval(self, train_set, train_labels, validate_set, validate_labels):
        self.cnn1d = Classifier()
        print('train and validate labels: ', Counter(train_labels), Counter(validate_labels))
        self.cnn1d.fit(train_set, train_labels, validate_set, validate_labels)
        start_time = time.time()
        cnn_predicted_val = self.cnn1d.predict(validate_set)
        end_time = time.time()
        print(f'\n ======= time costed to predict {len(cnn_predicted_val)} spectra: {(end_time - start_time) / len(cnn_predicted_val)} ======== \n')

        cnn_testset_predicted = self.cnn1d.predict(self.test_features)

        self.cnn_testset_predicted = [i[0] for i in cnn_testset_predicted]
        cnn_predicted, masks, samples = self.predict_(validate_set, cnn_predicted_val, validate_labels)
        print('\t\n\n ************************************* ')
        cnn_validate_acc = accuracy_score(y_true=validate_labels, y_pred=cnn_predicted)
        # print(validate_labels, cnn_predicted)
        print("Classification report for classifier %s:\n%s\n"
              % (self.cnn1d, metrics.classification_report(validate_labels, cnn_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(validate_labels, cnn_predicted))


        self.xgb = XGBClassifier(silent=False,
                                 scale_pos_weight=1,
                                 learning_rate=0.03,
                                 colsample_bytree=1,
                                 subsample=0.8,
                                 # objective='multi:softprob',
                                 objective='binary:logistic',
                                 n_estimators=200,
                                 reg_alpha=0.3,
                                 max_depth=4,
                                 gamma=0,
                                 random_state=123)
        XGBClassifier()
        self.xgb.fit(train_set, train_labels)
        xgb_predicted_val = self.xgb.predict(validate_set)
        xgb_testset_predicted = self.xgb.predict(self.test_features)
        self.xgb_testset_predicted = xgb_testset_predicted
        xgb_validate_acc = accuracy_score(y_true=validate_labels, y_pred=xgb_predicted_val)
        print("Classification report for classifier %s:\n%s\n"
              % (self.xgb, metrics.classification_report(validate_labels, xgb_predicted_val)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(validate_labels, xgb_predicted_val))


        # self.rf = RandomForestClassifier()
        # self.rf.fit(train_set, train_labels)
        # rf_predicted = self.rf.predict(validate_set)
        # rf_testset_predicted = self.rf.predict(self.test_features)
        # self.rf_testset_predicted = rf_testset_predicted
        # print("Classification report for classifier %s:\n%s\n"
        #       % (self.rf, metrics.classification_report(validate_labels, rf_predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(validate_labels, rf_predicted))


        # self.svm = SVC(gamma=2, C=1)
        # self.svm.fit(train_set, train_labels)
        # svm_predicted = self.svm.predict(validate_set)
        # svm_testset_predicted = self.svm.predict(self.test_features)
        # self.svm_testset_predicted = svm_testset_predicted
        # print("Classification report for classifier %s:\n%s\n"
        #       % (self.svm, metrics.classification_report(validate_labels, svm_predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(validate_labels, svm_predicted))


        # self.mlp = MLPClassifier(alpha=1, max_iter=1000)
        # self.mlp.fit(train_set, train_labels)
        # mlp_predicted = self.mlp.predict(validate_set)
        # mlp_testset_predicted = self.mlp.predict(self.test_features)
        # self.mlp_testset_predicted = mlp_testset_predicted
        # print("Classification report for classifier %s:\n%s\n"
        #       % (self.mlp, metrics.classification_report(validate_labels, mlp_predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(validate_labels, mlp_predicted))
        #
        #
        # self.gnb = GaussianNB()
        # self.gnb.fit(train_set, train_labels)
        # gnb_predicted = self.gnb.predict(validate_set)
        # gnb_testset_predicted = self.gnb.predict(self.test_features)
        # self.gnb_testset_predicted = gnb_testset_predicted
        # print("Classification report for classifier %s:\n%s\n"
        #       % (self.gnb, metrics.classification_report(validate_labels, gnb_predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(validate_labels, gnb_predicted))


        return cnn_validate_acc, xgb_validate_acc, masks, samples, cnn_predicted_val, xgb_predicted_val

    def get_feature_importance(self):

        importances = self.xgb.feature_importances_

        feature_importances = [[int(feature), importance] for feature, importance in
                               zip(self.feature_list, importances)]

        feature_importances = sorted(feature_importances, key=lambda x: x[0], reverse=False)
        feature_importances = [pair[1] for pair in feature_importances]

        return feature_importances

    def data_augmentation(self, train_set, label_to_augment):

        train_set['label'] = [Subtype2Idx[i] for i in list(train_set['label'])]
        train_labels = train_set['label']
        augmented_train_set = train_set.copy()
        augmented_train_labels = train_labels.copy()
        original = len(augmented_train_labels) * ['Original Data']
        augmented = []
        train_set_to_aug0 = train_set[train_labels == label_to_augment[0]].copy()
        train_labels_to_aug0 = train_labels[train_labels == label_to_augment[0]].copy()
        train_set_to_aug1 = train_set[train_labels == label_to_augment[1]].copy()
        train_labels_to_aug1 = train_labels[train_labels == label_to_augment[1]].copy()

        train_set = train_set.drop('label', axis=1)
        augmented_train_set = augmented_train_set.drop('label', axis=1)
        train_set_to_aug0 = train_set_to_aug0.drop('label', axis=1)
        train_set_to_aug1 = train_set_to_aug1.drop('label', axis=1)

        # train_set = normalize(np.array(train_set), copy=False, norm='l2')
        train_set = np.array(train_set)
        augmented_train_set = normalize(np.array(augmented_train_set), copy=False, norm='l2')
        train_set_to_aug0 = normalize(np.array(train_set_to_aug0), copy=False, norm='l2')
        train_set_to_aug1 = normalize(np.array(train_set_to_aug1), copy=False, norm='l2')


        if opt.aug == 'ros':
            for _ in range(5):
                ros = RandomOverSampler(sampling_strategy='not majority')
                ros_train_set, ros_train_labels = ros.fit_resample(train_set_to_aug, train_labels_to_aug)
                augmented.extend(len(ros_train_labels) * ['Augmented Data'])

                augmented_train_set = np.concatenate((augmented_train_set, ros_train_set), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, ros_train_labels), axis=0)

        elif opt.aug == 'smote':
            for _ in range(5):
                smote_train_set, smote_train_labels = SMOTE(sampling_strategy=1).fit_resample(train_set_to_aug,
                                                                                              train_labels_to_aug)
                augmented.extend(len(smote_train_labels) * ['Augmented Data'])

                augmented_train_set = np.concatenate((augmented_train_set, smote_train_set), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, smote_train_labels), axis=0)

        elif opt.aug == 'add_noise':

            for i in np.arange(0.01, 0.05, 0.01):
                X_addnoise = tsaug.AddNoise(scale=i).augment(train_set_to_aug)
                X_addnoise = np.array(X_addnoise)
                augmented.extend(len(X_addnoise) * ['Augmented Data'])
                augmented_train_set = np.concatenate((augmented_train_set, X_addnoise), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, train_labels_to_aug), axis=0)

            print('** after adding noise: ', augmented_train_set.shape)

        elif opt.aug == 'drift':
            for i in np.arange(0.1, 0.14, 0.01):
                X_drifted = tsaug.Drift(max_drift=i, n_drift_points=500).augment(train_set_to_aug)
                X_drifted = np.array(X_drifted)
                augmented.extend(len(X_drifted) * ['Augmented Data'])

                augmented_train_set = np.concatenate((augmented_train_set, X_drifted), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, train_labels_to_aug), axis=0)

            print('** after drifting: ', augmented_train_set.shape)

        elif opt.aug == 'quan':
            for i in range(10, 20, 2):
                X_quan = tsaug.Quantize(n_levels=i).augment(train_set_to_aug)
                X_quan = np.array(X_quan)
                augmented.extend(len(X_quan) * ['Augmented Data'])

                augmented_train_set = np.concatenate((augmented_train_set, X_quan), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, train_labels_to_aug), axis=0)

            print('** after quantizing: ', augmented_train_set.shape)

        elif opt.aug == 'dropout':
            for i in np.arange(0.2, 1.2, 0.2):
                X_dropout = tsaug.Dropout(p=i, size=(1, 5), fill='ffill', per_channel=True).augment(
                    train_set_to_aug)
                X_dropout = np.array(X_dropout)
                augmented.extend(len(X_dropout) * ['Augmented Data'])

                augmented_train_set = np.concatenate((augmented_train_set, X_dropout), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, train_labels_to_aug), axis=0)

            print('** after dropout: ', augmented_train_set.shape)

        elif opt.aug == 'pool':
            for i in range(4, 6):
                X_pooled = tsaug.Pool(size=i).augment(train_set_to_aug1)
                X_pooled = np.array(X_pooled)
                print('len(X_pooled): ', len(X_pooled))
                augmented.extend(len(X_pooled) * ['Augmented Data'])
                augmented_train_set = np.concatenate((augmented_train_set, X_pooled), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, train_labels_to_aug1), axis=0)

            for i in range(4, 8):
                X_pooled = tsaug.Pool(size=i).augment(train_set_to_aug0)
                X_pooled = np.array(X_pooled)
                print('len(X_pooled): ', len(X_pooled))
                augmented.extend(len(X_pooled) * ['Augmented Data'])
                augmented_train_set = np.concatenate((augmented_train_set, X_pooled), axis=0)
                augmented_train_labels = np.concatenate((augmented_train_labels, train_labels_to_aug0), axis=0)

            print('** after pooling: ', augmented_train_set.shape)

        else:
            ros = RandomOverSampler(sampling_strategy='not majority')
            augmented_train_set, augmented_train_labels = ros.fit_resample(train_set, train_labels)
            # pass

        train_set = augmented_train_set
        train_labels = augmented_train_labels
        return train_set, train_labels, original, augmented

    def run(self, round, d):
        # default set as cross val
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        fold = 1

        cnn_avg_validate_acc = []
        xgb_avg_validate_acc = []

        test_set_prediction = {}

        kfolds_masks = defaultdict(list)
        kfold_samples = defaultdict(list)
        kfolds_feature_importances = []

        validate_set_idxes = []
        validate_set_preds = []
        validate_set_gt = []
        validate_set_gt_strains = []

        for train_set_idx, validate_set_idx in skf.split(self.train_val_features, y=self.labels):
            print('fold: ', fold)
            train_set, validate_set, train_labels, validate_labels = self.train_val_features[train_set_idx], \
                                                             self.train_val_features[validate_set_idx], \
                                                             self.labels[train_set_idx], \
                                                             self.labels[validate_set_idx]

            validate_strain_labels = self.labels[validate_set_idx]

            print('train_labels, validate_labels: ', len(train_labels), len(validate_labels))


            print('\n---------------------- Fold {} ---------------------\n'.format(fold))

            # print('\n##################### Before Data Augmentation ###################\n')
            # print(Counter(train_labels))
            #
            # train_set, train_labels, original, augmented = self.data_augmentation(train_set, train_labels, label_to_augment=0)
            #
            # print('\n##################### After Data Augmentation ###################\n')
            # print(Counter(train_labels))


            # self.plot_tsne(np.concatenate((train_set, validate_set), axis=0),
            #                np.concatenate((train_labels, validate_labels), axis=0),
            #                np.array(original+augmented+['Original Data']*len(validate_labels)),
            #                self.data_file)

            cnn_validate_acc, xgb_validate_acc, masks, samples, cnn_predicted_val, xgb_predicted_val = self.eval(train_set, train_labels, validate_set, validate_labels)

            print('cnn_test pred: ', self.cnn_testset_predicted)
            print('xgb_test_pred: ', self.xgb_testset_predicted)


            test_set_prediction[f'cnn_round_{round}_fold_{fold}'] = self.cnn_testset_predicted
            test_set_prediction[f'xgb_round_{round}_fold_{fold}'] = list(self.xgb_testset_predicted)

            print('validate_strain_labels: ', validate_strain_labels)
            validate_set_idxes.extend(validate_set_idx)
            validate_set_preds.extend([i[0] for i in cnn_predicted_val])
            validate_set_gt.extend(validate_labels)
            validate_set_gt_strains.extend(validate_strain_labels)

            feature_importances = self.get_feature_importance()

            kfolds_masks = update_dict(masks, kfolds_masks)
            kfold_samples = update_dict(samples, kfold_samples)
            kfold_samples = samples

            cnn_avg_validate_acc.append(cnn_validate_acc)
            xgb_avg_validate_acc.append(xgb_validate_acc)

            kfolds_feature_importances.append(feature_importances)

            fold += 1

        d[f'round_{round}_indexes'] = validate_set_idxes
        d['ground_truth'] = validate_set_gt
        d[f'ground_truth_strains'] = validate_set_gt_strains
        d[f'round_{round}_results'] = validate_set_preds


        print('CNN: ', cnn_avg_validate_acc, np.mean(cnn_avg_validate_acc))
        print('XGB: ', cnn_avg_validate_acc, np.mean(cnn_avg_validate_acc))


        kfolds_masks = avg_dict(kfolds_masks, 5)
        kfold_samples = avg_dict(kfold_samples, 5)

        kfolds_feature_importances = np.mean(kfolds_feature_importances, axis=0)

        return kfolds_masks, kfold_samples, kfolds_feature_importances, \
               np.mean(cnn_avg_validate_acc), np.mean(xgb_avg_validate_acc), test_set_prediction

    def main(self):

        print('\n+++++++++++++++++++++++++++++++ Using {} +++++++++++++++++++++++++++++++\n'.format(self.csv_path))
        self.get_features(self.csv_path)
        all_rounds_masks = defaultdict(list)
        all_rounds_samples = defaultdict(list)
        all_rounds_feature_importances = []
        all_rounds_cnn_acc = 0
        all_rounds_xgb_acc = 0

        max_acc = 0
        all_rounds = 5

        test_set_to_save = pd.DataFrame()

        d = pd.DataFrame({})
        for round in range(all_rounds):
            print('\n=================== Round {} =====================\n'.format(round))
            kfolds_masks, kfold_samples, kfolds_feature_importances, kfolds_cnn_acc, kfolds_xgb_acc, test_set_prediction = self.run(
                round, d)

            print('** test_set_prediction: ', test_set_prediction)
            for k, v in test_set_prediction.items():
                test_set_to_save[k] = v
            print(test_set_to_save)
            # test_set_to_save.to_csv(
            #     os.path.join(
            #         RESULT_PATH,
            #         f'unknown_{opt.blind_sample}_result_{3 if opt.level == "virus_type" else 9}.csv'))

            if kfolds_cnn_acc > max_acc:
                torch.save(self.cnn1d, os.path.join('saved_pkls',
                                                    '{}.pkl'.format(opt.virus_type)))

                print('\n---- Since {} > {}, so CNN model saved to {}\n'.format(
                    kfolds_cnn_acc, max_acc,
                    os.path.join('saved_pkls',
                                 '{}.pkl'.format(opt.virus_type))
                ))

                max_acc = kfolds_cnn_acc

            visualize_1d_heatmap(kfolds_masks, kfold_samples, kfolds_feature_importances, self.feature_list,
                                 round=round, fold='1-5', save=True)

            all_rounds_masks = update_dict(kfolds_masks, all_rounds_masks)
            all_rounds_samples = update_dict(kfold_samples, all_rounds_samples)
            all_rounds_samples = kfold_samples
            all_rounds_feature_importances.append(kfolds_feature_importances)
            all_rounds_cnn_acc += kfolds_cnn_acc
            all_rounds_xgb_acc += kfolds_xgb_acc

        all_rounds_masks = avg_dict(all_rounds_masks, all_rounds)

        all_rounds_feature_importances = np.mean(all_rounds_feature_importances, axis=0)
        all_rounds_cnn_acc /= all_rounds
        all_rounds_xgb_acc /= all_rounds

        visualize_1d_heatmap(all_rounds_masks, all_rounds_samples, all_rounds_feature_importances,
                             self.feature_list,
                             round=f'1-{all_rounds}', fold=f'1-{all_rounds}', save=True)

        print('\n****** After {} rounds, cnn acc: {}, xgb ac: {} ******\n'.format(all_rounds, all_rounds_cnn_acc,
                                                                                  all_rounds_xgb_acc))


