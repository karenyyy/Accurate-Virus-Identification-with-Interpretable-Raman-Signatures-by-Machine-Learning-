from src.config import *


def update_dict(d1, d2):
    for k, v in d1.items():
        if k not in d2.keys():
            d2[k] = v
        else:
            d2[k] += v
    return d2


def avg_dict(d, divisor):
    d = {k: v / divisor for k, v in d.items()}
    return d


def visualize_1d_heatmap(masks, samples, feature_importances, feature_list, round, fold, save=False):
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    subpath = os.path.join(RESULT_PATH, opt.virus_type)
    if not os.path.exists(subpath):
        os.mkdir(subpath)

    plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(1 + len(CLASSES), 1,
                           height_ratios=[7] + [1] * len(CLASSES))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
    ax0 = plt.subplot(gs[0])

    ax0.plot(feature_list, feature_importances, color='blue')
    ax0.set_ylabel('Feature Importance')
    plt.xticks(np.arange(min(feature_list), max(feature_list), 150))
    plt.xlim(min(feature_list), max(feature_list))

    ax0_1 = ax0.twinx()
    for c in masks.keys():
        ax0_1.plot(feature_list, samples[c] + c * 0.03, label=CLASSES[c])
    ax0_1.set_ylabel('Normalized Signal Intensity')
    ax0_1.tick_params(axis='y')

    plt.legend()
    plt.title('Round {}, Fold {}: {}'.format(round, fold, opt.virus_type))

    cnn_feature_importances_all_classes = {}

    for c in np.arange(len(CLASSES)):
        current_mark = masks[c][0]

        cnn_feature_importances = [[int(feature), importance] for feature, importance in
                                   zip(feature_list, current_mark)]

        cnn_feature_importances = sorted(cnn_feature_importances, key=lambda x: x[1], reverse=True)

        cnn_feature_importances_all_classes[c] = cnn_feature_importances

        # current_mark[current_mark < np.quantile(current_mark.flatten(), opt.cnn_threshold)] = 0

        current_mark = current_mark.reshape(1, -1)
        current_mark = normalize(current_mark, copy=False, norm='l2')

        plt.subplot(gs[c + 1])

        sns.heatmap(pd.DataFrame(np.uint8(255 * current_mark)),
                    xticklabels=np.arange(600, 1800, 100),
                    cbar=True,
                    cmap='Reds',
                    cbar_kws=dict(use_gridspec=True, orientation='horizontal'),
                    )
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(CLASSES[c])

    print('Saved to {}/{}_round_{}_fold_{}_{}_heatmap.png'.format(subpath, opt.virus_type, round, fold, opt.virus_type))
    plt.savefig('{}/{}_round_{}_fold_{}_{}_heatmap.png'.format(subpath, opt.virus_type, round, fold, opt.virus_type))

    plt.show()

    importance_df = {}
    for virus_subtype, value in cnn_feature_importances_all_classes.items():
        importance_scores = []
        wvnumbers = []
        for pair in value:
            wvnumbers.append(pair[0])
            importance_scores.append(pair[1])
        importance_df['{}_Wavenumber'.format(CLASSES[virus_subtype])] = wvnumbers
        importance_df['{}_Importance'.format(CLASSES[virus_subtype])] = importance_scores

    if save:
        pd.DataFrame(importance_df).to_csv(os.path.join(subpath,
                                                        '{}_class_specific_importance.csv'.format(opt.virus_type)),
                                           index=False)
        pd.DataFrame(
            {
                'Wavenumber': feature_list,
                'Importance': feature_importances
            }
        ).to_csv(os.path.join(subpath, '{}_xgb_importance.csv'.format(opt.virus_type)), index=False)


def noise_filtering(x, box_pts):
    smoothed_x = savgol_filter(x, box_pts, 2)
    return smoothed_x


# "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005
def baseline_als(y, lam=100000000, p=0.0001, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in np.arange(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def bfg_sub(y, lam=100000, p=0.0001, niter=10):
    p1 = baseline_als(y, lam=lam, p=p, niter=niter)
    y = y - p1
    return y, p1

