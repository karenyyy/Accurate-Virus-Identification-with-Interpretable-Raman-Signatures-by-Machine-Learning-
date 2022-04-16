from src.config import *
from src.utils.util import *


def union(data):
    intervals = [Interval(begin, end) for (begin, end) in data]
    u = Union(*intervals)
    return [list(u.args[:2])] if isinstance(u, Interval) \
        else list(u.args)


def interpretation(order=5, box_pts=15, finetune=False):
    group_dict = defaultdict(dict)
    wv_overlap_dict = defaultdict(dict)
    wv_overlap_dict2 = defaultdict(dict)
    colors = ['skyblue', 'y', 'm', 'g', 'coral']*3
    for c in CLASSES:
        df = pd.read_csv(os.path.join(RESULT_PATH, f"{opt.virus_type}/{opt.virus_type}_class_specific_importance.csv"))
        df = df.sort_values(["{}_Wavenumber".format(c)])
        df["{}_Importance_smoothed".format(c)] = noise_filtering(list(df["{}_Importance".format(c)].values),
                                                                 box_pts=box_pts)

        df.to_csv(os.path.join(RESULT_PATH, f'{opt.virus_type}/{opt.virus_type}_class_specific_importance.csv'),
                  index=False)

        x = list(df["{}_Wavenumber".format(c)].values)
        y = df["{}_Importance_smoothed".format(c)].to_numpy()

        plt.figure(figsize=(10, 5))
        plt.fill_between(list(df["{}_Wavenumber".format(c)].values),
                         df["{}_Importance_smoothed".format(c)], color=colors[CLASSES.index(c)], alpha=0.2)

        plt.title(c)
        lefts = []
        rights = []
        weights = []

        peaks = argrelextrema(y, np.greater, order=order)[0]
        peaks = np.append(peaks, [0])

        threshold = np.percentile(y, 40)
        # threshold = np.mean(y)

        for idx in range(len(y) - 1):
            if y[idx] < threshold and y[idx + 1] > threshold and len(lefts) == len(rights):
                lefts.append(idx)

            if y[idx] > threshold and y[idx + 1] < threshold and len(lefts) == len(rights) + 1:
                rights.append(idx)

            # print(len(lefts), len(rights))
            if len(lefts) > 1 and len(rights) > 1:
                for p in peaks:
                    # print(x[lefts[-1]], x[p], x[rights[-1]], y[p], threshold)
                    if x[lefts[-1]] < x[p] < x[rights[-1]] and y[p] < threshold:
                        lefts = lefts[:-1]
                        rights = rights[:-1]

        # lefts = lefts[:-1]
        # plt.scatter(x, y, c='orange', s=1)

        # plt.scatter([x[i] for i in lefts], [y[i] for i in lefts], c='r', s=8)
        # plt.scatter([x[i] for i in rights], [y[i] for i in rights], c='m', s=8)
        plt.axhline(y=threshold, color='r', linestyle='--')

        intervals = [(x[lefts[i]], x[rights[i]]) for i in range(len(lefts))]

        for interval in intervals:
            plt.plot([interval[0], interval[1]], [-0.005, -0.005], c='red')

        fg = pd.read_csv(os.path.join(RESULT_PATH, 'dataset/stack_functional_groups.csv'))
        fg['diff'] = fg['end'] - fg['start']
        grouped_cnts = fg.groupby(['group'])['start'].count()
        grouped_sum = fg.groupby(['group'])['diff'].sum()
        grouped_range = fg.groupby(['group'])['diff'].apply(list)
        grouped_start = fg.groupby(['group'])['start'].apply(list)

        pair_fg = {}
        weights_for_matched = defaultdict(list)

        for idx, row in fg.iterrows():
            start = row['start']
            end = row['end']
            # print(row['group'])
            for idx, interval in enumerate(intervals):

                if start <= interval[0] <= end or start <= interval[1] <= end or (start >= interval[0] and end <= interval[1]):
                    tmp_w = []
                    for idx in range(len(x)):
                        if max(start, interval[0]) <= x[idx] <= min(interval[1], end):
                            # print(max(start, interval[0]), x[idx], min(interval[1], end))
                            tmp_w.append(y[idx]/(max(y)))
                    # print(y[idx], y[idx]/(max(y) - min(y)))
                    if len(tmp_w) > 0:
                        weights_for_matched[row['group']].append(np.mean(tmp_w))

            # print(weights_for_matched[row['group']])

        for idx, row in fg.iterrows():
            start = row['start']
            end = row['end']
            pair_fg[row['group']] = row['type']

            if 'Lipid' in row['group']:
                plt.plot([row['start'], row['end']], [-0.01, -0.01], c='blue')
            if row['group'] == 'Amide I':
                plt.plot([row['start'], row['end']], [-0.015, -0.015], c='m')
            if row['group'] == 'Amide III':
                plt.plot([row['start'], row['end']], [-0.02, -0.02], c='m')
            if row['group'] == 'RNA':
                plt.plot([row['start'], row['end']], [-0.025, -0.025], c='green')
            if row['group'] == 'Tyrosine':
                plt.plot([row['start'], row['end']], [-0.03, -0.03], c='black')
            if row['group'] == 'Phenylalanine':
                plt.plot([row['start'], row['end']], [-0.035, -0.035], c='black')
            if row['group'] == 'RBD Protein':
                plt.plot([row['start'], row['end']], [-0.04, -0.04], c='orange')


            for idx, interval in enumerate(intervals):
                if start <= interval[0] <= end or start <= interval[1] <= end or (start >= interval[0] and end <= interval[1]):
                    if min(interval[1], end) - max(start, interval[0]) > 0:
                        if c == 'non_env' and row['group'] == 'Lipid':
                            print(start, interval[0], interval[1], end, min(interval[1], end) - max(start, interval[0]))

                        if row['group'] in group_dict[c].keys():
                            group_dict[c][row['group']] += 1
                            # wv_overlap_dict[c][row['group']].append(
                            #     (max(start, interval[0]), (min(interval[1], end) - max(start, interval[0]))))
                            wv_overlap_dict[c][row['group']].append(
                                (min(interval[1], end) - max(start, interval[0])))

                        else:
                            group_dict[c][row['group']] = 1
                            # wv_overlap_dict[c][row['group']] = [
                            #     (max(start, interval[0]), (min(interval[1], end) - max(start, interval[0])))]
                            wv_overlap_dict[c][row['group']] = [
                                (min(interval[1], end) - max(start, interval[0]))]

        # plt.axis('off')
        plt.show()

        for key, val in group_dict[c].items():
            weights_for_matched[key] = [i/np.sum(weights_for_matched[key]) for i in weights_for_matched[key]]

            # if key == 'Lipid':
            #     print(weights_for_matched[key])
            #     print(wv_overlap_dict[c][key])
            #     print('before', np.sum(wv_overlap_dict[c][key][1]), grouped_sum[key])

            # wv_overlap_dict[c][key] = [i*j*len(weights_for_matched[key])
            #                            for i, j in zip(wv_overlap_dict[c][key],
            #                                             weights_for_matched[key])]

            match_ratio = np.sum(wv_overlap_dict[c][key]) / grouped_sum[key]

            # match_ratio = []
            eq = ''
            # for i, w in zip(wv_overlap_dict[c][key], weights_for_matched[key]):
            #     for idx in range(len(grouped_start[key])):
            #         if grouped_start[key][idx] <= i[0] <= grouped_start[key][idx] + grouped_range[key][idx]:
            #             # print(grouped_start[key][idx], i[0], grouped_start[key][idx] + grouped_range[key][idx])
            #             eq += ('+' if len(eq) > 0 else '') + '{:.3f}*({}/{})'.format(w, i[1], grouped_range[key][idx])
            #             match_ratio.append(w*(i[1]/grouped_range[key][idx]))
            #
            # assert len(match_ratio) == len(wv_overlap_dict[c][key])
            #
            # match_ratio = np.sum(match_ratio)

            # if key == 'Lipid':
            #     print('after', np.sum(wv_overlap_dict[c][key]), np.sum(grouped_range[key]))

            if not finetune:
                if (
                        pair_fg[key] == 'functional_group' and
                        match_ratio >= 0
                ) or (
                        pair_fg[key] == 'amino_acids' and
                        (match_ratio >= 0)
                ) or (
                        pair_fg[key] in ['protein', 'lipid'] and
                        (match_ratio >= 0)
                ) or (
                        pair_fg[key] == 'nucleic_acid' and
                        (match_ratio >= 0)
                ):
                    if val > grouped_cnts[key] or np.sum(wv_overlap_dict[c][key]) > grouped_sum[key]:
                        wv_overlap_dict2[c][(pair_fg[key], key)] = \
                            (eq,
                             '{} / {}'.format(val, grouped_cnts[key]),
                             1 if match_ratio > 1
                             else float(match_ratio)
                             )
                    else:
                        wv_overlap_dict2[c][(pair_fg[key], key)] = \
                            (eq,
                             '{} / {}'.format(val, grouped_cnts[key]),
                             float(match_ratio)
                             )
            else:
                if val > grouped_cnts[key] or np.sum(wv_overlap_dict[c][key]) > grouped_sum[key]:
                    wv_overlap_dict2[c][(pair_fg[key], key)] = \
                        (eq,
                         '{} / {}'.format(val, grouped_cnts[key]),
                         1 if match_ratio > 1
                         else float(match_ratio)
                         )
                else:
                    wv_overlap_dict2[c][(pair_fg[key], key)] = \
                        (eq,
                         '{} / {}'.format(val, grouped_cnts[key]),
                         float(match_ratio)
                         )

    return wv_overlap_dict2
