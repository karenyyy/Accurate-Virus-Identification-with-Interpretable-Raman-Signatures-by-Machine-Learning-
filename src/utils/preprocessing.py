from src.config import *


def get_name(filename):
    return filename.split('.')[0]


def aggregate_positions(file_path, save_path):

    df_final = pd.DataFrame()
    for filename in os.listdir(file_path):
        print(filename)
        df = pd.read_csv(os.path.join(
            file_path,
            filename
        ), sep="\t", header=None)

        df.columns = ['pos1', 'pos2', 'wavenumber', 'signal']

        df_new = pd.DataFrame(
            {
                'pos1': df['pos1'],
                'pos2': df['pos2'],
                'wavenumber': df['wavenumber'],
                'signal': df['signal']
            }
        )
        print(df_new)
        df_new.columns = ['pos1', 'pos2', 'wavenumber', 'signal']
        # print(df_new)
        df_final = pd.concat([df_final, df_new])

    df_final.to_csv(save_path, header=None, sep='\t', index=False)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if d.endswith(EXT)]
    classes = list(map(get_name, classes))
    classes.sort()
    class_to_idx = CLASSES2IDX
    return classes, class_to_idx


def merge_df(dataframe, label, cnt_wave, class_to_idx=CLASSES2IDX):
    idx_to_class = {idx: c for c, idx in class_to_idx.items()}
    dataframe.columns = ['pos1', 'pos2', 'wavenumber', 'signal']

    CNT_ROW = len(dataframe.index) // cnt_wave

    df = dataframe[0:cnt_wave]

    df_merged = pd.DataFrame({'pos1': [df.iloc[0]['pos1']],
                              'pos2': [df.iloc[0]['pos2']]})

    for idx_wave in range(cnt_wave):
        wavenumber = df.iloc[idx_wave]['wavenumber']
        signal = df.iloc[idx_wave]['signal']

        df_merged[str(int(wavenumber))] = signal
        df_merged['label'] = label

    cnt = 0
    for idx in range(1, CNT_ROW):

        df = dataframe[idx * cnt_wave:(idx + 1) * cnt_wave]

        df_new = pd.DataFrame({'pos1': [df.iloc[0]['pos1']], 'pos2': [df.iloc[0]['pos2']]})
        cnt_dup = 1
        for idx_wave in range(cnt_wave):
            wavenumber = df.iloc[idx_wave]['wavenumber']
            signal = df.iloc[idx_wave]['signal']

            if opt.remove_sudden_peaks:
                if str(int(wavenumber + 1)) in df_new.columns and signal > 1.5 * \
                        df_new[str(int(wavenumber) + 1)].values[0]:

                    signal = df_new[str(int(wavenumber) + 1)].values[0]
                elif str(int(wavenumber - 1)) in df_new.columns and signal > 1.5 * \
                        df_new[str(int(wavenumber) - 1)].values[
                            0]:
                    signal = df_new[str(int(wavenumber) - 1)].values[0]

            if str(int(wavenumber)) in df_new.columns:
                cnt_dup += 1
                df_new[str(int(wavenumber))] = (df_new[str(int(wavenumber))] * (cnt_dup - 1) + signal) / cnt_dup
            else:
                cnt_dup = 1
                df_new[str(int(wavenumber))] = signal
            df_new['label'] = label
        df_merged = pd.concat([df_merged, df_new], sort=False)
        if cnt % 10 == 0:
            print('{:.2f}% has done for {}'.format(cnt / CNT_ROW * 100, idx_to_class[label]))
        cnt += 1
    print(df_merged.shape)
    return df_merged


def run():
    tgt_path = os.path.join(RESULT_PATH, opt.virus_type)
    classes, class_to_idx = find_classes(tgt_path)
    print(classes)
    print(class_to_idx)
    file_final = pd.DataFrame()

    for filename in os.listdir(tgt_path):
        if filename.endswith(EXT):
            print(filename)
            file = pd.read_csv(os.path.join(tgt_path, filename), sep="\t", header=None)

            cnt = 0
            pos1 = file.iloc[0][0]
            pos2 = file.iloc[0][1]
            for idx in range(int(1e5)):
                row = file.iloc[idx]
                if row[0] == pos1 and row[1] == pos2:
                    cnt += 1
                else:
                    break
            # label = class_to_idx[filename.split('.txt')[0].split('_')[0]]

            label = class_to_idx[filename.split('.txt')[0]]

            file_final = pd.concat([file_final, merge_df(file, label, cnt, class_to_idx)], sort=False)

            file_final = file_final.replace('', np.nan)
            file_final = file_final.dropna(axis=1)

            print(file_final.shape, Counter(list(file_final['label'])), 'saved!')
            file_final.to_csv(os.path.join(RESULT_PATH, FEATURESET_FILE), sep=",", index=False)

    file_final.to_csv(os.path.join(RESULT_PATH, FEATURESET_FILE), sep=",", index=False)

    print(file_final.shape)
    print('all done!')
