from src.model.interpretation import interpretation
from src.model.pipeline import *
from src.utils.preprocessing import *
from src.utils.util import *

WORK_DIR = '/data/karenyyy/Virus2022/Accurate_Virus_Identification'

if __name__ == '__main__':
    if opt.task == 'classify':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        virus = VirusClassifier()
        virus.main()

    elif opt.task == 'aggregate':
        aggregate_positions(file_path=os.path.join(WORK_DIR, 'dataset/fullsets/CVB1'),
                            save_path=os.path.join(WORK_DIR, f'dataset/{opt.virus_type}/CVB1.txt'))

    elif opt.task == 'preprocess':
        run()
        file = pd.read_csv(os.path.join(RESULT_PATH, FEATURESET_FILE))
        file = file.dropna(axis=1)
        file.to_csv(os.path.join(RESULT_PATH, FEATURESET_FILE), index=False)

    elif opt.task == 'interpretation':

        overlap_dict = interpretation(order=4, box_pts=17)
        pprint(overlap_dict)
