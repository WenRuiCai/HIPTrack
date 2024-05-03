import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class NFSDataset(BaseDataset):
    """ NFS dataset.
    Publication:
        Need for Speed: A Benchmark for Higher Frame Rate Object Tracking
        H. Kiani Galoogahi, A. Fagg, C. Huang, D. Ramanan, and S.Lucey
        ICCV, 2017
        http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Need_for_Speed_ICCV_2017_paper.pdf
    Download the dataset from http://ci2cv.net/nfs/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nfs_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/30/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}/30/{}'.format(self.base_path, sequence_path, sequence_info['anno_path'])
        #import pdb; pdb.set_trace()
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        ground_truth_rect[:,2:4] = ground_truth_rect[:,2:4] - ground_truth_rect[:,0:2]
        #import pdb; pdb.set_trace()
        return Sequence(sequence_info['name'], frames, 'nfs', ground_truth_rect[0::8,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Gymnastics", "path": "Gymnastics", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "Gymnastics_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "MachLoop_jet", "path": "MachLoop_jet", "startFrame": 1, "endFrame": 99, "nz": 5, "ext": "jpg", "anno_path": "MachLoop_jet_new.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "Skiing_red", "path": "Skiing_red", "startFrame": 1, "endFrame": 69, "nz": 5, "ext": "jpg", "anno_path": "Skiing_red_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "Skydiving", "path": "Skydiving", "startFrame": 1, "endFrame": 196, "nz": 5, "ext": "jpg", "anno_path": "Skydiving_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "airboard_1", "path": "airboard_1", "startFrame": 1, "endFrame": 425, "nz": 5, "ext": "jpg", "anno_path": "airboard_1_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "airplane_landing", "path": "airplane_landing", "startFrame": 1, "endFrame": 81, "nz": 5, "ext": "jpg", "anno_path": "airplane_landing_new.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "airtable_3", "path": "airtable_3", "startFrame": 1, "endFrame": 482, "nz": 5, "ext": "jpg", "anno_path": "airtable_3_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "basketball_1", "path": "basketball_1", "startFrame": 1, "endFrame": 282, "nz": 5, "ext": "jpg", "anno_path": "basketball_1_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "basketball_2", "path": "basketball_2", "startFrame": 1, "endFrame": 102, "nz": 5, "ext": "jpg", "anno_path": "basketball_2_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "basketball_3", "path": "basketball_3", "startFrame": 1, "endFrame": 421, "nz": 5, "ext": "jpg", "anno_path": "basketball_3_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "basketball_6", "path": "basketball_6", "startFrame": 1, "endFrame": 224, "nz": 5, "ext": "jpg", "anno_path": "basketball_6_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "basketball_7", "path": "basketball_7", "startFrame": 1, "endFrame": 240, "nz": 5, "ext": "jpg", "anno_path": "basketball_7_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "basketball_player", "path": "basketball_player", "startFrame": 1, "endFrame": 369, "nz": 5, "ext": "jpg", "anno_path": "basketball_player_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "basketball_player_2", "path": "basketball_player_2", "startFrame": 1, "endFrame": 437, "nz": 5, "ext": "jpg", "anno_path": "basketball_player_2_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "beach_flipback_person", "path": "beach_flipback_person", "startFrame": 1, "endFrame": 61, "nz": 5, "ext": "jpg", "anno_path": "beach_flipback_person_new.txt", "object_class": "person head", 'occlusion': False},
            {"name": "bee", "path": "bee", "startFrame": 1, "endFrame": 45, "nz": 5, "ext": "jpg", "anno_path": "bee_new.txt", "object_class": "insect", 'occlusion': False},
            {"name": "biker_acrobat", "path": "biker_acrobat", "startFrame": 1, "endFrame": 128, "nz": 5, "ext": "jpg", "anno_path": "biker_acrobat_new.txt", "object_class": "bicycle", 'occlusion': False},
            {"name": "biker_all_1", "path": "biker_all_1", "startFrame": 1, "endFrame": 113, "nz": 5, "ext": "jpg", "anno_path": "biker_all_1_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "biker_head_2", "path": "biker_head_2", "startFrame": 1, "endFrame": 132, "nz": 5, "ext": "jpg", "anno_path": "biker_head_2_new.txt", "object_class": "person head", 'occlusion': False},
            {"name": "biker_head_3", "path": "biker_head_3", "startFrame": 1, "endFrame": 254, "nz": 5, "ext": "jpg", "anno_path": "biker_head_3_new.txt", "object_class": "person head", 'occlusion': False},
            {"name": "biker_upper_body", "path": "biker_upper_body", "startFrame": 1, "endFrame": 194, "nz": 5, "ext": "jpg", "anno_path": "biker_upper_body_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "biker_whole_body", "path": "biker_whole_body", "startFrame": 1, "endFrame": 572, "nz": 5, "ext": "jpg", "anno_path": "biker_whole_body_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "billiard_2", "path": "billiard_2", "startFrame": 1, "endFrame": 604, "nz": 5, "ext": "jpg", "anno_path": "billiard_2_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "billiard_3", "path": "billiard_3", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "billiard_3_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "billiard_6", "path": "billiard_6", "startFrame": 1, "endFrame": 771, "nz": 5, "ext": "jpg", "anno_path": "billiard_6_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "billiard_7", "path": "billiard_7", "startFrame": 1, "endFrame": 724, "nz": 5, "ext": "jpg", "anno_path": "billiard_7_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "billiard_8", "path": "billiard_8", "startFrame": 1, "endFrame": 778, "nz": 5, "ext": "jpg", "anno_path": "billiard_8_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "bird_2", "path": "bird_2", "startFrame": 1, "endFrame": 476, "nz": 5, "ext": "jpg", "anno_path": "bird_2_new.txt", "object_class": "bird", 'occlusion': False},
            {"name": "book", "path": "book", "startFrame": 1, "endFrame": 288, "nz": 5, "ext": "jpg", "anno_path": "book_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "bottle", "path": "bottle", "startFrame": 1, "endFrame": 2103, "nz": 5, "ext": "jpg", "anno_path": "bottle_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "bowling_1", "path": "bowling_1", "startFrame": 1, "endFrame": 303, "nz": 5, "ext": "jpg", "anno_path": "bowling_1_new.txt", "object_class": "ball", 'occlusion': True},
            {"name": "bowling_2", "path": "bowling_2", "startFrame": 1, "endFrame": 710, "nz": 5, "ext": "jpg", "anno_path": "bowling_2_new.txt", "object_class": "ball", 'occlusion': True},
            {"name": "bowling_3", "path": "bowling_3", "startFrame": 1, "endFrame": 271, "nz": 5, "ext": "jpg", "anno_path": "bowling_3_new.txt", "object_class": "ball", 'occlusion': True},
            {"name": "bowling_6", "path": "bowling_6", "startFrame": 1, "endFrame": 260, "nz": 5, "ext": "jpg", "anno_path": "bowling_6_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "bowling_ball", "path": "bowling_ball", "startFrame": 1, "endFrame": 275, "nz": 5, "ext": "jpg", "anno_path": "bowling_ball_new.txt", "object_class": "ball", 'occlusion': True},
            {"name": "bunny", "path": "bunny", "startFrame": 1, "endFrame": 705, "nz": 5, "ext": "jpg", "anno_path": "bunny_new.txt", "object_class": "mammal", 'occlusion': False},
            {"name": "car", "path": "car", "startFrame": 1, "endFrame": 2020, "nz": 5, "ext": "jpg", "anno_path": "car_new.txt", "object_class": "car", 'occlusion': True},
            {"name": "car_camaro", "path": "car_camaro", "startFrame": 1, "endFrame": 36, "nz": 5, "ext": "jpg", "anno_path": "car_camaro_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "car_drifting", "path": "car_drifting", "startFrame": 1, "endFrame": 173, "nz": 5, "ext": "jpg", "anno_path": "car_drifting_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "car_jumping", "path": "car_jumping", "startFrame": 1, "endFrame": 22, "nz": 5, "ext": "jpg", "anno_path": "car_jumping_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "car_rc_rolling", "path": "car_rc_rolling", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "car_rc_rolling_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "car_rc_rotating", "path": "car_rc_rotating", "startFrame": 1, "endFrame": 80, "nz": 5, "ext": "jpg", "anno_path": "car_rc_rotating_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "car_side", "path": "car_side", "startFrame": 1, "endFrame": 108, "nz": 5, "ext": "jpg", "anno_path": "car_side_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "car_white", "path": "car_white", "startFrame": 1, "endFrame": 2063, "nz": 5, "ext": "jpg", "anno_path": "car_white_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "cheetah", "path": "cheetah", "startFrame": 1, "endFrame": 167, "nz": 5, "ext": "jpg", "anno_path": "cheetah_new.txt", "object_class": "mammal", 'occlusion': True},
            {"name": "cup", "path": "cup", "startFrame": 1, "endFrame": 1281, "nz": 5, "ext": "jpg", "anno_path": "cup_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "cup_2", "path": "cup_2", "startFrame": 1, "endFrame": 182, "nz": 5, "ext": "jpg", "anno_path": "cup_2_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "dog", "path": "dog", "startFrame": 1, "endFrame": 1030, "nz": 5, "ext": "jpg", "anno_path": "dog_new.txt", "object_class": "dog", 'occlusion': True},
            {"name": "dog_1", "path": "dog_1", "startFrame": 1, "endFrame": 168, "nz": 5, "ext": "jpg", "anno_path": "dog_1_new.txt", "object_class": "dog", 'occlusion': False},
            {"name": "dog_2", "path": "dog_2", "startFrame": 1, "endFrame": 594, "nz": 5, "ext": "jpg", "anno_path": "dog_2_new.txt", "object_class": "dog", 'occlusion': True},
            {"name": "dog_3", "path": "dog_3", "startFrame": 1, "endFrame": 200, "nz": 5, "ext": "jpg", "anno_path": "dog_3_new.txt", "object_class": "dog", 'occlusion': False},
            {"name": "dogs", "path": "dogs", "startFrame": 1, "endFrame": 198, "nz": 5, "ext": "jpg", "anno_path": "dogs_new.txt", "object_class": "dog", 'occlusion': True},
            {"name": "dollar", "path": "dollar", "startFrame": 1, "endFrame": 1426, "nz": 5, "ext": "jpg", "anno_path": "dollar_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "drone", "path": "drone", "startFrame": 1, "endFrame": 70, "nz": 5, "ext": "jpg", "anno_path": "drone_new.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "ducks_lake", "path": "ducks_lake", "startFrame": 1, "endFrame": 107, "nz": 5, "ext": "jpg", "anno_path": "ducks_lake_new.txt", "object_class": "bird", 'occlusion': False},
            {"name": "exit", "path": "exit", "startFrame": 1, "endFrame": 359, "nz": 5, "ext": "jpg", "anno_path": "exit_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "first", "path": "first", "startFrame": 1, "endFrame": 435, "nz": 5, "ext": "jpg", "anno_path": "first_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "flower", "path": "flower", "startFrame": 1, "endFrame": 448, "nz": 5, "ext": "jpg", "anno_path": "flower_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "footbal_skill", "path": "footbal_skill", "startFrame": 1, "endFrame": 131, "nz": 5, "ext": "jpg", "anno_path": "footbal_skill_new.txt", "object_class": "ball", 'occlusion': True},
            {"name": "helicopter", "path": "helicopter", "startFrame": 1, "endFrame": 310, "nz": 5, "ext": "jpg", "anno_path": "helicopter_new.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "horse_jumping", "path": "horse_jumping", "startFrame": 1, "endFrame": 117, "nz": 5, "ext": "jpg", "anno_path": "horse_jumping_new.txt", "object_class": "horse", 'occlusion': True},
            {"name": "horse_running", "path": "horse_running", "startFrame": 1, "endFrame": 139, "nz": 5, "ext": "jpg", "anno_path": "horse_running_new.txt", "object_class": "horse", 'occlusion': False},
            {"name": "iceskating_6", "path": "iceskating_6", "startFrame": 1, "endFrame": 603, "nz": 5, "ext": "jpg", "anno_path": "iceskating_6_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "jellyfish_5", "path": "jellyfish_5", "startFrame": 1, "endFrame": 746, "nz": 5, "ext": "jpg", "anno_path": "jellyfish_5_new.txt", "object_class": "invertebrate", 'occlusion': False},
            {"name": "kid_swing", "path": "kid_swing", "startFrame": 1, "endFrame": 169, "nz": 5, "ext": "jpg", "anno_path": "kid_swing_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "motorcross", "path": "motorcross", "startFrame": 1, "endFrame": 39, "nz": 5, "ext": "jpg", "anno_path": "motorcross_new.txt", "object_class": "vehicle", 'occlusion': True},
            {"name": "motorcross_kawasaki", "path": "motorcross_kawasaki", "startFrame": 1, "endFrame": 65, "nz": 5, "ext": "jpg", "anno_path": "motorcross_kawasaki_new.txt", "object_class": "vehicle", 'occlusion': False},
            {"name": "parkour", "path": "parkour", "startFrame": 1, "endFrame": 58, "nz": 5, "ext": "jpg", "anno_path": "parkour_new.txt", "object_class": "person head", 'occlusion': False},
            {"name": "person_scooter", "path": "person_scooter", "startFrame": 1, "endFrame": 413, "nz": 5, "ext": "jpg", "anno_path": "person_scooter_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "pingpong_2", "path": "pingpong_2", "startFrame": 1, "endFrame": 1277, "nz": 5, "ext": "jpg", "anno_path": "pingpong_2_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "pingpong_7", "path": "pingpong_7", "startFrame": 1, "endFrame": 1290, "nz": 5, "ext": "jpg", "anno_path": "pingpong_7_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "pingpong_8", "path": "pingpong_8", "startFrame": 1, "endFrame": 296, "nz": 5, "ext": "jpg", "anno_path": "pingpong_8_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "purse", "path": "purse", "startFrame": 1, "endFrame": 968, "nz": 5, "ext": "jpg", "anno_path": "purse_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "rubber", "path": "rubber", "startFrame": 1, "endFrame": 1328, "nz": 5, "ext": "jpg", "anno_path": "rubber_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "running", "path": "running", "startFrame": 1, "endFrame": 677, "nz": 5, "ext": "jpg", "anno_path": "running_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "running_100_m", "path": "running_100_m", "startFrame": 1, "endFrame": 313, "nz": 5, "ext": "jpg", "anno_path": "running_100_m_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "running_100_m_2", "path": "running_100_m_2", "startFrame": 1, "endFrame": 337, "nz": 5, "ext": "jpg", "anno_path": "running_100_m_2_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "running_2", "path": "running_2", "startFrame": 1, "endFrame": 363, "nz": 5, "ext": "jpg", "anno_path": "running_2_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "shuffleboard_1", "path": "shuffleboard_1", "startFrame": 1, "endFrame": 42, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_1_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffleboard_2", "path": "shuffleboard_2", "startFrame": 1, "endFrame": 41, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_2_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffleboard_4", "path": "shuffleboard_4", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_4_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffleboard_5", "path": "shuffleboard_5", "startFrame": 1, "endFrame": 32, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_5_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffleboard_6", "path": "shuffleboard_6", "startFrame": 1, "endFrame": 52, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_6_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffletable_2", "path": "shuffletable_2", "startFrame": 1, "endFrame": 372, "nz": 5, "ext": "jpg", "anno_path": "shuffletable_2_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffletable_3", "path": "shuffletable_3", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "shuffletable_3_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "shuffletable_4", "path": "shuffletable_4", "startFrame": 1, "endFrame": 101, "nz": 5, "ext": "jpg", "anno_path": "shuffletable_4_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "ski_long", "path": "ski_long", "startFrame": 1, "endFrame": 274, "nz": 5, "ext": "jpg", "anno_path": "ski_long_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "soccer_ball", "path": "soccer_ball", "startFrame": 1, "endFrame": 163, "nz": 5, "ext": "jpg", "anno_path": "soccer_ball_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "soccer_ball_2", "path": "soccer_ball_2", "startFrame": 1, "endFrame": 1934, "nz": 5, "ext": "jpg", "anno_path": "soccer_ball_2_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "soccer_ball_3", "path": "soccer_ball_3", "startFrame": 1, "endFrame": 1381, "nz": 5, "ext": "jpg", "anno_path": "soccer_ball_3_new.txt", "object_class": "ball", 'occlusion': False},
            {"name": "soccer_player_2", "path": "soccer_player_2", "startFrame": 1, "endFrame": 475, "nz": 5, "ext": "jpg", "anno_path": "soccer_player_2_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "soccer_player_3", "path": "soccer_player_3", "startFrame": 1, "endFrame": 319, "nz": 5, "ext": "jpg", "anno_path": "soccer_player_3_new.txt", "object_class": "person", 'occlusion': True},
            {"name": "stop_sign", "path": "stop_sign", "startFrame": 1, "endFrame": 302, "nz": 5, "ext": "jpg", "anno_path": "stop_sign_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "suv", "path": "suv", "startFrame": 1, "endFrame": 2584, "nz": 5, "ext": "jpg", "anno_path": "suv_new.txt", "object_class": "car", 'occlusion': False},
            {"name": "tiger", "path": "tiger", "startFrame": 1, "endFrame": 1556, "nz": 5, "ext": "jpg", "anno_path": "tiger_new.txt", "object_class": "mammal", 'occlusion': False},
            {"name": "walking", "path": "walking", "startFrame": 1, "endFrame": 555, "nz": 5, "ext": "jpg", "anno_path": "walking_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "walking_3", "path": "walking_3", "startFrame": 1, "endFrame": 1427, "nz": 5, "ext": "jpg", "anno_path": "walking_3_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "water_ski_2", "path": "water_ski_2", "startFrame": 1, "endFrame": 47, "nz": 5, "ext": "jpg", "anno_path": "water_ski_2_new.txt", "object_class": "person", 'occlusion': False},
            {"name": "yoyo", "path": "yoyo", "startFrame": 1, "endFrame": 67, "nz": 5, "ext": "jpg", "anno_path": "yoyo_new.txt", "object_class": "other", 'occlusion': False},
            {"name": "zebra_fish", "path": "zebra_fish", "startFrame": 1, "endFrame": 671, "nz": 5, "ext": "jpg", "anno_path": "zebra_fish_new.txt", "object_class": "fish", 'occlusion': False},
        ]

        return sequence_info_list