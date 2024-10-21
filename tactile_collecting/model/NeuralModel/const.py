
class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range): # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item) # or super(RangeDict, self) for Python 2


CLASS_TYPE_INDEX = {
    # "stand": 0,
    # "right": 1,
    # "left": 2,
    "front": 0,
    "back": 1
}

INDEX_CLASS_TYPE = {
    # 0: "stand",
    # 1: "right",
    # 2: "left",
    0: "front",
    1: "back"
}

# EXCEPT_NAMES = [
#     'stand', 'right', 'left'
# ]
# EXCEPT_INDICES = [0, 1, 2]
EXCEPT_INDICES = []

ANGLE_CLASSES_NUM = 12
ANGLE_INTERVAL = int(360/ANGLE_CLASSES_NUM)
ANGLE_TO_CLASS = {}

temp = 0
for i in range(0, 360, ANGLE_INTERVAL):
    ANGLE_TO_CLASS[range(i, i+ANGLE_INTERVAL)] = temp
    temp += 1
ANGLE_TO_CLASS = RangeDict(ANGLE_TO_CLASS)

CLASS_NUM = len(CLASS_TYPE_INDEX.keys()) - len(EXCEPT_INDICES)
REGRESS_NUM = 2

'''
DATA_PATHS = [
    "./data/cyh1/",
    "./data/cyh2/",
    "./data/lsh/",
    "./data/pdh/",
    "./data/yws/",
    "./data/bic/",
    "./data/jhc/",
    "./data/mjy/",
    "./data/osm/",
    "./data/ph/",
    "./data/pth/"
]

'''
DATA_PATHS = [
    # "./data/cyh/",
    # "./data/pdh/",
    # "./data/shl/",
    # "./data/jhc/",
    # "./data/his/",
    "./model/whole_data_person/pdh/",
    "./model/whole_data_person/cyh/",
    "./model/whole_data_person/jhc/",
    "./model/whole_data_person/his/"
    ]

TEST_DATA_PATHS = [
    "./model/whole_data_person/ish/",

]





