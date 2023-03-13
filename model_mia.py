from config import *

from datasets.loader import dataset
from model_train import train, save

from mia import ShadowModels, AttackModels

# 一期计划，采用分割数据集的方式获得训练数据和影子数据
# 二期计划，采用爬山法或其他方法获取影子数据

# 1、载入训练好的被攻击的模型或是选中待被攻击的模型
TARGET_MODEL = MODEL(pretrained=True, mode_path=MIA_MODEL_PATH, 
                                   device=DEVICE, num_classes=NUM_CLASS)
# TARGET_MODEL = MODEL(device=DEVICE, num_classes=NUM_CLASS)
# OPTIMIZER = optim.Adam(TARGET_MODEL.parameters())
SHADOW_MODEL = MODEL(device=DEVICE, num_classes=NUM_CLASS)
ATTACK_MODEL = MODEL

# 如果想接着训练，增加epoch的话，那么数据必须得保存下来的

if __name__ == "__main__":
    # 2、获得影子数据，一期采用分割数据集的方式
    
    # ▲使用新拆分的数据训练
    # data = dataset(DATA_NAME, TRAIN_MODE)
    # train_size = len(data) // (NUM_SHADOW + 1)
    # shadow_size = len(data) - train_size
    # train_dataset, shadow_dataset = torch.utils.data.random_split(data, [train_size, shadow_size])
    # ▲存训练数据和影子数据
    # dataset_tuple = (train_dataset, shadow_dataset)
    # torch.save(dataset_tuple, MIA_DATA_PATH)
    
    # ▼读取拆分好的数据
    # train_dataset, shadow_dataset = torch.load(MIA_DATA_PATH)
    # print("Read finish, shadow dataset: ", len(shadow_dataset))
    
    # 训练模型
    # train(TARGET_MODEL, train_dataset, LOSS_FUNC, OPTIMIZER)
    # save(TARGET_MODEL, MIA_MODE, MIA_MODEL_PATH)
    
    # 2、训练影子数据
    # ▲获得影子数据
    # shadow_models = ShadowModels(shadow_dataset, NUM_SHADOW, NUM_CLASS, SHADOW_MODEL,
    #                              MIA_SHADOW_EPOCH, MIA_SHADOW_BATCH, MIA_RESULT_BATCH)
    # shadow_models.save_results(MIA_SHADOW_PATH)
    # ▼读取训练好的影子数据
    shadow_data = ShadowModels.load_results(MIA_SHADOW_PATH)
    
    # 3、训练攻击模型
    attack_models = AttackModels(NUM_CLASS, ATTACK_MODEL)
    attack_models.build(shadow_data, MIA_ATTACK_EPOCH)
    attack_models.save(MIA_ATTACK_PATH)
