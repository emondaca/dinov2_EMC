import os
import os, json, shutil, random
from dataset import get_data

import torch
import copy, time
from train import train_model
from evaluation import evaluate_model
import pandas as pd
import argparse

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))


def main(subLabel, wrs_mode, radDinoType, raddinoHead, classWeighted, DataAug, oversampling, label_count):
    wrs_mode = "t" if wrs_mode == "" else wrs_mode
    print("Start the Main ...")

    data = pd.read_csv(os.path.join(dir,"scripts/master_table.csv"))
    data_filtered = data[['ImageID', 'label', 'label_group','sentence_en']]

    data_filtered = data_filtered[data_filtered['label'] != "Normal"]

    print(data_filtered.head())

    images_src = os.path.join(dir,'dataset/PadChest_GR')
    RadDino_src = os.path.join(dir,'scripts/RadDino_PadChest/dinov2_src')
    RadDinoWeights = os.path.join(dir,'models/backbone_compatible.safetensors')
    #RadDinoWeights = os.path.join(dir,'models/radDinoMaria2.safetensors')

    Head_RadDinoWeights = os.path.join(dir,'models/dino_head.safetensors')

    supects_terms_path = os.path.join(dir,"scripts/terminos_sospechosos_coincidencia_lexica.xlsx")
    #model = radDino_Head(768,Head_RadDinoWeights, 25).to('cuda')
    #print(model)
    #model = None

    batch_size = 32
    train_loader, val_loader, test_loader, NUM_Classes,pos_weight = get_data(data_filtered, images_src,RadDino_src,RadDinoWeights, subLabel, IMAGE_SIZE = 518, batch_size = batch_size,wrs_mode= wrs_mode, radDinoType = radDinoType, xlsx_path = supects_terms_path, DataAug = DataAug, oversampler = oversampling, label_count = label_count)

    if subLabel:
        save_loss_path = os.path.join(dir,'Graphs/PadChest/loss_curve_ResNet50_Sublabel_Albumentations.png')
    else: 
        #wrs_mode.capitalize()
        save_loss_path = os.path.join(dir,'Graphs/PadChest/UniModal_Image/loss_curve_RadDinoFreeze_MultiModal_FineTuningTransformerHead.png')
    model = train_model(train_loader, val_loader, NUM_Classes,pos_weight, save_loss_path,RadDino_src,RadDinoWeights,Head_RadDinoWeights,classWeighted=classWeighted,  num_epochs=300, radDinoType = radDinoType, raddinoHead = raddinoHead)

    torch.save(model.state_dict(), os.path.join(dir,'models/RadDinoMAIRA1FT_5label.pth'))

    evaluate_model(model, test_loader,eval_test = True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main training script with configurable options.")
    parser.add_argument("--subLabel", action="store_true", help="Enable SubLabel mode (default: False)")
    parser.add_argument("--agg_mode", choices=["mean", "max", "none"], default="none",
                        help="Aggregation mode for label handling (default: none)")
    parser.add_argument("--radDino", type=int, choices=[1, 2], default=1,
                        help="Select RadDino type: 1=MAIRA-1, 2=MAIRA-2 (default: 1)")
    parser.add_argument("--radDinoHead", action="store_true", help="Enable PreTrained Head mode (default: False)")
    parser.add_argument("--classWeighted", action="store_true", help="Enable Class Weighted mode (default: False)")
    parser.add_argument("--DataAug", action="store_true", help="Enable Data Augmentation mode (default: False)")
    parser.add_argument("--oversampling", action="store_true", help="Enable OverSampling mode (default: False)")
    parser.add_argument("--label_count", type=int, choices=[25, 20, 15, 10,5], default=25,
                        help="Select Label counts: 25 label, 20 label, 15 label, 10 label, 5 label, (default: 25)")
    args = parser.parse_args()
    
    main(args.subLabel, None if args.agg_mode == "none" else args.agg_mode, args.radDino, args.radDinoHead, args.classWeighted, args.DataAug, args.oversampling, args.label_count)



