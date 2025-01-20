from collections import OrderedDict
import os
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import *
from fvcore.nn import FlopCountAnalysis
import numpy as np
from sklearn.model_selection import KFold
import torch

from unetr_pp.network_architecture.toothfairy2.unetr_pp_synapse import UNETR_PP
from unetr_pp.training.network_training.unetr_pp_trainer_synapse import unetr_pp_trainer_synapse
from unetr_pp.utilities.nd_softmax import softmax_helper
from unetr_pp.utilities.tensor_utilities import sum_tensor


class unetr_pp_trainer_toothfairy2(unetr_pp_trainer_synapse):

    def __init__(self, plans_file, *args, **kwargs):
        
        plans_file = Path(os.environ['nnFormer_preprocessed']) / (
            'Task112_ToothFairy2/nnFormerPlansv2.1_plans_3D.pkl'
        )
        super().__init__(plans_file, *args, **kwargs)

        self.crop_size = [80, 160, 160]

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """

        self.network = UNETR_PP(in_channels=self.input_channels,
                             out_channels=self.num_classes,
                             img_size=self.crop_size,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32, 64, 128, 256],
                             do_ds=True,
                             )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        # Print the network parameters & Flops
        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        input_res = (1, 80, 160, 160)
        input = torch.ones(()).new_empty((1, *input_res), dtype=next(self.network.parameters()).dtype,
                                         device=next(self.network.parameters()).device)
        flops = FlopCountAnalysis(self.network, input)
        model_flops = flops.total()
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
        
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def run_online_evaluation(self, output, target):
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
            
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
            # tp_hard = torch.zeros((target.shape[0], 8)).to(output_seg.device.index)
            # fp_hard = torch.zeros((target.shape[0], 8)).to(output_seg.device.index)
            # fn_hard = torch.zeros((target.shape[0], 8)).to(output_seg.device.index)
            i=0
            for c in range(1, num_classes):
                tp_hard[:, i] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, i] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, i] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)
                i+=1

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    # def setup_DA_params(self):
    #     super().setup_DA_params()

    #     self.data_aug_params['do_mirror'] = True
    #     self.data_aug_params['mirror_axes'] = (0, 1)
