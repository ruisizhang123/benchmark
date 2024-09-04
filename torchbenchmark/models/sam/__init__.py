# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from ...util.model import BenchmarkModel
from .build_sam import sam_model_registry
from .predictor import SamPredictor
import numpy as np
import cv2
from torchbenchmark.tasks import COMPUTER_VISION
import torch
import os
from torch import optim


class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"
    def __init__(self, test, device, batch_size=1, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        print("start to work with sam")
        # Checkpoint options are here https://github.com/facebookresearch/segment-anything#model-checkpoints
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
        sam_checkpoint = os.path.join(data_folder, "sam_vit_h_4b8939.pth")
        print("sam_checkpoint", sam_checkpoint)
        if not os.path.exists(sam_checkpoint):
            from torchbenchmark.util.framework.fb.installer import install_model_weights
            sam_checkpoint = install_model_weights(self.name)
        model_type = "vit_h"

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device=device)
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9
        )

        print("self.batch_size", self.batch_size)
        self.sample_image = torch.rand(
            (self.batch_size, 3, 256, 256), dtype=torch.bfloat16
        ).to(self.device)

        self.sample_masks = torch.randint(
                0, 1, (self.batch_size, 1, 256, 256), dtype=torch.bfloat16
            ).to(self.device)

        self.sample_masks_base = torch.randint(
                0, 1, (self.batch_size, 1, 256, 256), dtype=torch.bfloat16
            ).to(self.device)

        self.example_input = [
            {
                "image": x,
                "original_size": (256, 256),
                "mask_inputs": y
            } for x, y in zip(self.sample_image, self.sample_masks)
        ]

        self.multimask_output = False

        print("end work with sam")
    def get_module(self):
        return self.model, (self.example_input, self.multimask_output)

    def train(self):
        self.model.train()
        
        total_loss = 0
        self.optimizer.zero_grad()
        preidcts = self.model(self.example_input, self.multimask_output)
        loss = self.loss_fn(preidcts, self.sample_masks_base)
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

        # Return the average loss
        return total_loss

    def eval(self):
        # To test for bfloat16 uncomment the below line
        # predictor = SamPredictor(self.model.to(dtype=torch.bfloat16))

        predictor = SamPredictor(self.model)

        predictor.set_image(self.image)

        input_point = np.array([[500, 375]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )
        return (masks,)
