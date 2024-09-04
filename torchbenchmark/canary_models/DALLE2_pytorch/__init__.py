import torch

from dalle2_pytorch import DALLE2, Unet, Decoder, DiffusionPriorNetwork, DiffusionPrior, OpenAIClipAdapter

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION


class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 1
    CANNOT_SET_CUSTOM_OPTIMIZER = True

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        print("start to work with new dalle2")
        if self.device == "cpu":
            raise NotImplementedError("DALL-E 2 Not Supported on CPU")
    
        self.clip = OpenAIClipAdapter().to(self.device)
        self.sample_text = self.example_input = torch.randint(0, 49408, (self.batch_size, 256)).to(self.device)
        self.sample_images = torch.randn(self.batch_size, 3, 256, 256).to(torch.bfloat16).to(self.device)
 
        prior_network = DiffusionPriorNetwork(
            dim = 512,
            depth = 6,
            dim_head = 64,
            heads = 8
        ).to(self.device)

        self.diffusion_prior = DiffusionPrior(
            net = prior_network,
            clip = self.clip,
            timesteps = 100,
            cond_drop_prob = 0.2
        ).to(self.device)

        unet1 = Unet(
            dim = 128,
            image_embed_dim = 512,
            cond_dim = 128,
            channels = 3,
            dim_mults=(1, 2, 4, 8),
            text_embed_dim = 512,
            cond_on_text_encodings = True  # set to True for any unets that need to be conditioned on text encodings (ex. first unet in cascade)
        ).to(self.device)

        unet2 = Unet(
            dim = 16,
            image_embed_dim = 512,
            cond_dim = 128,
            channels = 3,
            dim_mults = (1, 2, 4, 8, 16)
        ).to(self.device)

        self.decoder = Decoder(
            unet = (unet1, unet2),
            image_sizes = (128, 256),
            clip = self.clip,
            timesteps = 1,
            sample_timesteps = (1, 1),
            image_cond_drop_prob = 0.1,
            text_cond_drop_prob = 0.5
        ).to(self.device)

        self.model = DALLE2(prior=self.diffusion_prior, decoder=self.decoder).to(self.device)
        self.model.to(torch.bfloat16)
        if test == "train":
            self.model.prior.train()
            self.model.decoder.train()
        #elif test == "eval":
        #    self.model.prior.eval()
        #    self.model.decoder.eval()

        self.optimizer1 = torch.optim.AdamW(
            list(self.model.prior.parameters()),
            lr=5.0e-4,
            weight_decay=1.0e-4,
            eps=1.0e-6,
        )

        self.optimizer2 = torch.optim.AdamW(
            list(self.model.decoder.parameters()),
            lr=5.0e-4,
            weight_decay=1.0e-4,
            eps=1.0e-6,
        )
    def get_module(self):
        return self.model, (self.example_input,)

    def set_module(self, new_model):
        self.model = new_model

    def eval(self):
        model, inputs = self.get_module()
        images = model(*inputs)
        return (images,)

    def train(self):
        # prior networks (with transformer)
        loss =  self.model(self.sample_text, self.sample_images)
        loss.backward()
        return loss.item()
