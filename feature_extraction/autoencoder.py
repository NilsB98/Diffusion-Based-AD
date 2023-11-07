from typing import Tuple, Callable

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None, *args, **kwargs):
        """
        Initialize the autoencoder. Will be used to train the feature extractor/encoder.
        Zse the @init_decoder method to automatically create the decoder.

        :param encoder: Encoder module. Expects to return a dictionary containing at least the 'out_layer' key.
        :param decoder: Decoder module.
        :param args: Passed to nn.Module.
        :param kwargs: Passed to nn.Module.
        """
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)['out_layer']
        x = self.decoder(x)
        return x

    def init_decoder(self, input_shape: Tuple[int, int, int], device='cuda'):
        """
        Initialize the decoder based on the given input shape (C,W,H).
        Only supports shapes with equal width and height.

        :param input_shape: Shape of the input for the autoencoder, i.e. the images. (c,h,w)
        :param device: Device to run this on.
        :return: None
        """

        self.encoder = self.encoder.to(device)
        encoder_out_shape = self.encoder(torch.zeros(input_shape).unsqueeze(0).to(device))['out_layer'].shape
        self.decoder = self.create_decoder(encoder_out_shape[1], encoder_out_shape[-1], input_shape[-1])

    @staticmethod
    def create_decoder(in_channels, input_size, output_size) -> nn.Module:
        """
        Utility function to create the decoder based on the image shape. Expects square shape of the images
        (width=height).

        :param input_size: Decoder input size
        :param output_size: Decoder output size
        :return: Decoder module
        """

        assert output_size / input_size % 2 == 0

        num_layers = (output_size // input_size).bit_length() - 1
        decoder = nn.Sequential()
        for layer in range(num_layers):
            num_in_channels = in_channels >> layer
            num_out_channels = num_in_channels // 2 if layer is not num_layers - 1 else 3
            decoder.add_module(f"layer_{layer + 1}",
                               nn.ConvTranspose2d(num_in_channels, num_out_channels, 3, output_padding=1, padding=1,
                                                  stride=2))
            decoder.add_module(f"relu_{layer}", nn.ReLU())

        decoder.add_module("out_layer", nn.Conv2d(3, 3, 3, 1, 1))

        return decoder


class AETrainer:

    def __init__(self, model: nn.Module, train_loader, test_loader, optimizer=None, loss_fn=None, device=None, writer:SummaryWriter=None):
        self.device = device if device is not None else 'cuda:0'
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            model.parameters(),
            weight_decay=1e-6,
            lr=1e-4,
            betas=(0.95, 0.999),
            eps=1e-08,
        )
        self.writer:SummaryWriter = writer

    def train(self, epochs):
        self.model.train()
        self.model = self.model.to(self.device)
        self.writer.add_text('FeatureExtractorTrainer', 'AETrainer')

        for param in self.model.parameters():
            param.requires_grad = True

        for epoch in range(epochs):
            train_loss = 0
            for batch, _ in self.train_loader:
                batch = batch.to(torch.device('cuda:0'))

                reconstructed = self.model(batch)

                loss = self.loss_fn(reconstructed, batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            self.writer.add_scalar('Loss/extractor_train', train_loss, epoch)

            test_loss = 0
            with torch.no_grad():
                self.model.eval()

                for batch, _, _ in self.test_loader:
                    batch = batch.to(torch.device('cuda:0'))

                    reconstructed = self.model(batch)

                    loss = self.loss_fn(reconstructed, batch)
                    test_loss += loss.item()
                self.model.train()
                test_loss /= len(self.test_loader)
                self.writer.add_scalar('Loss/extractor_test', test_loss, epoch)
                print(f"epoch {epoch}: {test_loss=:.5f} {train_loss=:.5f}")


class DBTrainer(AETrainer):
    """
    Diffusion-Based Trainer ot the auto-encoder
    """

    def __init__(self, model: nn.Module, diffusion_generator: Callable[[torch.Tensor], torch.Tensor], train_loader, test_loader, writer=None):
        super().__init__(model, train_loader, test_loader, writer=writer)
        self.diffusion_generator = diffusion_generator

    def train(self, epochs):
        self.model.train()
        self.model = self.model.to(self.device)
        self.writer.add_text('FeatureExtractorTrainer', 'DBTrainer')

        for param in self.model.parameters():
            param.requires_grad = True

        for epoch in range(epochs):
            train_loss = 0
            for batch, _ in self.train_loader:
                batch = batch.to(torch.device('cuda:0'))

                undiffused_imgs = self.diffusion_generator(batch).to(torch.device('cuda:0'))
                reconstructed = self.model(batch)

                loss = self.loss_fn(reconstructed, undiffused_imgs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            self.writer.add_scalar('Loss/extractor_train', train_loss, epoch)

            test_loss = 0
            with torch.no_grad():
                self.model.eval()

                for batch, _, _ in self.test_loader:
                    batch = batch.to(torch.device('cuda:0'))

                    reconstructed = self.model(batch)

                    loss = self.loss_fn(reconstructed, batch)
                    test_loss += loss.item()
                self.model.train()
                test_loss /= len(self.test_loader)
                self.writer.add_scalar('Loss/extractor_test', test_loss, epoch)
                print(f"epoch {epoch}: {test_loss=:.5f} {train_loss=:.5f}")
