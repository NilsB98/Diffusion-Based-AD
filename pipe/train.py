import piqa
import torch
from torch.nn import L1Loss

from noise.simplex import simplexGenerator
from anomalies.generation import ImageCorruptor
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn as nn
import torch.nn.functional as F
from piqa import ssim

def train_step(model, batch, noise_scheduler, lr_scheduler, loss_fn, optimizer, num_train_steps, noise_kind='gaussian'):
    assert noise_kind in ['gaussian', 'simplex']

    model.train()
    clean_imgs = batch.to(model.device)

    noise = None
    if noise_kind == 'gaussian':
        noise = torch.randn(clean_imgs.shape, dtype=clean_imgs.dtype)
    elif noise_kind == 'simplex':
        noise = simplexGenerator.batch_3d_octaves(clean_imgs.shape, 6, 0.6)
    noise = noise.to(clean_imgs.device)

    timesteps = torch.randint(0, num_train_steps, (batch.shape[0],), device=clean_imgs.device).long()
    noisy_images = noise_scheduler.add_noise(clean_imgs, noise, timesteps)

    optimizer.zero_grad()

    prediction = model(noisy_images, timesteps).sample

    loss = loss_fn(prediction, noise)
    loss.backward()

    optimizer.step()
    lr_scheduler.step()

    return loss.item()


def augment_mask(mask: torch.Tensor):
    """
    Augment a binary mask with dilation, erosion or set it to zero with a 25% chance.

    :param mask: mask to be augmented
    :return: augmented mask
    """

    # Select a random operation and apply it
    operation = torch.randint(0, 3, (1,))

    if operation == 0:
        iterations = torch.randint(1, 6, (1,))
        for _ in range(iterations):
            mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    elif operation == 1:
        iterations = torch.randint(1, 6, (1,))
        for _ in range(iterations):
            mask = torch.nn.functional.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    elif operation == 2 and torch.rand(1).item() < 0.25:
        return torch.zeros_like(mask)

    return mask


def calc_x_t_prev(x_t, beta_diff, mask_t, eps_t, normal_t):
    """
    Equation (4) in the transfusion paper.

    :param x_t:
    :param beta_diff:
    :param mask_t:
    :param eps_t:
    :param normal_t:
    :return:
    """

    x_t_prev = x_t - beta_diff * mask_t * eps_t + beta_diff * mask_t * normal_t
    return x_t_prev

def calc_x_t(mask, normal, eps, beta_t):
    """
    Equation (2) in the transfusion paper.

    :param mask:
    :param normal:
    :param eps:
    :param beta_t:
    :return:
    """
    inverse_mask = torch.abs(mask - 1)
    x_t = inverse_mask * normal + beta_t * (mask * eps) + (1 - beta_t) * (mask * normal)
    return x_t

def train_step_transfusion(model, batch, noise_scheduler, lr_scheduler, loss_fn, optimizer, num_train_steps, image_corruptor: ImageCorruptor):

    model.train()
    clean_imgs = batch.to(model.device)

    timesteps = torch.randint(0, num_train_steps, (batch.shape[0],), device=clean_imgs.device).long()
    transparency = torch.ones_like(timesteps) - timesteps / num_train_steps  # aka beta
    corrupted_img, noise, mask = image_corruptor(clean_imgs, transparency)  # TODO returns and run on batches

    mask_augmented = augment_mask(mask)

    optimizer.zero_grad()

    model_input = torch.cat(mask_augmented, corrupted_img)
    model_output = model(model_input, timesteps)

    #                                            1D                3D                  3D
    mask_t_pred, noise_t_pred, normal_t_pred = model_output[0], model_output[1:4], model_output[4:]

    # loss is FusionLoss
    # x = clean_imgs; n_t = normal_t_pred; eps_t = noise_t_pred; eps = noise; M_t = mask_t_pred; M = mask; M_alpha = mask_augmented
    # x_t_pred = calculated from estimated variables
    # x_t_gt = calculated from gt variables
    x_t_prev_pred = calc_x_t_prev(corrupted_img, 1/num_train_steps, mask_t_pred, noise_t_pred, normal_t_pred)
    x_t_prev_gt = calc_x_t_prev(corrupted_img, 1/num_train_steps, mask, noise, clean_imgs)
    loss = loss_fn(normal_t_pred, clean_imgs, mask_t_pred, mask, noise_t_pred, noise, x_t_prev_pred, x_t_prev_gt)

    loss.backward()
    optimizer.step()

    # TODO check missing steps?


class FusionLoss(nn.Module):
    def __init__(self, focal_alpha=5, focal_gamma=2, log_loss=False):
        super(FusionLoss, self).__init__()
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, log_loss)
        self.l1_smooth = torch.nn.SmoothL1Loss()
        self.l2 = torch.nn.MSELoss()
        self.ssim = piqa.SSIM().cuda()


    def _loss_normal_appearance(self, normal_pred, normal_gt):
        # ssim -> https://github.com/francois-rozet/piqa
        return self.ssim(normal_pred, normal_gt) + torch.nn.functional.l1_loss(normal_pred, normal_gt)

    def _loss_anomaly_mask(self, mask_pred, mask_gt):
        return self.focal_loss(mask_pred, mask_gt) + self.l1_smooth(mask_pred)

    def _loss_anomaly_appearance(self, eps_pred, eps_gt):
        return self.l2(eps_pred, eps_gt)

    def _loss_consistency(self, x_prev_pred, x_prev_gt):
        return self.l2(x_prev_pred, x_prev_gt)

    def forward(self, normal_pred, normal_gt, mask_pred, mask_gt, eps_pred, eps_gt, x_prev_pred, x_prev_gt):
        return (self._loss_normal_appearance(normal_pred, normal_gt) + self._loss_anomaly_mask(mask_pred, mask_gt) +
                self._loss_anomaly_appearance(eps_pred, eps_gt) + self._loss_consistency(x_prev_pred, x_prev_gt))


class FocalLoss(nn.Module):
    def __init__(self, alpha=5, gamma=2, log_loss=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.log_loss = log_loss

    def forward(self, inputs, targets):
        if self.log_loss:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)

        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()