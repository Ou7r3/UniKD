"""FGD feature distillation loss implemented in pure PyTorch."""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


@register()
class FGDFeatureLoss(nn.Module):
    def __init__(self,
                 student_channels: int,
                 teacher_channels: int,
                 temp: float = 0.5,
                 alpha_fgd: float = 0.001,
                 beta_fgd: float = 0.0005,
                 gamma_fgd: float = 0.001,
                 lambda_fgd: float = 0.000005) -> None:
        super().__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        else:
            self.align = nn.Identity()

        channels = teacher_channels
        self.conv_mask_s = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(channels, 1, kernel_size=1)
        mid_channels = max(channels // 2, 1)
        norm_shape = (mid_channels, 1, 1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.LayerNorm(norm_shape),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1)
        )
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.LayerNorm(norm_shape),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1)
        )

        self._reset_parameters()

    def forward(self,
                student_feat: torch.Tensor,
                teacher_feat: torch.Tensor,
                gt_bboxes: Sequence[torch.Tensor],
                img_metas: Sequence[dict]) -> torch.Tensor:
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            raise ValueError('Student and teacher features must share spatial size for FGD loss.')

        if not isinstance(self.align, nn.Identity):
            student_feat = self.align(student_feat)

        n, c, h, w = student_feat.shape
        s_attention_t, c_attention_t = self._get_attention(teacher_feat)
        s_attention_s, c_attention_s = self._get_attention(student_feat)

        mask_fg = torch.zeros((n, h, w), device=student_feat.device, dtype=student_feat.dtype)
        mask_bg = torch.ones_like(mask_fg)

        for idx in range(n):
            boxes = gt_bboxes[idx]
            if boxes.numel() == 0:
                continue
            img_h, img_w = img_metas[idx]['img_shape'][:2]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / img_w * w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / img_h * h
            xmin = boxes[:, 0].floor().clamp(0, w - 1).long()
            xmax = boxes[:, 2].ceil().clamp(0, w - 1).long()
            ymin = boxes[:, 1].floor().clamp(0, h - 1).long()
            ymax = boxes[:, 3].ceil().clamp(0, h - 1).long()

            area = 1.0 / ((ymax - ymin + 1).clamp(min=1).float() * (xmax - xmin + 1).clamp(min=1).float())
            for box_id in range(boxes.size(0)):
                y0, y1 = ymin[box_id], ymax[box_id]
                x0, x1 = xmin[box_id], xmax[box_id]
                current = mask_fg[idx, y0:y1 + 1, x0:x1 + 1]
                mask_fg[idx, y0:y1 + 1, x0:x1 + 1] = torch.maximum(current, area[box_id])

            mask_bg[idx] = torch.where(mask_fg[idx] > 0, torch.zeros_like(mask_bg[idx]), mask_bg[idx])
            bg_sum = mask_bg[idx].sum()
            if bg_sum > 0:
                mask_bg[idx] = mask_bg[idx] / bg_sum

        fg_loss, bg_loss = self._feature_loss(student_feat, teacher_feat,
                                              mask_fg, mask_bg,
                                              c_attention_s, c_attention_t,
                                              s_attention_s, s_attention_t)
        mask_loss = self._mask_loss(c_attention_s, c_attention_t, s_attention_s, s_attention_t)
        relation_loss = self._relation_loss(student_feat, teacher_feat)

        loss = (self.alpha_fgd * fg_loss +
                self.beta_fgd * bg_loss +
                self.gamma_fgd * mask_loss +
                self.lambda_fgd * relation_loss)
        return loss

    def _get_attention(self, features: torch.Tensor):
        n, c, h, w = features.shape
        value = features.abs()
        spatial_map = value.mean(dim=1)
        spatial_attention = (h * w * F.softmax((spatial_map / self.temp).view(n, -1), dim=1))
        spatial_attention = spatial_attention.view(n, h, w)

        channel_map = value.mean(dim=(2, 3))
        channel_attention = c * F.softmax(channel_map / self.temp, dim=1)
        return spatial_attention, channel_attention

    def _feature_loss(self,
                      student_feat: torch.Tensor,
                      teacher_feat: torch.Tensor,
                      mask_fg: torch.Tensor,
                      mask_bg: torch.Tensor,
                      c_s: torch.Tensor,
                      c_t: torch.Tensor,
                      s_s: torch.Tensor,
                      s_t: torch.Tensor):
        loss_mse = nn.MSELoss(reduction='sum')

        mask_fg = mask_fg.unsqueeze(1)
        mask_bg = mask_bg.unsqueeze(1)

        c_t = c_t.unsqueeze(-1).unsqueeze(-1)
        s_t = s_t.unsqueeze(1)

        teacher_weighted = teacher_feat * torch.sqrt(s_t) * torch.sqrt(c_t)
        student_weighted = student_feat * torch.sqrt(s_t) * torch.sqrt(c_t)

        fg_t = teacher_weighted * torch.sqrt(mask_fg)
        bg_t = teacher_weighted * torch.sqrt(mask_bg)
        fg_s = student_weighted * torch.sqrt(mask_fg)
        bg_s = student_weighted * torch.sqrt(mask_bg)

        fg_loss = loss_mse(fg_s, fg_t) / mask_fg.size(0)
        bg_loss = loss_mse(bg_s, bg_t) / mask_bg.size(0)
        return fg_loss, bg_loss

    def _mask_loss(self,
                   c_s: torch.Tensor,
                   c_t: torch.Tensor,
                   s_s: torch.Tensor,
                   s_t: torch.Tensor) -> torch.Tensor:
        channel_loss = torch.abs(c_s - c_t).sum() / c_s.size(0)
        spatial_loss = torch.abs(s_s - s_t).sum() / s_s.size(0)
        return channel_loss + spatial_loss

    def _relation_loss(self,
                       student_feat: torch.Tensor,
                       teacher_feat: torch.Tensor) -> torch.Tensor:
        context_s = self._spatial_pool(student_feat, self.conv_mask_s, 0)
        context_t = self._spatial_pool(teacher_feat, self.conv_mask_t, 1)

        out_s = student_feat + self.channel_add_conv_s(context_s)
        out_t = teacher_feat + self.channel_add_conv_t(context_t)

        loss_mse = nn.MSELoss(reduction='sum')
        rela_loss = loss_mse(out_s, out_t) / out_s.size(0)
        return rela_loss

    def _spatial_pool(self,
                      features: torch.Tensor,
                      conv_mask: nn.Module,
                      _: int) -> torch.Tensor:
        n, c, h, w = features.size()
        input_x = features.view(n, c, h * w).unsqueeze(1)
        context_mask = conv_mask(features).view(n, 1, h * w)
        context_mask = F.softmax(context_mask, dim=2).unsqueeze(-1)
        context = torch.matmul(input_x, context_mask).view(n, c, 1, 1)
        return context

    def _reset_parameters(self) -> None:
        if not isinstance(self.align, nn.Identity):
            nn.init.kaiming_normal_(self.align.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv_mask_s.weight, mode='fan_in')
        nn.init.constant_(self.conv_mask_s.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_mask_t.weight, mode='fan_in')
        nn.init.constant_(self.conv_mask_t.bias, 0.0)
        self._last_zero_init(self.channel_add_conv_s)
        self._last_zero_init(self.channel_add_conv_t)

    @staticmethod
    def _last_zero_init(module: nn.Sequential) -> None:
        nn.init.constant_(module[-1].weight, 0.0)
        if module[-1].bias is not None:
            nn.init.constant_(module[-1].bias, 0.0)
