"""Teacher-student distillation wrapper for DEIM."""

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..core import register


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


@register()
class DEIMFGDDistiller(nn.Module):
    """Wrap a DEIM student model with a frozen teacher to expose distillation features."""

    __inject__ = ['student', 'teacher']

    def __init__(self,
                 student: nn.Module,
                 teacher: nn.Module,
                 feature_pairs: Optional[Sequence[Dict[str, Any]]] = None,
                 teacher_ckpt: Optional[str] = None,
                 student_ckpt: Optional[str] = None,
                 distill_schedule: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_ckpt = teacher_ckpt
        self.student_ckpt = student_ckpt
        self.feature_pairs = self._format_pairs(feature_pairs)
        self.distill_schedule = distill_schedule or {}
        self._current_epoch = 0
        self._deploy_mode = False

        _freeze_module(self.teacher)
        self.teacher.eval()
        self._apply_schedule_to_pairs()

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch so we can honour warm-up schedules."""
        self._current_epoch = int(epoch)

    def _apply_schedule_to_pairs(self) -> None:
        if not self.feature_pairs:
            return
        for pair in self.feature_pairs:
            meta = dict(pair.get('meta', {}) or {})
            schedule_entry = None
            if self.distill_schedule:
                schedule_entry = self.distill_schedule.get(pair['name'])
            if isinstance(schedule_entry, dict):
                if 'start_epoch' in schedule_entry and 'start_epoch' not in meta:
                    meta['start_epoch'] = schedule_entry['start_epoch']
            pair['meta'] = meta

    def _pair_start_epoch(self, pair: Dict[str, Any]) -> int:
        meta = pair.get('meta') or {}
        value = meta.get('start_epoch', 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def forward(self,
                images: torch.Tensor,
                targets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if self._deploy_mode:
            return self.student(images, targets=targets)

        active_pairs = [pair for pair in self.feature_pairs
                        if self._current_epoch >= self._pair_start_epoch(pair)]

        teacher_backbone_feats: Sequence[torch.Tensor] = ()
        teacher_encoded_feats: Sequence[torch.Tensor] = ()

        need_teacher = bool(active_pairs) and self._teacher_enabled()
        if need_teacher:
            with torch.no_grad():
                teacher_backbone_feats = self.teacher.backbone(images)
                if any((pair.get('meta') or {}).get('source', 'backbone') == 'encoder'
                       for pair in active_pairs):
                    teacher_encoded_feats = self.teacher.encoder(teacher_backbone_feats)

        student_backbone_feats = self.student.backbone(images)
        student_encoded_feats = self.student.encoder(student_backbone_feats)
        student_outputs = self.student.decoder(student_encoded_feats, targets)

        teacher_backbone_tuple = tuple(teacher_backbone_feats)
        teacher_encoded_tuple = tuple(teacher_encoded_feats)
        student_backbone_tuple = tuple(student_backbone_feats)
        student_encoded_tuple = tuple(student_encoded_feats)

        distill_pairs: List[Dict[str, Any]] = []
        if need_teacher and (teacher_backbone_tuple or teacher_encoded_tuple):
            for pair in active_pairs:
                meta = pair.get('meta', {})
                source = meta.get('source', 'backbone')
                if source == 'encoder':
                    if not teacher_encoded_tuple:
                        raise RuntimeError('Encoder features requested but teacher encoder not computed.')
                    student_feat = student_encoded_tuple[pair['student_index']]
                    teacher_feat = teacher_encoded_tuple[pair['teacher_index']]
                else:
                    student_feat = student_backbone_tuple[pair['student_index']]
                    teacher_feat = teacher_backbone_tuple[pair['teacher_index']]
                distill_pairs.append({
                    'name': pair['name'],
                    'student_feat': student_feat,
                    'teacher_feat': teacher_feat,
                    'meta': meta
                })

        student_outputs['distill_pairs'] = distill_pairs
        student_outputs['backbone_feats'] = student_backbone_tuple
        student_outputs['encoded_feats'] = student_encoded_tuple

        teacher_outputs: Dict[str, Any] = {'active_pairs': [p['name'] for p in active_pairs]}
        if teacher_backbone_tuple:
            teacher_outputs['backbone_feats'] = teacher_backbone_tuple
        if teacher_encoded_tuple:
            teacher_outputs['encoded_feats'] = teacher_encoded_tuple
        student_outputs['teacher_outputs'] = teacher_outputs
        return student_outputs

    def _teacher_enabled(self) -> bool:
        if self._deploy_mode:
            return False
        if not self.feature_pairs:
            return False
        for pair in self.feature_pairs:
            if self._current_epoch >= self._pair_start_epoch(pair):
                return True
        return False

    def load_teacher_checkpoint(self,
                                checkpoint_path: str,
                                strict: bool = True) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        raw_state = checkpoint.get('model', checkpoint)

        # Extract teacher submodule weights if checkpoint contains the whole distiller
        cleaned_state = {}
        has_teacher_prefix = False
        for k, v in raw_state.items():
            key = k
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('teacher.'):
                has_teacher_prefix = True
                key = key[len('teacher.'):]  # strip teacher prefix
                cleaned_state[key] = v
        # If no explicit teacher prefix is found, assume raw_state is already teacher-only
        state_dict = cleaned_state if has_teacher_prefix else raw_state

        decoder = getattr(self.teacher, 'decoder', None)
        target_anchors = state_dict.get('decoder.anchors')
        if decoder is not None and target_anchors is not None and hasattr(decoder, 'anchors'):
            current_anchors = getattr(decoder, 'anchors', None)
            if current_anchors is None or current_anchors.shape != target_anchors.shape:
                num_total = target_anchors.shape[1]
                strides = getattr(decoder, 'feat_strides', None)

                def _infer_eval_size(total_anchors, feat_strides):
                    if not feat_strides:
                        return None
                    max_stride = max(feat_strides)
                    step = min(feat_strides)
                    for size in range(max_stride, 4097, step):
                        if any(size % s != 0 for s in feat_strides):
                            continue
                        calc = sum((size // s) ** 2 for s in feat_strides)
                        if calc == total_anchors:
                            return [size, size]
                    return None

                inferred_size = _infer_eval_size(num_total, strides)
                if inferred_size is not None:
                    decoder.eval_spatial_size = inferred_size
                    anchors, valid_mask = decoder._generate_anchors()
                    decoder._buffers['anchors'] = anchors
                    decoder._buffers['valid_mask'] = valid_mask

        # When strict is False, filter out any shape-mismatched params to avoid RuntimeError
        if not strict:
            current = self.teacher.state_dict()
            filtered = {}
            mismatched = []
            for k, v in state_dict.items():
                if k in current and hasattr(current[k], 'shape') and hasattr(v, 'shape'):
                    if current[k].shape == v.shape:
                        filtered[k] = v
                    else:
                        mismatched.append(k)
            state_to_load = filtered
        else:
            state_to_load = state_dict

        missing_keys, unexpected_keys = self.teacher.load_state_dict(state_to_load, strict=False if not strict else True)
        if missing_keys:
            print(f"[DEIMFGDDistiller] Missing keys when loading teacher: {missing_keys}")
        if unexpected_keys:
            print(f"[DEIMFGDDistiller] Unexpected keys when loading teacher: {unexpected_keys}")
        if not strict:
            print(f"[DEIMFGDDistiller] (Teacher) Loaded with filtered params: {len(state_to_load)} keys.")

    def load_student_checkpoint(self,
                                 checkpoint_path: str,
                                 strict: bool = True) -> None:   #注意
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        source = 'raw'
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get('ema'), dict) and 'module' in checkpoint['ema']:
                state_dict = checkpoint['ema']['module']
                source = 'ema.module'
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                source = 'model'
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # remove a leading 'module.' if present (from DDP/EMA saves)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if key.startswith('student.'):
                key = key[len('student.'):]
            cleaned_state_dict[key] = v

        # Avoid size mismatch errors when strict=False by filtering matched shapes only
        if not strict:
            current = self.student.state_dict()
            filtered = {}
            mismatched = []
            for k, v in cleaned_state_dict.items():
                if k in current and hasattr(current[k], 'shape') and hasattr(v, 'shape'):
                    if current[k].shape == v.shape:
                        filtered[k] = v
                    else:
                        mismatched.append(k)
            state_to_load = filtered
        else:
            state_to_load = cleaned_state_dict

        missing_keys, unexpected_keys = self.student.load_state_dict(state_to_load, strict=False if not strict else True)
        print(f"[DEIMFGDDistiller] Loaded student from {checkpoint_path} using `{source}` weights.")
        if missing_keys:
            print(f"[DEIMFGDDistiller] Missing keys when loading student: {missing_keys}")
        if unexpected_keys:
            print(f"[DEIMFGDDistiller] Unexpected keys when loading student: {unexpected_keys}")
        if not strict:
            print(f"[DEIMFGDDistiller] (Student) Loaded with filtered params: {len(state_to_load)} keys. Unmatched: {len(mismatched)}")

    @staticmethod
    def _format_pairs(feature_pairs: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        if feature_pairs is None:
            return formatted

        for idx, pair in enumerate(feature_pairs):
            name = pair.get('name', f'pair_{idx}')
            if 'student_index' not in pair or 'teacher_index' not in pair:
                raise ValueError('feature_pairs entries must include student_index and teacher_index')
            formatted.append({
                'name': name,
                'student_index': int(pair['student_index']),
                'teacher_index': int(pair['teacher_index']),
                'meta': pair.get('meta', {})
            })
        return formatted

    def set_teacher_eval(self) -> None:
        self.teacher.eval()

    def deploy(self) -> 'DEIMFGDDistiller':
        """Switch to deploy mode for profiling or inference-only usage.
        In deploy mode, the distiller forwards only the student to avoid double-counting FLOPs.
        """
        self._deploy_mode = True
        if hasattr(self.student, 'deploy'):
            self.student.deploy()
        # Keep teacher in eval; it won't be used in deploy forward
        self.teacher.eval()
        return self

    def train(self, mode: bool = True) -> 'DEIMFGDDistiller':
        super().train(mode)
        # keep teacher frozen in eval mode even during student training
        self.teacher.eval()
        return self


