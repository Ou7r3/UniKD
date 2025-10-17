"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)

    # Provide separate counts for student / teacher / total so users can see
    # the real distillation footprint instead of just the deployable student.
    student_module = getattr(model_for_info, 'student', None)
    teacher_module = getattr(model_for_info, 'teacher', None)

    student_params = sum(p.numel() for p in student_module.parameters()) if student_module else 0
    teacher_params = sum(p.numel() for p in teacher_module.parameters()) if teacher_module else 0
    total_params = sum(p.numel() for p in model_for_info.parameters())

    del model_for_info

    summary = ["Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, total_params)]
    if student_module is not None:
        summary.append(f"Student Params:{student_params}")
    if teacher_module is not None:
        summary.append(f"Teacher Params:{teacher_params}")

    return total_params, summary
