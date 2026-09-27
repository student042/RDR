# -*- coding: utf-8 -*-
import copy

import torch


def build_ema_model(model):
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def update_ema_model(student, teacher, decay: float = 0.999):
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()

    for key, teacher_value in teacher_state.items():
        student_value = student_state[key]
        if not torch.is_floating_point(teacher_value):
            teacher_value.copy_(student_value)
        else:
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
