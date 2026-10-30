# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marinskyrl.distillation import TeacherModelSpec
from marinskyrl.hf_model import staged_model_path

from skyrl_train.teacher_runtime import teacher_model_location

IDENTITY = "sha256:" + "a" * 64


def test_an_object_store_teacher_loads_from_its_staged_copy_without_a_hub_revision():
    model = TeacherModelSpec(path="s3://bucket/expert/exports/global_step_240/policy", revision=IDENTITY)
    assert teacher_model_location(model) == (staged_model_path(model.path, IDENTITY), None)


def test_a_hub_teacher_keeps_its_repo_and_revision():
    model = TeacherModelSpec(path="open-athena/expert", revision="b" * 40)
    assert teacher_model_location(model) == ("open-athena/expert", "b" * 40)
