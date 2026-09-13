"""Durable migration state and Fargate ownership tests."""

import hashlib
import io
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")

from botocore.exceptions import ClientError

import geotessera._migration_state as state_module
from geotessera._migration_state import BusyError, State


class S3:
    def __init__(self):
        self.objects = {}
        self.denied = False
        self.conditions = []

    @staticmethod
    def error(code):
        raise ClientError({"Error": {"Code": code}}, "state")

    def get_object(self, Bucket, Key):
        if self.denied:
            self.error("AccessDenied")
        if Key not in self.objects:
            self.error("NoSuchKey")
        data, token = self.objects[Key]
        return {"Body": io.BytesIO(data), "ETag": token}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.conditions.append(kwargs)
        assert kwargs["ChecksumAlgorithm"] == "SHA256"
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            self.error("PreconditionFailed")
        if (
            kwargs.get("IfMatch") is not None
            and self.objects.get(Key, (None, None))[1] != kwargs["IfMatch"]
        ):
            self.error("PreconditionFailed")
        token = '"' + hashlib.sha256(Body).hexdigest() + '"'
        self.objects[Key] = Body, token

    def delete_object(self, Bucket, Key, **kwargs):
        self.error("AccessDenied")


def test_s3_conditional_ownership_and_access_errors(monkeypatch):
    s3 = S3()
    monkeypatch.setattr(state_module, "s3_client", lambda options: s3)
    state = State("s3://control/migration")
    with state.owner("utm31"), pytest.raises(BusyError), state.owner("utm31"):
        pass
    owner = json.loads(s3.objects["migration/owners/utm31.json"][0])
    assert owner["released"] is True
    with state.owner("utm31"):
        pass
    assert s3.conditions[0]["IfNoneMatch"] == "*"
    state.write("a.json", {"a": 1})
    _, token = state.get("a.json")
    state.write("a.json", {"a": 2}, match=token)
    s3.denied = True
    with pytest.raises(ClientError, match="AccessDenied"):
        state.get("missing.json")


def test_old_fargate_attempt_must_be_stopped(monkeypatch):
    import boto3

    status = ["RUNNING"]
    calls = []

    def client(service, **kwargs):
        def describe_tasks(**kwargs):
            calls.append(kwargs)
            return {"tasks": [{"lastStatus": status[0]}]}

        return SimpleNamespace(describe_tasks=describe_tasks)

    monkeypatch.setattr(boto3, "client", client)
    owner = {
        "task_arn": "arn:aws:ecs:us-west-2:123456789012:task/cluster/task",
        "cluster": "cluster",
        "job_id": "job",
    }
    assert not state_module.owner_stopped(owner)
    status[0] = "STOPPED"
    assert state_module.owner_stopped(owner)
    assert len(calls) == 2
