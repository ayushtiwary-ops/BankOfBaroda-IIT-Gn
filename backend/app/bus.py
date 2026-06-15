"""Kafka event bus - events keyed by identity_id (executable architecture).

Ingress produces to ``pramaan.events`` keyed by identity_id (so a given
identity's events land on one partition → ordered, and a consumer group of N
scoring pods partitions the load). Scoring pods consume and score.
"""
from __future__ import annotations

import json
import os
import time

TOPIC = os.environ.get("PRAMAAN_EVENTS_TOPIC", "pramaan.events")


def make_producer(brokers: str, retries: int = 40):
    from kafka import KafkaProducer

    last = None
    for _ in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=brokers.split(","),
                value_serializer=lambda v: json.dumps(v).encode(),
                key_serializer=lambda k: str(k).encode(),
                acks="all", retries=5)
        except Exception as exc:  # broker not up yet
            last = exc
            time.sleep(2)
    raise RuntimeError(f"kafka producer unavailable after retries: {last}")


def make_consumer(brokers: str, group: str, retries: int = 40):
    from kafka import KafkaConsumer

    last = None
    for _ in range(retries):
        try:
            return KafkaConsumer(
                TOPIC,
                bootstrap_servers=brokers.split(","),
                group_id=group,
                value_deserializer=lambda v: json.loads(v.decode()),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000)
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise RuntimeError(f"kafka consumer unavailable after retries: {last}")
