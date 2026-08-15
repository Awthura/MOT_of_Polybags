"""
MQTT publisher — OVGU AMS algorithm phase.

The streamer's one outbound connection to the broker. Thin wrapper over paho so
the rest of the code publishes a Python dict and never touches the client. QoS 0
(fire-and-forget) is right here: positions are a live stream — a dropped frame is
simply superseded by the next one a fifth of a second later, and redelivery would
only ever hand the dashboard a stale position.
"""

from __future__ import annotations

import json

import paho.mqtt.client as mqtt


class Publisher:
    def __init__(self, host: str = "localhost", port: int = 1883,
                 topic: str = "/polybags", client_id: str = "ams-streamer"):
        self.topic = topic
        self._c = mqtt.Client(client_id=client_id,
                              callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self._connected = False
        self._host, self._port = host, port

    def connect(self) -> None:
        # connect_async + loop_start so a broker that is not up yet does not
        # block the streamer from starting; paho reconnects on its own.
        self._c.connect_async(self._host, self._port, keepalive=30)
        self._c.loop_start()

    def publish(self, payload: dict) -> None:
        self._c.publish(self.topic, json.dumps(payload), qos=0)

    def close(self) -> None:
        self._c.loop_stop()
        try:
            self._c.disconnect()
        except Exception:  # noqa: BLE001 — best-effort on shutdown
            pass
