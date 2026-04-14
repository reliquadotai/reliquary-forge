from __future__ import annotations

from typing import Any


class LocalChainAdapter:
    def __init__(self) -> None:
        self._weights: dict[str, float] = {}
        self._events: list[dict[str, Any]] = []

    def _record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "mode": "local",
            "event_type": event_type,
            "payload": payload,
            "event_index": len(self._events),
        }
        self._events.append(event)
        return event

    def publish_policy_reference(self, policy_artifact: dict[str, Any]) -> dict[str, Any]:
        return self._record(
            "policy_reference",
            {
                "policy_id": policy_artifact["artifact_id"],
                "window_id": policy_artifact["window_id"],
                "model_ref": policy_artifact["payload"]["model_ref"],
            },
        )

    def publish_window_metadata(
        self,
        *,
        run_id: str,
        window_id: int,
        policy_id: str,
        challenge_ids: list[str],
        miner_ids: list[str],
    ) -> dict[str, Any]:
        return self._record(
            "window_metadata",
            {
                "run_id": run_id,
                "window_id": window_id,
                "policy_id": policy_id,
                "challenge_ids": challenge_ids,
                "miner_ids": miner_ids,
            },
        )

    def submit_miner_result_reference(self, rollout_artifact: dict[str, Any]) -> dict[str, Any]:
        payload = rollout_artifact["payload"]
        return self._record(
            "miner_result_reference",
            {
                "rollout_id": rollout_artifact["artifact_id"],
                "window_id": rollout_artifact["window_id"],
                "miner_id": rollout_artifact["producer_id"],
                "challenge_id": payload.get("challenge_id") or payload.get("task_id"),
                "artifact_type": rollout_artifact["artifact_type"],
            },
        )

    def publish_scorecard(self, scorecard_artifact: dict[str, Any]) -> dict[str, Any]:
        self._weights = dict(scorecard_artifact["payload"]["weights"])
        return self._record(
            "scorecard",
            {
                "scorecard_id": scorecard_artifact["artifact_id"],
                "window_id": scorecard_artifact["window_id"],
                "weights": self._weights,
                "totals": scorecard_artifact["payload"]["totals"],
            },
        )

    def publish_checkpoint_reference(
        self,
        checkpoint_artifact: dict[str, Any],
        next_policy_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        return self._record(
            "checkpoint_reference",
            {
                "checkpoint_id": checkpoint_artifact["artifact_id"],
                "checkpoint_path": checkpoint_artifact["payload"]["checkpoint_path"],
                "next_policy_id": next_policy_artifact["artifact_id"],
                "window_id": checkpoint_artifact["window_id"],
            },
        )

    def publish_weights(self, weights: dict[str, float]) -> dict[str, Any]:
        self._weights = dict(weights)
        return self._record("weights", {"weights": self._weights})

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


class BittensorChainAdapter:
    def __init__(self, network: str, netuid: int, wallet_name: str, hotkey_name: str):
        import bittensor as bt
        from bittensor_wallet import Wallet

        self.bt = bt
        self.network = network
        self.netuid = netuid
        self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
        self.subtensor = bt.Subtensor(network=network)

    def publish_policy_reference(self, policy_artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": "bittensor",
            "network": self.network,
            "netuid": self.netuid,
            "event_type": "policy_reference",
            "policy_id": policy_artifact["artifact_id"],
        }

    def publish_window_metadata(
        self,
        *,
        run_id: str,
        window_id: int,
        policy_id: str,
        challenge_ids: list[str],
        miner_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "mode": "bittensor",
            "network": self.network,
            "netuid": self.netuid,
            "event_type": "window_metadata",
            "run_id": run_id,
            "window_id": window_id,
            "policy_id": policy_id,
            "challenge_ids": challenge_ids,
            "miner_ids": miner_ids,
        }

    def submit_miner_result_reference(self, rollout_artifact: dict[str, Any]) -> dict[str, Any]:
        payload = rollout_artifact["payload"]
        return {
            "mode": "bittensor",
            "network": self.network,
            "netuid": self.netuid,
            "event_type": "miner_result_reference",
            "rollout_id": rollout_artifact["artifact_id"],
            "miner_id": rollout_artifact["producer_id"],
            "window_id": rollout_artifact["window_id"],
            "challenge_id": payload.get("challenge_id") or payload.get("task_id"),
            "artifact_type": rollout_artifact["artifact_type"],
        }

    def publish_scorecard(self, scorecard_artifact: dict[str, Any]) -> dict[str, Any]:
        return self.publish_weights(scorecard_artifact["payload"]["weights"])

    def publish_checkpoint_reference(
        self,
        checkpoint_artifact: dict[str, Any],
        next_policy_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "mode": "bittensor",
            "network": self.network,
            "netuid": self.netuid,
            "event_type": "checkpoint_reference",
            "checkpoint_id": checkpoint_artifact["artifact_id"],
            "next_policy_id": next_policy_artifact["artifact_id"],
        }

    def publish_weights(self, uid_weights: dict[int, float] | dict[str, float]) -> dict[str, Any]:
        uids = list(uid_weights)
        weights = [uid_weights[uid] for uid in uids]
        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        return {
            "mode": "bittensor",
            "network": self.network,
            "netuid": self.netuid,
            "success": bool(success),
            "uids": uids,
            "weights": weights,
        }
