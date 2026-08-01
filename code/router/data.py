"""Dataset loading and indexed lookups.

Everything the router knows about a user, group, business, or past message is
served from here. Loaded once, indexed by key, and passed around read-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CSV_FILES = [
    "messages",
    "sample_messages",
    "users",
    "groups",
    "group_members",
    "business_accounts",
    "user_business_history",
    "message_history",
    "message_events",
    "images",
    "voice_notes",
    "daily_notification_summary",
]


def _clean(value: Any) -> Any:
    """pandas hands back NaN for blank CSV cells; callers want None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _row_dict(row: pd.Series) -> dict[str, Any]:
    return {k: _clean(v) for k, v in row.to_dict().items()}


@dataclass
class Dataset:
    root: Path
    tables: dict[str, pd.DataFrame]

    # Indexed views, built once in `load`.
    users: dict[str, dict]
    groups: dict[str, dict]
    businesses: dict[str, dict]
    memberships: dict[tuple[str, str], dict]
    business_history: dict[tuple[str, str], dict]
    history_by_id: dict[str, dict]
    history_by_user: dict[str, list[dict]]
    events: dict[tuple[str, str], dict]
    media_paths: dict[str, Path]
    notification_load: dict[str, dict]

    @classmethod
    def load(cls, root: str | Path = "dataset") -> "Dataset":
        root = Path(root)
        tables = {name: pd.read_csv(root / f"{name}.csv") for name in CSV_FILES}

        users = {r.user_id: _row_dict(r) for _, r in tables["users"].iterrows()}
        groups = {r.group_id: _row_dict(r) for _, r in tables["groups"].iterrows()}
        businesses = {
            r.business_id: _row_dict(r) for _, r in tables["business_accounts"].iterrows()
        }
        memberships = {
            (r.group_id, r.user_id): _row_dict(r)
            for _, r in tables["group_members"].iterrows()
        }
        business_history = {
            (r.user_id, r.business_id): _row_dict(r)
            for _, r in tables["user_business_history"].iterrows()
        }
        events = {
            (r.user_id, r.message_id): _row_dict(r)
            for _, r in tables["message_events"].iterrows()
        }

        history_by_id: dict[str, dict] = {}
        history_by_user: dict[str, list[dict]] = {}
        for _, r in tables["message_history"].iterrows():
            record = _row_dict(r)
            # Attach the user's reaction so retrieval never has to re-join.
            record["reaction"] = events.get((record["user_id"], record["message_id"]))
            history_by_id[record["message_id"]] = record
            history_by_user.setdefault(record["user_id"], []).append(record)

        media_paths: dict[str, Path] = {}
        for _, r in tables["images"].iterrows():
            media_paths[r.image_id] = root / r.file_path
        for _, r in tables["voice_notes"].iterrows():
            media_paths[r.voice_note_id] = root / r.file_path

        load_table = tables["daily_notification_summary"]
        notification_load = {
            user_id: {
                "avg_sent_per_day": round(float(chunk.notifications_sent.mean()), 2),
                "avg_dismissed_per_day": round(float(chunk.notifications_dismissed.mean()), 2),
                "dismiss_rate": round(
                    float(chunk.notifications_dismissed.sum())
                    / max(float(chunk.notifications_sent.sum()), 1.0),
                    3,
                ),
            }
            for user_id, chunk in load_table.groupby("user_id")
        }

        return cls(
            root=root,
            tables=tables,
            users=users,
            groups=groups,
            businesses=businesses,
            memberships=memberships,
            business_history=business_history,
            history_by_id=history_by_id,
            history_by_user=history_by_user,
            events=events,
            media_paths=media_paths,
            notification_load=notification_load,
        )

    def messages(self, which: str = "messages") -> list[dict]:
        return [_row_dict(r) for _, r in self.tables[which].iterrows()]

    def media_path(self, media_id: str | None) -> Path | None:
        if not media_id:
            return None
        return self.media_paths.get(media_id)
