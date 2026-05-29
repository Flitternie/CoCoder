"""partition_into_groups tool — partition + init task list + spawn agents.

Composite tool that:
1. Partitions files into cohesion-based groups (cosine + InfoMap + role + sibling)
2. Initializes shared_task_list from RIB dependency graph
3. Spawns one group agent per partition group with initial messages

This eliminates ~200s of LLM thinking time that the leader previously spent
constructing task_list init args and spawn calls manually.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor

from common.utils.rib_helpers import load_architecture, flatten_files

from cohesionbase.partition import (
    detect_roles,
    attach_init_files,
    infomap_partition, merge_small_groups, role_grouping, lift_independent,
    weights_rib_cosine,
)

log = logging.getLogger(__name__)

ROLE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class PartitionIntoGroupsAction(Action):
    rib_path: str = Field(
        default="architecture/rib.json",
        description="Path to RIB JSON (relative to workspace)",
    )
    output_path: str = Field(
        default="architecture/partition.json",
        description="Path to write partition result (relative to workspace)",
    )


class PartitionIntoGroupsObservation(Observation):
    num_groups: int = 0
    num_files: int = 0
    mq_score: float = 0.0
    groups_summary: str = ""
    roles_summary: str = ""
    auto_status: str = ""
    error: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"partition_into_groups FAILED: {self.error}")]
        text = (
            f"Partition written to architecture/partition.json\n"
            f"Groups: {self.num_groups}, Files: {self.num_files}, MQ: {self.mq_score:.3f}\n\n"
            f"Roles: {self.roles_summary}\n\n"
            f"{self.groups_summary}"
        )
        if self.auto_status:
            text += f"\n\nAuto-setup: {self.auto_status}"
        return [TextContent(text=text)]


# ---------------------------------------------------------------------------
# MQ Score
# ---------------------------------------------------------------------------

def _compute_mq(partition: dict[str, int], weights: dict[tuple, float]) -> float:
    if len(set(partition.values())) <= 1:
        return 0.0
    intra = inter = 0.0
    for (s, t), w in weights.items():
        if s in partition and t in partition:
            if partition[s] == partition[t]:
                intra += w
            else:
                inter += w
    total = intra + inter
    return (intra - inter) / total if total else 0.0


# ---------------------------------------------------------------------------
# Group Naming
# ---------------------------------------------------------------------------

def _assign_group_names(groups: dict[int, list[str]], roles: dict[str, str]) -> dict[int, str]:
    group_names: dict[int, str] = {}
    used: set[str] = set()

    for gid, files in sorted(groups.items()):
        file_roles = {roles.get(f, "core") for f in files}
        if file_roles == {"in_hub"} and len(files) == 1:
            name = f"in_hub_{Path(files[0]).stem}"
        elif "out_hub" in file_roles:
            name = "group_out_hub"
        else:
            stems = sorted(Path(f).stem for f in files)
            name = f"group_{stems[0]}"

        base = name
        counter = 2
        while name in used:
            name = f"{base}_{counter}"
            counter += 1
        used.add(name)
        group_names[gid] = name

    return group_names


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class PartitionIntoGroupsExecutor(ToolExecutor[PartitionIntoGroupsAction, PartitionIntoGroupsObservation]):
    """Partition + init task list + spawn agents in one step."""

    def __init__(self, workspace_dir: Path, orchestrator=None, message_bus=None, task_list_path: Path | None = None):
        self.workspace_dir = workspace_dir
        self.orchestrator = orchestrator
        self.message_bus = message_bus
        self.task_list_path = task_list_path

    def __call__(
        self,
        action: PartitionIntoGroupsAction,
        conversation=None,
    ) -> PartitionIntoGroupsObservation:
        rib_abs = self.workspace_dir / action.rib_path
        output_abs = self.workspace_dir / action.output_path

        try:
            rib = load_architecture(rib_abs)
        except Exception as e:
            return PartitionIntoGroupsObservation(error=f"Cannot read RIB: {e}")

        # Step 1: Cosine similarity weights
        try:
            weights = weights_rib_cosine(rib)
        except Exception as e:
            return PartitionIntoGroupsObservation(error=f"Weight computation failed: {e}")

        # Step 2: Role Detection + InfoMap on core + reassemble
        roles = detect_roles(rib, threshold=ROLE_THRESHOLD)
        in_hubs = sorted(f for f, r in roles.items() if r == "in_hub")
        out_hubs = sorted(f for f, r in roles.items() if r == "out_hub")
        cores = sorted(f for f, r in roles.items() if r == "core")

        try:
            partition = role_grouping(rib, weights, infomap_partition, threshold=ROLE_THRESHOLD)
        except Exception as e:
            return PartitionIntoGroupsObservation(error=f"Partition failed: {e}")

        # Step 3: Latent parallelism exploitation
        partition = lift_independent(partition, rib)

        # Compute MQ (original)
        mq = _compute_mq(partition, weights)
        partition_with_init, init_attachments = attach_init_files(partition, rib, roles)

        # --- Build original groups output ---
        def _build_output(part):
            gbi: dict[int, list[str]] = defaultdict(list)
            for f, g in part.items():
                gbi[g].append(f)
            for g in gbi:
                gbi[g] = sorted(gbi[g])
            gnames = _assign_group_names(gbi, roles)
            go, ftg = [], {}
            for g, files in sorted(gbi.items()):
                gn = gnames[g]
                go.append({"id": gn, "files": files})
                for f in files:
                    ftg[f] = gn
            return go, ftg

        groups_output, file_to_group = _build_output(partition_with_init)

        # Step 4: Optionally merge small groups
        use_merged = os.environ.get("ENABLE_MERGE_GROUPS", "0") == "1"
        if use_merged:
            merged_partition = merge_small_groups(partition, rib, weights=weights)
            merged_partition_with_init, merged_init_attachments = attach_init_files(
                merged_partition, rib, roles
            )
            merged_groups_output, merged_file_to_group = _build_output(merged_partition_with_init)
            merged_mq = _compute_mq(merged_partition, weights)
            active_groups = merged_groups_output
            active_ftg = merged_file_to_group
            active_mq = merged_mq
        else:
            merged_groups_output = None
            merged_file_to_group = None
            merged_init_attachments = None
            merged_mq = None
            active_groups = groups_output
            active_ftg = file_to_group
            active_mq = mq

        result = {
            "groups": active_groups,
            "file_to_group": active_ftg,
            "mq_score": round(active_mq, 4),
            "roles": {f: r for f, r in roles.items()},
            "algorithm": "rib_cos__infomap__role__sibling",
            "init_attachments": init_attachments,
            "original_groups": groups_output,
            "original_file_to_group": file_to_group,
            "original_mq_score": round(mq, 4),
        }
        if use_merged:
            result["merged_groups"] = merged_groups_output
            result["merged_file_to_group"] = merged_file_to_group
            result["merged_init_attachments"] = merged_init_attachments
            result["merged_mq_score"] = round(merged_mq, 4)

        output_abs.parent.mkdir(parents=True, exist_ok=True)
        output_abs.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

        # --- Auto init task list + spawn agents ---
        auto_status = ""
        if self.orchestrator and self.message_bus and self.task_list_path:
            try:
                auto_status = self._auto_init_and_spawn(rib, active_groups, active_ftg)
            except Exception as e:
                auto_status = f"Auto init/spawn FAILED: {e}"
                log.error(f"[PartitionTool] {auto_status}")

        # Summaries
        summary_lines = []
        for g in active_groups:
            summary_lines.append(f"  {g['id']}: {g['files']}")

        roles_parts = []
        if in_hubs:
            roles_parts.append(f"in_hub={[f.split('/')[-1] for f in in_hubs]}")
        if out_hubs:
            roles_parts.append(f"out_hub={[f.split('/')[-1] for f in out_hubs]}")
        roles_parts.append(f"core={len(cores)} files")

        merge_note = ""
        if use_merged and len(merged_groups_output) < len(groups_output):
            merge_note = f" (merged {len(groups_output)}→{len(merged_groups_output)})"

        return PartitionIntoGroupsObservation(
            num_groups=len(active_groups),
            num_files=len(active_ftg),
            mq_score=round(active_mq, 4),
            groups_summary="\n".join(summary_lines) + merge_note,
            roles_summary=", ".join(roles_parts),
            auto_status=auto_status,
        )

    def _auto_init_and_spawn(
        self,
        rib: list[dict],
        groups_output: list[dict],
        file_to_group: dict[str, str],
    ) -> str:
        """Init shared_task_list and spawn group agents automatically."""
        from cohesionbase.tools.shared_task_list import _cmd_init
        from parallelbase.message_bus import AgentMessage

        # Build file -> deps from RIB
        file_deps: dict[str, list[str]] = {}
        for f_item in flatten_files(rib):
            path = f_item.get("path", "")
            if path:
                file_deps[path] = f_item.get("dependencies", [])

        # Build task defs for shared_task_list init
        task_defs = []
        for path, deps in file_deps.items():
            owner = file_to_group.get(path)
            task_defs.append({
                "id": path,
                "deps": deps,
                "owner": owner,
                "description": "",
            })

        # Init task list
        init_result = _cmd_init(self.task_list_path, task_defs)
        if "error" in init_result:
            return f"Task list init failed: {init_result['error']}"

        total_tasks = init_result.get("total_tasks", 0)
        ready_tasks = init_result.get("immediately_ready", 0)

        # Topo-sort files within each group for the initial message
        spawned = []
        for group in groups_output:
            gname = group["id"]
            group_files = group["files"]

            # Topo sort within group: files with fewer deps first
            group_file_set = set(group_files)
            dep_count = {}
            for f in group_files:
                dep_count[f] = len([d for d in file_deps.get(f, []) if d in group_file_set])
            sorted_files = sorted(group_files, key=lambda f: (dep_count[f], f))

            # Build initial message
            file_list = "\n".join(
                f"{i+1}. {f}" for i, f in enumerate(sorted_files)
            )
            initial_message = (
                f"Responsible files (in dependency order):\n{file_list}\n\n"
                f"Reminder: Follow Implementation Integrity Rules strictly. "
                f"Do not use test-only shortcuts."
            )

            # Spawn agent
            self.orchestrator.spawn_agent(name=gname, role="group", files=group_files)

            # Send initial message via message bus
            self.message_bus.send(AgentMessage(
                from_agent="leader",
                to_agent=gname,
                content=initial_message,
                timestamp=time.time(),
            ))
            spawned.append(gname)

        return (
            f"Task list initialized: {total_tasks} tasks ({ready_tasks} immediately ready). "
            f"Spawned {len(spawned)} agents: {spawned}"
        )


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Partition files into groups, initialize task list, and spawn group agents.

This is a composite tool that does everything needed for Phase 2+3 setup:
1. Partitions files into cohesion-based groups (cosine + InfoMap + role + sibling)
2. Writes partition to architecture/partition.json
3. Automatically initializes shared_task_list with file dependencies from RIB
4. Spawns one group agent per partition group with initial messages

After this tool completes, all group agents are running and working.
You do NOT need to call shared_task_list(init) or agent_manager(spawn) — it's done.
Just call yield_turn and wait for an "All tasks completed" message.
"""


class PartitionIntoGroupsTool(ToolDefinition[PartitionIntoGroupsAction, PartitionIntoGroupsObservation]):
    """Partition + init task list + spawn agents in one step."""

    @classmethod
    def create(cls, conv_state, workspace_dir=None, orchestrator=None,
               message_bus=None, task_list_path=None) -> Sequence[ToolDefinition]:
        if workspace_dir is None:
            raise ValueError("PartitionIntoGroupsTool requires workspace_dir")
        executor = PartitionIntoGroupsExecutor(
            Path(workspace_dir),
            orchestrator=orchestrator,
            message_bus=message_bus,
            task_list_path=Path(task_list_path) if task_list_path else None,
        )
        return [cls(
            description=_DESCRIPTION,
            action_type=PartitionIntoGroupsAction,
            observation_type=PartitionIntoGroupsObservation,
            executor=executor,
        )]
