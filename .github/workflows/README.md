# PR CI

This repository runs path-aware unit tests for pull requests targeting `main`.
There is no nightly workflow and no custom PR planner. A separate cluster ST workflow runs after qualifying changes reach `main`.

## Module routing

| Workflow | Changed path | Test directory |
| --- | --- | --- |
| `PR Test (hcu-training-reliability)` | `hcu-training-reliability/**` | `hcu-training-reliability/test/unit` |
| `PR Test (nvidia_resiliency_ext)` | `nvidia_resiliency_ext/**`, `.gitmodules` | `nvidia_resiliency_ext/tests` |
| `PR Test (hcu-envcheck)` | `hcu-envcheck/**` | `hcu-envcheck/tests` |
| `PR Test (hygon-ft-k8s)` | `hygon-ft-k8s/**` | `hygon-ft-k8s/tests` |
| `PR Test (stack-analyzer)` | `stack-analyzer/**` | `stack-analyzer/tests` |

Each event-level `paths` filter starts only the corresponding workflow. A
change to the shared `_run-unit-tests.yml` implementation or
`.github/actionlint.yaml` starts all five workflows. Unrelated documentation
changes do not start any test workflow.

The PR gate uses `pull_request_target`. GitHub reads its controller from the
target branch and checks out the PR head repository at its immutable SHA with
`persist-credentials: false`. Each module retains its own event-level path filter.

## Kubernetes runner and runtime

Unit jobs use the image-capable `hygon-build` ARC pool. The pool supplies the
Kubernetes job container, CPU, memory, shared-memory and mount configuration.
These tests request zero HCU devices. Authorize this repository for the pool's
organization runner group.

Set repository variable `ST_TRAINING_IMAGE` to the reviewed runtime's complete
`registry/image@sha256:<digest>` reference. Kubernetes resolves this fixed image
for every job. The runtime supplies Python, Bash, Git and the product libraries.
The workflow installs `numpy==2.3.5` and `pytest==9.1.1` in a job-local directory.
When available, the runtime's DTK environment initializes its library paths.

Tests run against an independent writable copy under `/tmp`, with pytest plugin
autoloading disabled and `--import-mode=importlib`. Old callers using
`cluster_manager/test/unit` or `hcu_resiliency_ext/tests` retain compatibility
with the renamed project directories, including PRs introducing
`nvidia_resiliency_ext/tests`.

The runner's ephemeral job container owns the test workspace and dependency
directory. Existing queued runs retain their original runner requirements.
After the reusable workflow reaches `main`, a fresh qualifying event on the PR
starts a run using `hygon-build`.

## Cluster ST

`st-cluster.yml` runs `tests/st/ci_entry.py` directly on the same `ci-general`
runner group and `[self-hosted, ci, nmz2, bw1100]` labels. It does not reuse the
network-disabled PR unit-test container and does not change the existing PR jobs.
A push to `main` triggers ST when it changes files under `hcu-envcheck/`,
`hcu-training-reliability/`, or `hcu_resiliency_ext/`. The workflow checks out
the pushed commit with credential persistence disabled. Actions are pinned to commits.

Configure repository variables `ST_CONTROL_PYTHON`, `ST_SHARED_ROOT`, and
`ST_HOSTFILE`. The hostfile is an
allocated pool supplied by operations/resource scheduling, not a discovery scan.
Real addresses and runner-specific paths are not embedded in the workflow.
With two 8-GPU nodes in the hostfile, the default CI site template uses one
node for training and the other as standby. Adjust the topology in `site.ci.toml`.
See `tests/st/README.md` for the complete site template and prerequisites.

The runner Python must already contain pytest, ClusterShell and reliability's
controller dependencies. Git, SSH, Bash and clush, existing node SSH access,
same-path shared storage, the training image and the pinned per-node NHC deployment
are required. Training libraries remain in the training image; this workflow
neither installs them nor bootstraps NHC.
Multi-node training uses MPI inside the image; this Docker ST does not require
a controller-host MPI installation.

The fixed `hcu-cluster-st` concurrency group serializes this repository's ST runs
across refs; `cancel-in-progress` is false. External use of those machines must
still be coordinated by the resource owner. The 90-minute workflow guard is
provisional and separate from each case's timeout settings.

ST stores ownership receipts in shared storage and provides cancellation cleanup.
The always-run fallback checks only this run's exact identities; mismatches fail
rather than deleting unknown resources. Even a closed case must pass the manager
exit check before its containers can be removed; an unconfirmed stop leaves them
untouched and fails cleanup. A lost runner requires recovery-time
receipt reconciliation. Logs/JSON/TSV and checkpoint file metadata are uploaded
with `upload-artifact`; weights and Bridge source trees remain on shared storage.
Bridge preparation writes detailed Git logs and manifests into the CI run's
`bridge-dependency` directory, including fetch failures and Git timeouts.
No ST run is performed by merely collecting tests or omitting `--site`.

Each triggered ST run starts with the offline ST regression suite. Changes
limited to `tests/st/` or workflow files do not trigger a cluster run.

## Kubernetes recovery ST

`st-cluster.yml` runs cross-node checkpoint recovery on matching PRs and
manual dispatches. Its controller uses the image-capable `hygon-build` pool, and training
resources come from the `hygon-bw1100-8card` pool template. See
[the ST guide](../../tests/st/README.md) for credentials, allocated nodes,
coverage, isolation impact, and compensating cleanup.
