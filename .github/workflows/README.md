# PR CI

This repository runs path-aware unit tests for pull requests targeting `main`.
There is no nightly workflow and no custom PR planner. A separate cluster ST workflow runs after qualifying changes reach `main`.

## Module routing

| Workflow | Changed path | Test directory |
| --- | --- | --- |
| `PR Test (hcu-training-reliability)` | `hcu-training-reliability/**` | `hcu-training-reliability/test/unit` |
| `PR Test (hcu_resiliency_ext)` | `hcu_resiliency_ext/**`, `.gitmodules` | `hcu_resiliency_ext/tests` |
| `PR Test (hcu-envcheck)` | `hcu-envcheck/**` | `hcu-envcheck/tests` |
| `PR Test (hygon-ft-k8s)` | `hygon-ft-k8s/**` | `hygon-ft-k8s/tests` |
| `PR Test (stack-analyzer)` | `stack-analyzer/**` | `stack-analyzer/tests` |

Each event-level `paths` filter starts only the corresponding workflow. A
change to the shared `_run-unit-tests.yml` implementation or
`.github/actionlint.yaml` starts all five workflows. Unrelated documentation
changes do not start any test workflow.

The PR gate uses `pull_request_target`, so the triggering workflow text always
comes from the target branch. Same-repository and fork PRs both run automatically
without an approval job. Checkout uses the PR head repository and immutable head
SHA from the event; changes to workflow files in the PR cannot change the trusted
controller for that run.
The checked-out PR source is mounted read-only at `/ci-source` and copied into
a writable, temporary `/workspace` inside the container. Tests can create
fixtures beside the source without modifying the runner checkout. `/tmp` and
`/workspace` allow executable test fixtures but disallow devices and setuid
executables. The container receives no GitHub token or network access. All
Linux capabilities are dropped except SETUID/SETGID, needed by cross-UID
process-control tests; privilege escalation is disabled.
The test container is limited to 2 CPUs, 2 GiB RAM, and 256 processes.

Existing failed runs retain the old target-branch workflow. After this change
reaches `main`, emit a fresh qualifying PR event (for example, a new commit or
closing and reopening the PR) to test an existing fork PR.

During the resiliency directory rename, an old caller targeting
`hcu_resiliency_ext/tests` also supports PR source with `nvidia_resiliency_ext/tests`.
The shared workflow switches the test target only when the old directory is
absent, and adds the new project root to the Python import path for the new
target. PR #16 updates the module routing and names.

## Runner and image preparation

Authorize this repository for runner group `ci-general`. The runner must have
the labels `self-hosted`, `ci`, `nmz2`, and `bw1100`, plus a working Docker
daemon. This routes the jobs to `ci-nmz2`.

Load the reviewed image archive on every eligible runner before enabling the
workflows:

```bash
docker load --input /path/to/hygon-ft-controller-latest.tar
docker image inspect --format '{{.Id}}' hygon/ft-controller:latest
```

The expected image metadata is:

| Field | Value |
| --- | --- |
| Tag | `hygon/ft-controller:latest` |
| Image ID | `sha256:e2bab66a143d17a9526c14af34f6ba2bd6486d74c51b5c4deb74f33f9d98eb93` |
| Archive SHA-256 | `94c82bc110b2dbfb7e52b453bf2604e1088b8a3f8b45949a58c0d95a41202f0a` |
| Python | `3.11.16` |

The workflow rejects a matching tag with a different image ID. The supplied
image contains the runtime Kubernetes and requests dependencies but not the
test runner. A trusted setup container installs fixed versions of `pytest` and
`numpy` into a per-job Docker volume. The test container mounts that volume
read-only and runs with `--network none`.

Because these are new `pull_request_target` workflows, the first real run can
only occur after the reviewed workflows are present on the default branch and
a new qualifying pull-request event is emitted.


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
manual dispatches. Its controller uses `hygon-cpu`; training
resources come from the `hygon-bw1100-8card` pool template. See
[the ST guide](../../tests/st/README.md) for credentials, allocated nodes,
coverage, isolation impact, and compensating cleanup.
