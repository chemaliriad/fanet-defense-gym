# Kubernetes: sharded generation and evaluation

Three Indexed Jobs share one ConfigMap:

* `fanet-generate-suite` writes `GENERATE_SIZE` training scenarios as `NUM_SHARDS` JSONL files;
* `fanet-evaluate` evaluates `POLICY` on the `SUITE_SIZE` test scenarios, one shard per pod;
* `fanet-dataset` writes language-model training records (prompt, teacher answer,
  verifiable reward) for `GENERATE_SIZE` training scenarios, one JSONL shard per pod.

Each pod reads `JOB_COMPLETION_INDEX` (set by Kubernetes) and handles scenarios
`index, index + N, index + 2N, ...`. Shards are disjoint and their union is the full suite,
so results can be merged without coordination. Pods run as a non-root user with a read-only
root filesystem, no service-account token and all capabilities dropped.

Run locally with [kind](https://kind.sigs.k8s.io/):

```bash
docker build -t fanet-defense:0.1.0 .
kind create cluster --name fanet
kind load docker-image fanet-defense:0.1.0 --name fanet
kubectl apply -k k8s/base
kubectl wait --for=condition=complete job/fanet-evaluate --timeout=600s
kubectl logs -l app.kubernetes.io/component=evaluation --tail=20
```

CI renders the manifests with `kubectl kustomize` and validates them with `kubeconform`.
These manifests have been validated, not run in a production cluster. Writing results to a
PersistentVolume or object storage is left to the deployment.
