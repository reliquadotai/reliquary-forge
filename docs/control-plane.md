# Control Plane

Reliquary includes a lightweight FastAPI control plane for policy publication, artifact lookup, and promotion decisions.

## Start

```bash
reliquary serve-control-plane
```

Default bind settings come from:

- `RELIQUARY_API_BIND`
- `RELIQUARY_API_PORT`

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness check |
| `GET /artifacts/{artifact_type}` | List artifacts by type, optionally filtered by `window_id` |
| `GET /artifacts/{artifact_type}/{artifact_id}` | Fetch a single artifact |
| `GET /policies/latest` | Return the latest published policy |
| `POST /policy-authority/publish` | Publish a new distillation policy |
| `POST /policy-authority/promote` | Record a policy-authority decision and optionally promote a checkpoint |

## Production intent

The control plane is not the chain. Its job is to:

- expose operational APIs
- register and fetch artifact metadata
- coordinate trusted policy-authority decisions
- bridge local runs and managed infrastructure

On-chain publication still belongs to the Bittensor adapter layer.

For the first RTX6000B deployment the control plane remains localhost-only and is intended to be reached through SSH tunneling, not a public port exposure.
