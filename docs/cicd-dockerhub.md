# GitHub Actions and Docker Hub setup

The repository implements a quality-gated release pipeline:

1. pushes and pull requests run linting, tests, coverage, configuration validation, and both Docker builds;
2. a push to `main` or a semantic release tag such as `v1.0.0` invokes the same CI workflow as a publication gate; and
3. only after that gate passes are the backend and frontend images published to Docker Hub.

## Why `.env` is not used by GitHub Actions

`.env` is local runtime configuration and is intentionally excluded by `.gitignore` and `.dockerignore`. Uploading or committing it would expose the Gemini key and database password and would mix local application secrets with registry credentials.

Normal CI uses explicit non-production placeholder environment values because all Gemini and database interactions are mocked. Docker Hub credentials are stored separately as encrypted GitHub Actions repository secrets. They are made available only to the release workflow and are never copied into either image.

## One-time Docker Hub setup

1. Create or sign in to a Docker Hub account.
2. Create these two repositories under that account:
   - `enterprise-ai-sql-copilot-backend`
   - `enterprise-ai-sql-copilot-frontend`
3. In Docker Hub, open **Account settings → Personal access tokens**.
4. Create an access token with read/write permission. Copy it when shown; do not put it in `.env`.

## One-time GitHub setup

In the GitHub repository, open **Settings → Secrets and variables → Actions → New repository secret** and create:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub personal access token |

No repository variable or application `.env` value is required for publishing.

## CI and release flow

Push the Phase 6 commit to `main`:

```bash
git push origin main
```

This starts `Build and publish images`, runs the complete CI gate, and automatically publishes both images with `main` and `latest` tags. No Git tag or manual `docker push` command is required for this normal continuous-delivery path.

Semantic tags remain available when a permanent versioned release is wanted. After the `main` publication is green, optionally create one:

```bash
git tag -a v1.0.0 -m "Enterprise AI SQL Analytics Copilot v1.0.0"
git push origin v1.0.0
```

The tag starts the same workflow. It reruns all quality gates, then additionally publishes:

- `<dockerhub-username>/enterprise-ai-sql-copilot-backend:1.0.0`
- `<dockerhub-username>/enterprise-ai-sql-copilot-backend:1.0`
- `<dockerhub-username>/enterprise-ai-sql-copilot-backend:latest`
- `<dockerhub-username>/enterprise-ai-sql-copilot-frontend:1.0.0`
- `<dockerhub-username>/enterprise-ai-sql-copilot-frontend:1.0`
- `<dockerhub-username>/enterprise-ai-sql-copilot-frontend:latest`

A normal `main` push publishes these rolling tags:

- `<dockerhub-username>/enterprise-ai-sql-copilot-backend:main`
- `<dockerhub-username>/enterprise-ai-sql-copilot-backend:latest`
- `<dockerhub-username>/enterprise-ai-sql-copilot-frontend:main`
- `<dockerhub-username>/enterprise-ai-sql-copilot-frontend:latest`

Verify the published images in Docker Hub or pull them explicitly:

```bash
docker pull <dockerhub-username>/enterprise-ai-sql-copilot-backend:latest
docker pull <dockerhub-username>/enterprise-ai-sql-copilot-frontend:latest
```

Use `:1.0.0` instead after publishing the optional `v1.0.0` Git tag.

If publishing fails, check the release workflow logs without pasting token values into issues or logs. The most common causes are a missing repository, a username mismatch, or a token without write permission.
