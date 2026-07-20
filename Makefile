.PHONY: install run test lint format clean-runtime init-db demo-data didi-example-data \
	sidecar-dev docker-build docker-push

IMAGE ?= hub.xiaojukeji.com/parkerzhang/forecast:v3

install:
	uv sync

run:
	uv run streamlit run app.py

# 本地起 SSO sidecar(可配合 SSO_DEV_FAKE_USER 联调鉴权链路)
sidecar-dev:
	uv run uvicorn deploy.sso_sidecar.main:app --reload --port 9000

# 构建内网部署镜像(Apple Silicon 需指定 amd64)
docker-build:
	docker build --platform linux/amd64 -t $(IMAGE) .

docker-push:
	docker push $(IMAGE)

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

init-db:
	uv run python scripts/init_db.py

demo-data:
	uv run python scripts/generate_demo_data.py

didi-example-data:
	uv run python -m scripts.generate_didi_example_data

clean-runtime:
	uv run python scripts/clean_runtime.py
