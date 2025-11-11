.PHONY: smoke up down
smoke:
	docker compose --profile deps --profile mesh --profile ollama up -d
	curl -fsS http://localhost:7702/health
	curl -fsS http://localhost:6337/readyz
	docker compose --profile cortex up -d --build
	curl -fsS http://localhost:8100/health

up:
	docker compose --profile deps --profile mesh --profile ollama --profile cortex up -d

down:
	docker compose down -v
