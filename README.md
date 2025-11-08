# hub-nexus-cortex (v2)

Cortex agentique : MoME (/route), consensus multi-LLM (/vote), scheduler (/schedule/predict), models warm-up (/models/swap).

## Démarrage rapide (CPU)

```bash
cp configs/.env.example .env
docker compose --profile deps up -d
docker compose --profile cortex up -d
# Option: LLMs locaux
docker compose --profile ollama up -d

curl -s http://localhost:8100/health
```
