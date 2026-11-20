.PHONY: up down logs restart test clean

## Build and start backend (:8000) and frontend (:3000)
up:
	docker compose up --build -d
	@echo "Backend:  http://localhost:8000/health"
	@echo "Frontend: http://localhost:3000"

## Stop the stack (the chat-data volume and history are kept)
down:
	docker compose down

logs:
	docker compose logs -f

restart: down up

## Run the backend test suite inside a throwaway container
test:
	docker compose run --rm --no-deps --user root -v "$(CURDIR)/backend:/src" -w /src backend \
		sh -c "pip install -q -r requirements-dev.txt && python -m pytest -q"

## Stop the stack AND delete the SQLite volume
clean:
	docker compose down -v
