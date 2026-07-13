# Ollama server pre-loaded with an embedding model and a small chat model.
# Baking models into the image avoids re-downloading them every time the
# Serverless Endpoint container starts, which would waste billed uptime.
 
FROM ollama/ollama:latest
 
# Start Ollama in the background, wait for it to be ready, pull models,
# then let the final CMD run the server in the foreground.
RUN (ollama serve &) && \
    sleep 5 && \
    ollama pull qwen3:32b
 
EXPOSE 11434
 
ENTRYPOINT ["ollama", "serve"]
 