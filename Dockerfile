# KryoSens Experimental Copilot - Hugging Face Docker Space. One process, standard library only
# (the Gemini REST path uses urllib; langgraph/langchain are optional and not installed).
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=7860
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app run_server.py ./
COPY --chown=app:app src/ src/
USER app
EXPOSE 7860
CMD ["python", "run_server.py"]
