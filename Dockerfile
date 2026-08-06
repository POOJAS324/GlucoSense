# ─────────────────────────────────────────────────────────────────
# GlucoSense — Dockerfile
# ─────────────────────────────────────────────────────────────────
# Builds a self-contained image that:
#   1. Installs all Python dependencies
#   2. Trains the model at image-build time (models/ is baked in)
#   3. Launches the Streamlit app on port 8501
#
# Build & run:
#   docker build -t glucosense .
#   docker run -p 8501:8501 glucosense
#
# Then open http://localhost:8501 in your browser.
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (layer-cache friendly)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Train the model once at build time so the image is self-contained.
# Remove this line if you prefer to mount a pre-trained models/ volume.
RUN python src/train.py

# Expose Streamlit's default port
EXPOSE 8501

# Health-check so container orchestrators can verify the app is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
