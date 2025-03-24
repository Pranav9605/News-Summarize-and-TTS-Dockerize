FROM python:3.9

# Install Supervisor as root
USER root
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Create a new user with UID 1000
RUN useradd -m -u 1000 user

# Switch to the new non-root user
USER user

# Ensure PATH includes /home/user/.local/bin and /usr/bin (where supervisord is installed)
ENV PATH="/home/user/.local/bin:/usr/bin:$PATH"

# Set working directory to /app
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files into /app with proper ownership
COPY --chown=user . /app

# Expose port 7860 (the only port exposed to Hugging Face Spaces)
EXPOSE 7860

# Start Supervisor in non-daemon mode using the configuration file at /app/supervisord.conf
CMD ["supervisord", "-n", "-c", "/app/supervisord.conf"]
