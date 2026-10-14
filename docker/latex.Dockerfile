# syntax=docker/dockerfile:1
# Minimal TeX Live for building report/main.tex with latexmk (Debian base shared with the trainer image).
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
        lmodern make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
CMD ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
