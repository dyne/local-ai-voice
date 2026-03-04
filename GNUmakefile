SHELL := /bin/sh
PYTHON ?= python
PIP := $(PYTHON) -m pip

SCRIPT := transcribe_wav.py
APP_NAME := transcribe_wav
REQUIREMENTS := numpy sounddevice
OPENVINO_PACKAGES := openvino openvino-genai openvino-tokenizers

.PHONY: all install install-build build run clean

all: install

install:
	$(PIP) install --upgrade pip
	$(PIP) install $(REQUIREMENTS)
	$(PIP) install --upgrade $(OPENVINO_PACKAGES)

install-build:
	$(PIP) install pyinstaller

build: install install-build
	$(PYTHON) -m PyInstaller --onefile --name $(APP_NAME) \
		--collect-binaries openvino \
		--collect-data openvino \
		--collect-binaries openvino_genai \
		--collect-data openvino_genai \
		--collect-binaries openvino_tokenizers \
		--collect-data openvino_tokenizers \
		$(SCRIPT)

run:
	$(PYTHON) $(SCRIPT)

clean:
	$(RM) -r build dist __pycache__ *.spec
