SHELL := /bin/sh
PYTHON ?= python
PIP := $(PYTHON) -m pip

SCRIPT := local-ai-voice.py
APP_NAME := local-ai-voice
WEBRTC_SCRIPT := browser_webrtc.py
WEBRTC_APP_NAME := local-ai-voice-webrtc
REQUIREMENTS := numpy sounddevice fastapi uvicorn aiortc av pydantic
OPENVINO_PACKAGES := openvino openvino-genai openvino-tokenizers

.PHONY: all install install-build build build-webrtc run run-webrtc clean

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

build-webrtc: install install-build
	$(PYTHON) -m PyInstaller --onefile --name $(WEBRTC_APP_NAME) \
		--collect-binaries openvino \
		--collect-data openvino \
		--collect-binaries openvino_genai \
		--collect-data openvino_genai \
		--collect-binaries openvino_tokenizers \
		--collect-data openvino_tokenizers \
		--collect-binaries av \
		--collect-data aiortc \
		--add-data "web/index.html;web" \
		$(WEBRTC_SCRIPT)

run:
	$(PYTHON) $(SCRIPT)

run-webrtc:
	$(PYTHON) $(WEBRTC_SCRIPT)

clean:
	$(RM) -r build dist __pycache__ *.spec
