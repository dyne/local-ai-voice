SHELL := /bin/sh
PYTHON ?= py -3.11
PIP := $(PYTHON) -m pip

SCRIPT := local-ai-voice.py
APP_NAME := local-ai-voice
SPEC := $(APP_NAME).spec
REQUIREMENTS := numpy noisereduce webrtcvad-wheels sounddevice fastapi uvicorn websockets pydantic
OPENVINO_PACKAGES := openvino openvino-genai openvino-tokenizers

.PHONY: all install install-build spec build build-webrtc run run-web clean

all: install

install:
	$(PIP) install --upgrade pip
	$(PIP) install $(REQUIREMENTS)
	$(PIP) install --upgrade $(OPENVINO_PACKAGES)

install-build:
	$(PIP) install pyinstaller

spec: install-build
	$(PYTHON) -m PyInstaller.utils.cliutils.makespec --onefile --name $(APP_NAME) \
		--hidden-import browser_webrtc \
		--hidden-import noisereduce \
		--hidden-import webrtcvad \
		--additional-hooks-dir hooks \
		--collect-binaries openvino \
		--collect-data openvino \
		--collect-binaries openvino_genai \
		--collect-data openvino_genai \
		--collect-binaries openvino_tokenizers \
		--collect-data openvino_tokenizers \
		--add-data "web/index.html;web" \
		--add-data "web/audio-worklet.js;web" \
		$(SCRIPT)

build: install spec
	$(PYTHON) -m PyInstaller --clean $(SPEC)

build-webrtc: build

run:
	$(PYTHON) $(SCRIPT)

run-web:
	$(PYTHON) $(SCRIPT) web

clean:
	$(RM) -r build dist __pycache__ *.spec
