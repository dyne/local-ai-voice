class PcmResampleProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetSampleRate = Number(opts.targetSampleRate || 16000);
    this.chunkSamples = Number(opts.chunkSamples || 2048);
    this.sourcePos = 0;
    this.sourceBuffer = new Float32Array(0);
    this.outputBuffer = new Float32Array(this.chunkSamples);
    this.outputOffset = 0;
    this.step = sampleRate / this.targetSampleRate;
  }

  appendInput(input) {
    const merged = new Float32Array(this.sourceBuffer.length + input.length);
    merged.set(this.sourceBuffer, 0);
    merged.set(input, this.sourceBuffer.length);
    this.sourceBuffer = merged;
  }

  emitOutput() {
    if (this.outputOffset !== this.chunkSamples) {
      return;
    }
    const out = this.outputBuffer;
    this.port.postMessage(out.buffer, [out.buffer]);
    this.outputBuffer = new Float32Array(this.chunkSamples);
    this.outputOffset = 0;
  }

  resampleAvailable() {
    while (this.sourcePos + 1 < this.sourceBuffer.length) {
      const left = Math.floor(this.sourcePos);
      const frac = this.sourcePos - left;
      const sample = this.sourceBuffer[left] + (this.sourceBuffer[left + 1] - this.sourceBuffer[left]) * frac;
      this.outputBuffer[this.outputOffset++] = sample;
      this.sourcePos += this.step;
      this.emitOutput();
    }

    const drop = Math.floor(this.sourcePos);
    if (drop > 0) {
      this.sourceBuffer = this.sourceBuffer.slice(drop);
      this.sourcePos -= drop;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    const frames = input[0].length;
    const mono = new Float32Array(frames);
    for (let channel = 0; channel < input.length; channel += 1) {
      const samples = input[channel];
      for (let i = 0; i < frames; i += 1) {
        mono[i] += samples[i];
      }
    }
    for (let i = 0; i < frames; i += 1) {
      mono[i] /= input.length;
    }

    this.appendInput(mono);
    this.resampleAvailable();
    return true;
  }
}

registerProcessor("pcm-resample-processor", PcmResampleProcessor);
