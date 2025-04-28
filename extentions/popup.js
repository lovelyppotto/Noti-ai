let ws;
let audioContext;
let mediaStreamSource;
let processor;
let capturedStream;
let audioBufferQueue = [];

document.getElementById('start').addEventListener('click', async () => {
  chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
    if (!stream) {
      console.error("오디오 캡처 실패");
      if (chrome.runtime.lastError) console.error(chrome.runtime.lastError.message);
      return;
    }

    capturedStream = stream;

    ws = new WebSocket('ws://localhost:8000/ws');
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      audioContext = new AudioContext({ sampleRate: 16000 });
      mediaStreamSource = audioContext.createMediaStreamSource(stream);

      processor = audioContext.createScriptProcessor(4096, 1, 1);
      mediaStreamSource.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer.getChannelData(0);
        audioBufferQueue.push(new Float32Array(inputBuffer));

        const totalSamples = audioBufferQueue.reduce((sum, buf) => sum + buf.length, 0);
        
        if (totalSamples >= 32000) { // 2초 분량 (16000Hz × 2초)
          const mergedBuffer = new Float32Array(totalSamples);
          let offset = 0;
          for (const buf of audioBufferQueue) {
            mergedBuffer.set(buf, offset);
            offset += buf.length;
          }
          audioBufferQueue = []; // 큐 비우기

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(mergedBuffer.buffer);
          }
        }
      };

      console.log("[녹음 시작]");
    };

    ws.onmessage = (event) => {
      const transcriptText = event.data;
      const transcriptDiv = document.getElementById('transcript');

      const placeholder = document.getElementById('placeholder');

      // placeholder 제거
      if (placeholder) {
        placeholder.remove();
      }

      // 기존 내용 유지하면서 줄바꿈 추가
      transcriptDiv.append(document.createTextNode(transcriptText + '\n'));
      transcriptDiv.scrollTop = transcriptDiv.scrollHeight; // 스크롤 자동 내려가게
    };
  });
});

document.getElementById('stop').addEventListener('click', () => {
  if (processor) processor.disconnect();
  if (mediaStreamSource) mediaStreamSource.disconnect();
  if (audioContext) audioContext.close();

  if (capturedStream) {
    capturedStream.getTracks().forEach(track => track.stop());
    capturedStream = null;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  audioBufferQueue = []; // 큐도 비우기
  console.log("[녹음 중지]");
});

// 요약 버튼
document.getElementById("summaryBtn").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("__SUMMARY__");
  }
});