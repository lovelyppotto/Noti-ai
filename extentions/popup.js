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

    ws = new WebSocket('ws://localhost:8000/ws/realtime-stt');
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      audioContext = new AudioContext({ sampleRate: 16000 });
      mediaStreamSource = audioContext.createMediaStreamSource(stream);

      // (1) 원본 오디오를 직접 재생
      mediaStreamSource.connect(audioContext.destination);

      processor = audioContext.createScriptProcessor(4096, 1, 1);
      mediaStreamSource.connect(processor);
      processor.connect(audioContext.destination);

      function float32ToInt16(buffer) {
  const l = buffer.length;
  const buf = new Int16Array(l);
  
  for (let i = 0; i < l; i++) {
    // 값의 범위를 제한하고 스케일링
    const s = Math.max(-1, Math.min(1, buffer[i]));
    buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  
  return buf;
}

      // 오디오 처리 함수 수정
processor.onaudioprocess = (event) => {
  const inputBuffer = event.inputBuffer.getChannelData(0);
  audioBufferQueue.push(new Float32Array(inputBuffer));

  const totalSamples = audioBufferQueue.reduce((sum, buf) => sum + buf.length, 0);
  
  if (totalSamples >= 32000) { // 2초 분량
    const mergedBuffer = new Float32Array(totalSamples);
    let offset = 0;
    for (const buf of audioBufferQueue) {
      mergedBuffer.set(buf, offset);
      offset += buf.length;
    }
    audioBufferQueue = []; // 큐 비우기

    if (ws.readyState === WebSocket.OPEN) {
      // Float32Array를 Int16Array로 변환 후 전송
      const int16Data = float32ToInt16(mergedBuffer);
      ws.send(int16Data.buffer);
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
    ws.send("__SUMMARY__");
  }

  audioBufferQueue = []; // 큐도 비우기
  console.log("[녹음 중지]");
});

// // 요약 버튼
// document.getElementById("summaryBtn").addEventListener("click", () => {
//   if (ws && ws.readyState === WebSocket.OPEN) {
//     ws.send("__SUMMARY__");
//   }
// });