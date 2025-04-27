let ws = null;
let mediaRecorder = null;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'START_STT') {
    startCapturing();
  } else if (message.type === 'STOP_STT') {
    stopCapturing();
  }
});

function startCapturing() {
  chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
    if (!stream) {
      console.error("[tabCapture 실패]");
      if (chrome.runtime.lastError) console.error(chrome.runtime.lastError.message);
      return;
    }

    ws = new WebSocket('ws://localhost:8000');
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log("[WebSocket 연결 완료]");
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          event.data.arrayBuffer().then((buffer) => {
            ws.send(buffer);
          });
        }
      };

      mediaRecorder.start(1000); // 1초마다 조각 보내기
      console.log("[녹음 시작]");
    };

    ws.onmessage = (event) => {
      const transcriptText = event.data;
      chrome.runtime.sendMessage({ type: 'TRANSCRIPT', text: transcriptText });
    };

    ws.onerror = (error) => {
      console.error("[WebSocket 오류]", error);
    };

    ws.onclose = () => {
      console.warn("[WebSocket 연결 종료]");
    };
  });
}

function stopCapturing() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    console.log("[녹음 중지]");
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
}