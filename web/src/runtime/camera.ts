// Webcam helper: getUserMedia with graceful error reporting + device listing.

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export async function listCameras(): Promise<CameraDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((d) => d.kind === 'videoinput')
    .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
}

export async function openCamera(deviceId: string | null): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia is unavailable in this runtime.');
  }
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    },
  };
  try {
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    const name = error instanceof DOMException ? error.name : '';
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      throw new Error('No webcam found. Check the connection or use the video-file input.');
    }
    if (name === 'NotAllowedError') {
      throw new Error('Camera access was denied.');
    }
    if (name === 'NotReadableError') {
      throw new Error('Cannot open the camera (it may be in use by another app).');
    }
    throw error instanceof Error ? error : new Error(String(error));
  }
}

export function stopStream(stream: MediaStream | null): void {
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
}
