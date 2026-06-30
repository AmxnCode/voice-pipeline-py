"""
Basic usage of the voice-pipeline Python SDK.

Install:
    pip install "voice-pipeline[all]"
"""

from voice_pipeline import VoicePipeline, PipelineConfig


def on_transcript(text: str):
    print(f"\n📝 You said: {text}")


def on_token(token: str):
    print(token, end="", flush=True)


def on_audio(wav: bytes):
    with open("output.wav", "wb") as f:
        f.write(wav)
    print(f"\n🔊 Audio saved: output.wav ({len(wav)} bytes)")


def on_error(err: Exception):
    print(f"\n❌ Error: {err}")


def on_download_progress(name: str, progress: float):
    print(f"⬇️  {name}: {progress * 100:.0f}%")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <16kHz PCM file>")
        sys.exit(1)

    print("Initializing voice pipeline...")

    pipeline = VoicePipeline(PipelineConfig())
    pipeline.on_transcript = on_transcript
    pipeline.on_token = on_token
    pipeline.on_audio = on_audio
    pipeline.on_error = on_error
    pipeline.on_download_progress = on_download_progress

    pipeline.initialize()

    with open(sys.argv[1], "rb") as f:
        audio_data = f.read()

    print(f"Processing {len(audio_data)} bytes of audio...")
    result = pipeline.process_audio(audio_data)

    if result:
        print(f"\n✅ Done — transcript: '{result.transcript}', "
              f"response: '{result.response[:60]}...', "
              f"audio: {len(result.audio) if result.audio else 0} bytes")

    pipeline.release()


if __name__ == "__main__":
    main()
