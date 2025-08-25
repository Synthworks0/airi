const COMMANDS: &[&str] = &[
  "load_ort_model_whisper",
  "ipc_audio_transcription",
  "list_models",
  "list_installed_models",
];

fn main() {
  tauri_plugin::Builder::new(COMMANDS).build();
}
