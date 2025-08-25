const COMMANDS: &[&str] = &[
    "list_models",
    "list_voices",
    "load_model",
    "synthesize",
    "list_installed_models",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS).build();
}
