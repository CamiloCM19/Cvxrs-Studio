#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use cvxrs_api::{Method, Solver};
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::solution::{Solution, Status};
use cvxrs_io::{read_json_problem, write_solution, JsonProblem};
use eframe::egui::{
    self, Align, Color32, FontData, FontDefinitions, FontFamily, FontId, Margin, RichText, Stroke,
    TextStyle,
};
use eframe::{App, CreationContext, Frame, NativeOptions};
use reqwest::blocking::Client;
use rfd::FileDialog;

const SAMPLE_QP_JSON: &str = r#"
{
  "kind": "qp",
  "problem": {
    "quadratic": {
      "nrows": 2,
      "ncols": 2,
      "indptr": [0, 1, 2],
      "indices": [0, 1],
      "data": [2.0, 2.0]
    },
    "linear": [-2.0, -5.0],
    "inequalities": {
      "matrix": {
        "nrows": 2,
        "ncols": 2,
        "indptr": [0, 1, 2],
        "indices": [0, 1],
        "data": [1.0, 1.0]
      },
      "rhs": [1.0, 1.0]
    },
    "equalities": null,
    "bounds": {
      "lower": [0.0, 0.0],
      "upper": [10.0, 10.0]
    }
  }
}
"#;

const SAMPLE_DESCRIPTION: &str =
    "Problema cuadratico sencillo con dos variables, limites 0 <= x <= 10 y restricciones x <= 1.";

const GEMINI_API_KEY: &str = "AIzaSyDFhGSAhYarVemN00g_7s5J16tbHR6_qT8";
const GEMINI_MODEL: &str = "gemini-2.5-flash";
const GEMINI_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const GEMINI_SYSTEM_PROMPT: &str = r#"Eres un asistente integrado en cvxrs Studio. Debes leer problemas de optimizacion convexa escritos a mano desde imagenes y devolver exclusivamente un objeto JSON valido que siga exactamente el esquema de entrada de cvxrs. Formato requerido:
- Objeto de nivel superior con campos `kind` y `problem`.
- `kind` es "qp" o "lp" en minusculas.
- `problem` contiene `linear` (vector), `bounds` (objeto con `lower` y `upper` o null) y las secciones opcionales `equalities` e `inequalities`.
- Cuando `kind` sea "qp" incluye `quadratic`; si no existe usa null.
- Las matrices (`quadratic`, `equalities.matrix`, `inequalities.matrix`) usan formato CSC con `nrows`, `ncols`, `indptr`, `indices`, `data`.
- `equalities` e `inequalities` son null o un objeto con `matrix` y `rhs` (vector).
- Para limites sin cota utiliza las cadenas "Infinity" o "-Infinity" en los vectores de `bounds`.
- Emplea numeros reales en notacion decimal y usa null donde falten datos.
La respuesta DEBE ser unicamente el objeto JSON crudo: comienza en `{` y termina en `}`, sin texto adicional, sin bloques de codigo, sin ```json, sin encabezados ni explicaciones."#;
const GEMINI_USER_PROMPT: &str =
    "Analiza la imagen adjunta, interpreta el problema matematico y responde UNICAMENTE con JSON valido listo para cvxrs.";

const FONT_REGULAR: &str = "plus-jakarta-regular";
const FONT_SEMIBOLD: &str = "plus-jakarta-semibold";

struct Palette;

impl Palette {
    fn background() -> Color32 {
        Color32::from_rgb(11, 15, 24)
    }

    fn top_panel() -> Color32 {
        Color32::from_rgb(16, 21, 34)
    }

    fn main_panel() -> Color32 {
        Color32::from_rgb(19, 26, 38)
    }

    fn surface() -> Color32 {
        Color32::from_rgb(24, 32, 48)
    }

    fn surface_alt() -> Color32 {
        Color32::from_rgb(27, 36, 54)
    }

    fn surface_highlight() -> Color32 {
        Color32::from_rgb(30, 42, 62)
    }

    fn border_soft() -> Color32 {
        Color32::from_rgb(46, 58, 82)
    }

    fn border_strong() -> Color32 {
        Color32::from_rgb(58, 70, 96)
    }

    fn button_idle() -> Color32 {
        Color32::from_rgb(36, 46, 68)
    }

    fn button_hover() -> Color32 {
        Color32::from_rgb(46, 58, 82)
    }

    fn button_active() -> Color32 {
        Color32::from_rgb(54, 68, 96)
    }

    fn accent_gold() -> Color32 {
        Color32::from_rgb(206, 176, 112)
    }

    fn accent_gold_soft() -> Color32 {
        Color32::from_rgb(224, 192, 128)
    }

    fn accent_azure() -> Color32 {
        Color32::from_rgb(154, 200, 236)
    }

    fn accent_lavender() -> Color32 {
        Color32::from_rgb(198, 170, 226)
    }

    fn accent_mint() -> Color32 {
        Color32::from_rgb(160, 220, 204)
    }

    fn text_primary() -> Color32 {
        Color32::from_rgb(230, 234, 242)
    }

    fn text_secondary() -> Color32 {
        Color32::from_rgb(192, 198, 212)
    }

    fn text_muted() -> Color32 {
        Color32::from_rgb(154, 160, 176)
    }

    fn outline_faint() -> Color32 {
        Color32::from_rgb(34, 42, 60)
    }

    fn banner_info_fg() -> Color32 {
        Color32::from_rgb(154, 203, 240)
    }

    fn banner_info_bg() -> Color32 {
        Color32::from_rgb(22, 36, 54)
    }

    fn banner_success_fg() -> Color32 {
        Color32::from_rgb(156, 216, 190)
    }

    fn banner_success_bg() -> Color32 {
        Color32::from_rgb(22, 42, 38)
    }

    fn banner_error_fg() -> Color32 {
        Color32::from_rgb(232, 150, 150)
    }

    fn banner_error_bg() -> Color32 {
        Color32::from_rgb(44, 26, 34)
    }

    fn status_optimal() -> Color32 {
        Color32::from_rgb(168, 226, 198)
    }

    fn status_warning() -> Color32 {
        Color32::from_rgb(240, 196, 140)
    }

    fn status_error() -> Color32 {
        Color32::from_rgb(236, 148, 148)
    }

    fn hyperlink() -> Color32 {
        Color32::from_rgb(156, 210, 236)
    }

    fn shadow_color() -> Color32 {
        Color32::from_black_alpha(110)
    }
}

fn main() -> Result<()> {
    install_tracing();

    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(900.0, 640.0)),
        ..Default::default()
    };

    eframe::run_native(
        "cvxrs Studio",
        native_options,
        Box::new(|cc| Box::new(CvxrsApp::new(cc))),
    )?;
    Ok(())
}

fn install_tracing() {
    #[cfg(debug_assertions)]
    {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init()
            .ok();
    }
    #[cfg(not(debug_assertions))]
    {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::ERROR)
            .without_time()
            .try_init()
            .ok();
    }
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        FONT_REGULAR.to_owned(),
        FontData::from_static(include_bytes!(
            "../assets/fonts/PlusJakartaSans-Regular.ttf"
        )),
    );
    fonts.font_data.insert(
        FONT_SEMIBOLD.to_owned(),
        FontData::from_static(include_bytes!(
            "../assets/fonts/PlusJakartaSans-SemiBold.ttf"
        )),
    );

    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, FONT_REGULAR.to_owned());
    fonts.families.insert(
        FontFamily::Name(FONT_SEMIBOLD.into()),
        vec![FONT_SEMIBOLD.to_owned()],
    );

    ctx.set_fonts(fonts);
}

fn small_shadow() -> egui::epaint::Shadow {
    egui::epaint::Shadow {
        offset: egui::vec2(0.0, 6.0),
        blur: 18.0,
        spread: 0.0,
        color: Palette::shadow_color(),
    }
}

#[derive(Clone, Copy)]
struct SectionStyle {
    fill: Color32,
    border: Color32,
    accent: Color32,
}

impl SectionStyle {
    fn problem() -> Self {
        Self {
            fill: Palette::surface(),
            border: Palette::border_strong(),
            accent: Palette::accent_gold(),
        }
    }

    fn gemini() -> Self {
        Self {
            fill: Palette::surface_alt(),
            border: Palette::border_soft(),
            accent: Palette::accent_azure(),
        }
    }

    fn quick_start() -> Self {
        Self {
            fill: Palette::surface_highlight(),
            border: Palette::border_strong(),
            accent: Palette::accent_lavender(),
        }
    }

    fn status() -> Self {
        Self {
            fill: Palette::surface_alt(),
            border: Palette::border_soft(),
            accent: Palette::accent_mint(),
        }
    }
}

fn section_card<R>(
    ui: &mut egui::Ui,
    style: SectionStyle,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let inner = egui::Frame::group(ui.style())
        .fill(style.fill)
        .stroke(Stroke::new(1.4, style.border))
        .rounding(12.0)
        .inner_margin(Margin::symmetric(20.0, 18.0))
        .outer_margin(Margin::symmetric(0.0, 6.0))
        .shadow(small_shadow())
        .show(ui, add_contents);

    let stripe_area = inner.response.rect.shrink2(egui::vec2(3.0, 6.0));
    let painter = ui.painter_at(inner.response.rect);
    let accent_width = 4.0;

    let left_stripe = egui::Rect::from_min_max(
        egui::pos2(stripe_area.left(), stripe_area.top()),
        egui::pos2(stripe_area.left() + accent_width, stripe_area.bottom()),
    );
    let right_stripe = egui::Rect::from_min_max(
        egui::pos2(stripe_area.right() - accent_width, stripe_area.top()),
        egui::pos2(stripe_area.right(), stripe_area.bottom()),
    );

    painter.rect_filled(left_stripe, 3.0, style.accent);
    painter.rect_filled(right_stripe, 3.0, style.accent);

    inner
}

#[derive(Clone)]
enum TaskState {
    Idle,
    Running,
    Success(SolveSummary),
    Failure(String),
}

impl Default for TaskState {
    fn default() -> Self {
        Self::Idle
    }
}

impl TaskState {
    fn is_running(&self) -> bool {
        matches!(self, TaskState::Running)
    }
}

#[derive(Clone)]
enum GeminiState {
    Idle,
    Running,
    Success(String),
    Failure(String),
}

impl Default for GeminiState {
    fn default() -> Self {
        GeminiState::Idle
    }
}

impl GeminiState {
    fn is_running(&self) -> bool {
        matches!(self, GeminiState::Running)
    }
}

#[derive(Clone)]
struct SolveSummary {
    method: MethodChoice,
    problem_path: PathBuf,
    output_path: Option<PathBuf>,
    solution: Solution<Scalar>,
    solution_json: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MethodChoice {
    Admm,
    Ipm,
}

impl MethodChoice {
    fn all() -> [MethodChoice; 2] {
        [MethodChoice::Admm, MethodChoice::Ipm]
    }

    fn display_name(self) -> &'static str {
        match self {
            MethodChoice::Admm => "ADMM (robust & warm-start friendly)",
            MethodChoice::Ipm => "IPM (fast interior-point method)",
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            MethodChoice::Admm => "ADMM",
            MethodChoice::Ipm => "IPM",
        }
    }

    fn to_method(self) -> Method {
        match self {
            MethodChoice::Admm => Method::Admm,
            MethodChoice::Ipm => Method::Ipm,
        }
    }
}

#[derive(Clone)]
struct BannerMessage {
    text: String,
    kind: BannerKind,
}

#[derive(Clone, Copy)]
enum BannerKind {
    Info,
    Success,
    Error,
}

impl BannerKind {
    fn color(self) -> Color32 {
        match self {
            BannerKind::Info => Palette::banner_info_fg(),
            BannerKind::Success => Palette::banner_success_fg(),
            BannerKind::Error => Palette::banner_error_fg(),
        }
    }

    fn background(self) -> Color32 {
        match self {
            BannerKind::Info => Palette::banner_info_bg(),
            BannerKind::Success => Palette::banner_success_bg(),
            BannerKind::Error => Palette::banner_error_bg(),
        }
    }
}

struct CvxrsApp {
    problem_path: Option<PathBuf>,
    problem_input: String,
    output_path: Option<PathBuf>,
    output_input: String,
    method: MethodChoice,
    tolerance_input: String,
    max_iters_input: String,
    time_limit_input: String,
    write_solution: bool,
    log_json: bool,
    last_problem_dir: Option<PathBuf>,
    last_output_dir: Option<PathBuf>,
    banner: Option<BannerMessage>,
    task_state: Arc<Mutex<TaskState>>,
    gemini_image_path: Option<PathBuf>,
    gemini_image_input: String,
    gemini_last_image_dir: Option<PathBuf>,
    gemini_state: Arc<Mutex<GeminiState>>,
    gemini_last_export_dir: Option<PathBuf>,
}

impl CvxrsApp {
    fn new(cc: &CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);

        let default_dir = env::current_dir().ok();

        Self {
            problem_path: None,
            problem_input: String::new(),
            output_path: None,
            output_input: String::new(),
            method: MethodChoice::Admm,
            tolerance_input: String::new(),
            max_iters_input: String::new(),
            time_limit_input: String::new(),
            write_solution: false,
            log_json: false,
            last_problem_dir: default_dir.clone(),
            last_output_dir: default_dir.clone(),
            banner: Some(BannerMessage {
                text: "Paso 1: selecciona o carga un problema en formato JSON para comenzar."
                    .into(),
                kind: BannerKind::Info,
            }),
            task_state: Arc::new(Mutex::new(TaskState::Idle)),
            gemini_image_path: None,
            gemini_image_input: String::new(),
            gemini_last_image_dir: default_dir.clone(),
            gemini_state: Arc::new(Mutex::new(GeminiState::Idle)),
            gemini_last_export_dir: default_dir,
        }
    }

    fn is_busy(&self) -> bool {
        matches!(
            *self.task_state.lock().expect("task state poisoned"),
            TaskState::Running
        )
    }

    fn is_gemini_busy(&self) -> bool {
        matches!(
            *self.gemini_state.lock().expect("gemini state poisoned"),
            GeminiState::Running
        )
    }

    fn start_solve(&mut self, ctx: egui::Context) {
        if self.is_busy() {
            return;
        }

        if self.problem_path.is_none() {
            if let Some(candidate) = Self::path_from_input(&self.problem_input) {
                self.problem_path = Some(candidate);
            }
        }

        let problem_path = match self.problem_path.clone() {
            Some(path) => path,
            None => {
                self.set_failure("Selecciona un archivo de problema antes de resolver.");
                ctx.request_repaint();
                return;
            }
        };

        let tolerance = match parse_optional_f64(&self.tolerance_input) {
            Ok(value) => value,
            Err(err) => {
                self.set_failure(format!("Tolerancia invalida: {}", err));
                ctx.request_repaint();
                return;
            }
        };

        let max_iters = match parse_optional_usize(&self.max_iters_input) {
            Ok(value) => value,
            Err(err) => {
                self.set_failure(format!("Iteraciones maximas invalidas: {}", err));
                ctx.request_repaint();
                return;
            }
        };

        let time_limit = match parse_optional_u64(&self.time_limit_input) {
            Ok(value) => value,
            Err(err) => {
                self.set_failure(format!("Limite de tiempo invalido: {}", err));
                ctx.request_repaint();
                return;
            }
        };

        let output_path = if self.write_solution {
            if self.output_path.is_none() {
                let trimmed = self.output_input.trim();
                if !trimmed.is_empty() {
                    self.output_path = Some(PathBuf::from(trimmed));
                }
            }
            match self.output_path.clone() {
                Some(path) => Some(path),
                None => {
                    self.set_failure("Selecciona un archivo de salida para guardar la solucion.");
                    ctx.request_repaint();
                    return;
                }
            }
        } else {
            None
        };

        {
            let mut state = self.task_state.lock().expect("task state poisoned");
            *state = TaskState::Running;
        }

        self.set_banner(
            BannerKind::Info,
            format!(
                "Resolviendo {:?} con el metodo {}...",
                problem_path.file_name().unwrap_or_default(),
                self.method.short_name()
            ),
        );
        ctx.request_repaint();

        let task_state = self.task_state.clone();
        let method = self.method;
        let log_json = self.log_json;
        let problem_path_clone = problem_path.clone();
        let output_path_clone = output_path.clone();

        std::thread::spawn(move || {
            let result = solve_problem(
                method,
                problem_path_clone.clone(),
                tolerance,
                max_iters,
                time_limit,
                output_path_clone.clone(),
                log_json,
            );

            let mut state = task_state.lock().expect("task state poisoned");
            *state = match result {
                Ok((solution, solution_json)) => TaskState::Success(SolveSummary {
                    method,
                    problem_path: problem_path_clone,
                    output_path: output_path_clone,
                    solution,
                    solution_json,
                }),
                Err(err) => TaskState::Failure(err.to_string()),
            };
            drop(state);
            ctx.request_repaint();
        });
    }

    fn start_gemini_conversion(&mut self, ctx: egui::Context) {
        if self.is_gemini_busy() {
            return;
        }

        if self.gemini_image_path.is_none() {
            if let Some(candidate) = Self::path_from_input(&self.gemini_image_input) {
                self.gemini_image_path = Some(candidate);
            }
        }

        let image_path = match self.gemini_image_path.clone() {
            Some(path) => path,
            None => {
                self.set_banner(
                    BannerKind::Error,
                    "Selecciona una imagen antes de convertirla a JSON.",
                );
                ctx.request_repaint();
                return;
            }
        };

        if !image_path.exists() {
            self.set_banner(
                BannerKind::Error,
                "No pudimos encontrar la imagen indicada. Revisa la ruta.",
            );
            ctx.request_repaint();
            return;
        }

        {
            let mut state = self.gemini_state.lock().expect("gemini state poisoned");
            *state = GeminiState::Running;
        }
        ctx.request_repaint();

        let gemini_state = self.gemini_state.clone();
        let image_path_clone = image_path.clone();

        std::thread::spawn(move || {
            let result = convert_image_with_gemini(&image_path_clone);
            let mut state = gemini_state.lock().expect("gemini state poisoned");
            *state = match result {
                Ok(json) => GeminiState::Success(json),
                Err(err) => GeminiState::Failure(err.to_string()),
            };
            drop(state);
            ctx.request_repaint();
        });
    }

    fn set_failure(&mut self, message: impl Into<String>) {
        let message = message.into();
        {
            let mut state = self.task_state.lock().expect("task state poisoned");
            *state = TaskState::Failure(message.clone());
        }
        self.set_banner(BannerKind::Error, message);
    }

    fn set_success(&mut self, message: impl Into<String>) {
        self.set_banner(BannerKind::Success, message);
    }

    fn set_banner(&mut self, kind: BannerKind, message: impl Into<String>) {
        self.banner = Some(BannerMessage {
            text: message.into(),
            kind,
        });
    }

    fn clear_banner(&mut self) {
        self.banner = None;
    }

    fn path_from_input(input: &str) -> Option<PathBuf> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        let candidate = Path::new(trimmed);
        if candidate.exists() {
            Some(candidate.to_path_buf())
        } else {
            None
        }
    }
    fn handle_file_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        if let Some(path) = dropped.into_iter().next() {
            self.on_problem_selected(path);
            self.set_success("Problema cargado desde un archivo arrastrado.");
        }
    }

    fn pick_problem_file(&mut self) -> Option<PathBuf> {
        let mut dialog = FileDialog::new();
        if let Some(dir) = &self.last_problem_dir {
            dialog = dialog.set_directory(dir);
        }
        dialog
            .add_filter("Problemas JSON", &["json", "JSON"])
            .set_title("Selecciona un problema en formato JSON")
            .pick_file()
    }

    fn pick_output_file(&mut self) -> Option<PathBuf> {
        let mut dialog = FileDialog::new();
        if let Some(dir) = &self.last_output_dir {
            dialog = dialog.set_directory(dir);
        }
        dialog
            .add_filter("JSON", &["json", "JSON"])
            .set_title("Elige donde guardar la solucion")
            .set_file_name("solucion.json")
            .save_file()
    }

    fn pick_image_file(&mut self) -> Option<PathBuf> {
        let mut dialog = FileDialog::new();
        if let Some(dir) = &self.gemini_last_image_dir {
            dialog = dialog.set_directory(dir);
        }
        dialog
            .add_filter("Imagenes", &["png", "jpg", "jpeg", "bmp", "webp"])
            .set_title("Selecciona la imagen del problema")
            .pick_file()
    }

    fn browse_problem(&mut self) {
        if let Some(path) = self.pick_problem_file() {
            self.on_problem_selected(path);
            self.set_success("Archivo de problema listo.");
        }
    }

    fn browse_output(&mut self) {
        if let Some(path) = self.pick_output_file() {
            self.on_output_selected(path);
            self.set_success("Ruta de salida configurada.");
        }
    }

    fn browse_gemini_image(&mut self) {
        if let Some(path) = self.pick_image_file() {
            self.on_gemini_image_selected(path);
        }
    }

    fn on_problem_selected(&mut self, path: PathBuf) {
        if path.exists() {
            self.problem_input = path.display().to_string();
            self.problem_path = Some(path.clone());
            self.last_problem_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    fn on_output_selected(&mut self, path: PathBuf) {
        if let Some(parent) = path.parent() {
            if parent.exists() {
                self.last_output_dir = Some(parent.to_path_buf());
            }
        }
        self.output_input = path.display().to_string();
        self.output_path = Some(path);
    }

    fn on_gemini_image_selected(&mut self, path: PathBuf) {
        if path.exists() {
            self.gemini_image_input = path.display().to_string();
            self.gemini_image_path = Some(path.clone());
            self.gemini_last_image_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    fn save_gemini_json(&mut self, json: &str) {
        let mut dialog = FileDialog::new()
            .add_filter("JSON", &["json", "JSON"])
            .set_title("Guardar JSON generado por Gemini")
            .set_file_name("problema_gemini.json");
        if let Some(dir) = &self.gemini_last_export_dir {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.save_file() {
            if let Some(parent) = path.parent() {
                if parent.exists() {
                    self.gemini_last_export_dir = Some(parent.to_path_buf());
                }
            }
            match fs::write(&path, json) {
                Ok(_) => self.set_banner(
                    BannerKind::Success,
                    format!("JSON guardado en {}", path.display()),
                ),
                Err(err) => self.set_banner(
                    BannerKind::Error,
                    format!("No se pudo guardar el JSON: {}", err),
                ),
            }
        }
    }

    fn load_example_problem(&mut self) {
        match self.write_example_problem() {
            Ok(path) => {
                self.on_problem_selected(path.clone());
                self.set_success("Ejemplo de prueba listo. Puedes resolverlo de inmediato.");
                if let Ok(mut state) = self.task_state.lock() {
                    if !matches!(*state, TaskState::Running) {
                        *state = TaskState::Idle;
                    }
                }
            }
            Err(err) => self.set_failure(format!("No se pudo preparar el ejemplo: {}", err)),
        }
    }

    fn write_example_problem(&self) -> Result<PathBuf> {
        serde_json::from_str::<JsonProblem>(SAMPLE_QP_JSON)
            .map_err(|err| anyhow!("Ejemplo interno invalido: {}", err))?;

        let dir = env::temp_dir().join("cvxrs-studio");
        fs::create_dir_all(&dir)?;
        let path = dir.join("ejemplo_qp.json");
        fs::write(&path, SAMPLE_QP_JSON)?;
        Ok(path)
    }

    fn render_banner(&mut self, ui: &mut egui::Ui) {
        if let Some(banner) = self.banner.clone() {
            egui::Frame::none()
                .fill(banner.kind.background())
                .rounding(8.0)
                .inner_margin(Margin::symmetric(16.0, 12.0))
                .show(ui, |ui| {
                    let width = ui.available_width();
                    ui.set_width(width);
                    let button_width = 90.0;
                    ui.horizontal(|ui| {
                        let label_width =
                            (width - button_width - ui.spacing().item_spacing.x).max(0.0);
                        let label = egui::Label::new(
                            RichText::new(banner.text.clone())
                                .color(banner.kind.color())
                                .size(16.0)
                                .line_height(Some(20.0)),
                        )
                        .wrap(true);
                        ui.add_sized([label_width, 0.0], label);
                        if ui
                            .add_sized(
                                [button_width, ui.spacing().interact_size.y * 1.1],
                                egui::Button::new(
                                    RichText::new("Cerrar")
                                        .color(Palette::text_primary())
                                        .text_style(TextStyle::Button),
                                )
                                .stroke(Stroke::new(1.0, Palette::border_soft()))
                                .fill(Palette::button_idle()),
                            )
                            .clicked()
                        {
                            self.clear_banner();
                        }
                    });
                });
        }
    }

    fn render_problem_section(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, busy: bool) {
        section_card(ui, SectionStyle::problem(), |ui| {
            ui.vertical(|ui| {
                ui.heading(
                    RichText::new("1. Prepara tu problema")
                        .color(Palette::text_primary())
                        .text_style(TextStyle::Heading),
                );
                ui.label(
                    RichText::new(
                        "Abre un archivo JSON compatible o carga el ejemplo incluido para probar.",
                    )
                    .color(Palette::text_secondary()),
                );
                ui.add_space(10.0);

                let display_text = if self.problem_input.trim().is_empty() {
                    String::from("Selecciona un archivo JSON...")
                    } else {
                        self.problem_input.clone()
                    };

                    let mut open_problem_dialog = false;
                    let picker_button = egui::Button::new(
                        RichText::new(display_text.as_str())
                            .color(Palette::text_secondary())
                            .monospace(),
                    )
                    .stroke(Stroke::new(1.0, Palette::outline_faint()))
                    .fill(Palette::surface_alt())
                    .min_size(egui::vec2(ui.available_width(), 40.0))
                    .wrap(true);
                    if ui.add(picker_button).clicked() {
                        open_problem_dialog = true;
                    }

                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Examinar...")
                                        .color(Palette::text_primary())
                                        .text_style(TextStyle::Button),
                                )
                                .stroke(Stroke::new(1.0, Palette::border_soft()))
                                .min_size(egui::vec2(140.0, 36.0)),
                            )
                            .clicked()
                        {
                            open_problem_dialog = true;
                        }
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Usar ejemplo guiado")
                                        .color(Palette::accent_gold())
                                        .text_style(TextStyle::Button),
                                )
                                .fill(Color32::from_rgba_unmultiplied(206, 176, 112, 45))
                                .stroke(Stroke::new(1.0, Palette::accent_gold()))
                                .min_size(egui::vec2(178.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.load_example_problem();
                        }
                    });

                    if open_problem_dialog {
                        self.browse_problem();
                    }

                    ui.add_space(6.0);
                    ui.small(
                        "Tambien puedes arrastrar un JSON sobre la ventana o pegar la ruta manualmente.",
                    );

                    ui.collapsing("Editar ruta manualmente", |ui| {
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut self.problem_input)
                                    .hint_text("Ej: C:\\datos\\mi_problema.json")
                                    .desired_width(f32::INFINITY),
                            )
                            .changed()
                        {
                            self.problem_path = Self::path_from_input(&self.problem_input);
                        }
                    });

                    if let Some(path) = &self.problem_path {
                        ui.label(
                            RichText::new(format!("Seleccionado: {}", path.display()))
                                .color(Palette::text_secondary()),
                        );
                    } else if !self.problem_input.trim().is_empty() {
                        ui.colored_label(
                            Palette::status_error(),
                            "No encontramos ese archivo. Revisa la ruta o usa Examinar.",
                        );
                    }

                    ui.add_space(16.0);
                    ui.heading(
                        RichText::new("2. Configura el solucionador")
                            .color(Palette::text_primary())
                            .size(20.0),
                    );

                    egui::ComboBox::from_id_source("method_combo")
                        .width(260.0)
                        .selected_text(self.method.display_name())
                        .show_ui(ui, |combo| {
                            for option in MethodChoice::all() {
                                combo.selectable_value(
                                    &mut self.method,
                                    option,
                                    option.display_name(),
                                );
                            }
                        });

                    ui.label(
                        RichText::new(match self.method {
                            MethodChoice::Admm => "ADMM: estable y admite warm-start.",
                            MethodChoice::Ipm => "IPM: rapido en problemas bien condicionados.",
                        })
                        .size(14.0)
                        .color(Palette::text_muted()),
                    );

                    ui.add_space(12.0);
                    ui.columns(3, |columns| {
                        columns[0].vertical(|ui| {
                            ui.label("Tolerancia");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.tolerance_input)
                                    .hint_text("Ej: 1e-6"),
                            );
                        });
                        columns[1].vertical(|ui| {
                            ui.label("Iteraciones maximas");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.max_iters_input)
                                    .hint_text("Ej: 1000"),
                            );
                        });
                        columns[2].vertical(|ui| {
                            ui.label("Tiempo max (s)");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.time_limit_input)
                                    .hint_text("Ej: 60"),
                            );
                        });
                    });

                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.log_json, "Mostrar solucion como JSON");
                        ui.checkbox(&mut self.write_solution, "Guardar solucion en archivo");
                    });

                    if self.write_solution {
                        ui.add_space(6.0);

                        let output_display = if self.output_input.trim().is_empty() {
                            String::from("Selecciona un archivo para guardar...")
                        } else {
                            self.output_input.clone()
                        };

                        let mut open_output_dialog = false;
                        let picker_button = egui::Button::new(
                            RichText::new(output_display.as_str())
                                .color(Palette::text_secondary())
                                .monospace(),
                        )
                        .stroke(Stroke::new(1.0, Palette::outline_faint()))
                        .fill(Palette::surface_alt())
                        .min_size(egui::vec2(ui.available_width(), 40.0))
                        .wrap(true);
                        if ui.add(picker_button).clicked() {
                            open_output_dialog = true;
                        }

                        ui.add_space(8.0);
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Elegir destino...")
                                        .color(Palette::text_primary())
                                        .text_style(TextStyle::Button),
                                )
                                .stroke(Stroke::new(1.0, Palette::border_soft()))
                                .min_size(egui::vec2(156.0, 36.0)),
                            )
                            .clicked()
                        {
                            open_output_dialog = true;
                        }

                        if open_output_dialog {
                            self.browse_output();
                        }

                        ui.collapsing("Editar ruta de salida manualmente", |ui| {
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.output_input)
                                        .hint_text("Ej: C:\\datos\\solucion.json")
                                        .desired_width(f32::INFINITY),
                                )
                                .changed()
                            {
                                let trimmed = self.output_input.trim();
                                if trimmed.is_empty() {
                                    self.output_path = None;
                                } else {
                                    self.output_path = Some(PathBuf::from(trimmed));
                                }
                            }
                        });
                    }

                    ui.add_space(18.0);
                    let button_label = if busy { "Resolviendo..." } else { "Resolver problema" };
                    let button = egui::Button::new(
                        RichText::new(button_label)
                            .color(if busy {
                                Palette::text_muted()
                            } else {
                                Palette::text_primary()
                            })
                            .text_style(TextStyle::Button),
                    )
                    .stroke(Stroke::new(1.2, Palette::accent_gold()))
                    .fill(if busy {
                        Palette::button_idle()
                    } else {
                        Palette::accent_gold()
                    })
                    .min_size(egui::vec2(228.0, 46.0))
                    .rounding(12.0);

                    if ui.add_enabled(!busy, button).clicked() {
                        self.start_solve(ctx.clone());
                    }

                    if busy {
                        ui.add_space(6.0);
                        ui.spinner();
                        ui.label(
                            RichText::new("Ejecutando solver...")
                                .color(Palette::text_secondary()),
                        );
                    }
                });
        });
    }

    fn render_gemini_section(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let gemini_state_snapshot = self
            .gemini_state
            .lock()
            .expect("gemini state poisoned")
            .clone();
        let busy = gemini_state_snapshot.is_running();

        section_card(ui, SectionStyle::gemini(), |ui| {
            ui.set_width(ui.available_width());
            ui.vertical(|ui| {
                ui.heading(
                    RichText::new("Convertir imagen a JSON con Gemini")
                        .color(Palette::accent_azure())
                        .size(21.0),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Reconoce un problema escrito a mano y obtén inmediatamente el JSON compatible con cvxrs.",
                    )
                    .color(Palette::text_secondary()),
                );

                    ui.add_space(18.0);
                    let mut open_dialog = false;
                    let display_text = if self.gemini_image_input.trim().is_empty() {
                        String::from("Selecciona una imagen o arrástrala aquí…")
                    } else {
                        self.gemini_image_input.clone()
                    };

                    let available_width = ui.available_width();
                    ui.horizontal(|ui| {
                        let selector = egui::Button::new(
                            RichText::new(display_text.as_str())
                                .color(Palette::text_secondary())
                                .monospace(),
                        )
                        .stroke(Stroke::new(1.0, Palette::outline_faint()))
                        .fill(Palette::surface_alt())
                        .rounding(10.0)
                        .min_size(egui::vec2((available_width - 190.0).max(220.0), 44.0))
                        .wrap(true);
                        if ui.add(selector).clicked() {
                            open_dialog = true;
                        }

                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Examinar…")
                                        .color(Palette::accent_azure())
                                        .text_style(TextStyle::Button),
                                )
                                .stroke(Stroke::new(1.1, Palette::accent_azure()))
                                .rounding(10.0)
                                .fill(Color32::from_rgba_unmultiplied(154, 200, 236, 45))
                                .min_size(egui::vec2(164.0, 44.0)),
                            )
                            .clicked()
                        {
                            open_dialog = true;
                        }
                    });

                    if open_dialog {
                        self.browse_gemini_image();
                    }

                    ui.add_space(10.0);
                    ui.with_layout(egui::Layout::left_to_right(Align::Center), |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.gemini_image_input)
                                .hint_text("Ej: C:\\imagenes\\problema.png")
                                .desired_width((available_width - 160.0).max(240.0))
                                .font(TextStyle::Monospace),
                        );
                        ui.add_space(12.0);
                        let convert_button = egui::Button::new(
                            RichText::new(if busy {
                                "Convirtiendo…"
                            } else {
                                "Convertir a JSON"
                            })
                            .color(if busy {
                                Palette::text_muted()
                            } else {
                                Palette::text_primary()
                            })
                            .text_style(TextStyle::Button),
                        )
                        .rounding(12.0)
                        .fill(if busy {
                            Palette::button_idle()
                        } else {
                            Palette::accent_azure()
                        })
                        .stroke(Stroke::new(1.2, Palette::accent_azure()))
                        .min_size(egui::vec2(180.0, 46.0));

                        if ui.add_enabled(!busy, convert_button).clicked() {
                            self.start_gemini_conversion(ctx.clone());
                        }
                    });

                    if let Some(path) = &self.gemini_image_path {
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(format!("Imagen seleccionada: {}", path.display()))
                                .monospace()
                                .color(Palette::text_secondary()),
                        );
                    }

                    ui.add_space(18.0);
                    egui::Frame::none()
                        .fill(Palette::surface())
                        .stroke(Stroke::new(1.0, Palette::border_soft()))
                        .rounding(10.0)
                        .inner_margin(Margin::symmetric(18.0, 16.0))
                        .show(ui, |ui| {
                            ui.set_min_height(240.0);
                            match gemini_state_snapshot {
                                GeminiState::Idle => {
                                    ui.vertical_centered(|ui| {
                                        ui.label(
                                            RichText::new(
                                                "Carga una imagen para obtener la representación JSON del problema.",
                                            )
                                            .color(Palette::text_secondary()),
                                        );
                                        ui.label(
                                            RichText::new(
                                                "Gemini analizará variables, restricciones y parámetros automáticamente.",
                                            )
                                            .color(Palette::text_muted()),
                                        );
                                    });
                                }
                                GeminiState::Running => {
                                    ui.vertical_centered(|ui| {
                                        ui.spinner();
                                        ui.add_space(8.0);
                                        ui.label(
                                            RichText::new(
                                                "Convirtiendo la imagen con Gemini. Esto puede tardar unos segundos…",
                                            )
                                            .color(Palette::accent_azure()),
                                        );
                                    });
                                }
                                GeminiState::Failure(message) => {
                                    ui.vertical_centered(|ui| {
                                        ui.colored_label(
                                            Palette::status_error(),
                                            format!(
                                                "No se pudo generar el JSON: {}",
                                                message
                                            ),
                                        );
                                    });
                                }
                                GeminiState::Success(json) => {
                                    ui.label(
                                        RichText::new("JSON generado")
                                            .color(Palette::accent_azure())
                                            .size(17.0),
                                    );
                                    ui.add_space(8.0);
                                    let mut json_preview = json.clone();
                                    let editor = egui::TextEdit::multiline(&mut json_preview)
                                        .font(TextStyle::Monospace)
                                        .desired_rows(18)
                                        .desired_width(f32::INFINITY)
                                        .interactive(false)
                                        .frame(true);
                                    ui.add(editor);
                                    ui.add_space(10.0);
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("Guardar JSON en archivo")
                                                    .color(Palette::text_primary())
                                                    .text_style(TextStyle::Button),
                                            )
                                            .rounding(10.0)
                                            .stroke(Stroke::new(1.1, Palette::accent_azure()))
                                            .fill(Color32::from_rgba_unmultiplied(154, 200, 236, 60))
                                            .min_size(egui::vec2(220.0, 40.0)),
                                        )
                                        .clicked()
                                    {
                                        self.save_gemini_json(&json);
                                    }
                                }
                            }
                        });
                });
        });
    }

    fn render_quick_start(&mut self, ui: &mut egui::Ui) {
        section_card(ui, SectionStyle::quick_start(), |ui| {
            ui.vertical(|ui| {
                ui.heading(RichText::new("Guia rapida").color(Palette::text_primary()));
                ui.label("- Exporta tu problema en JSON usando el esquema de cvxrs.");
                ui.label("- Carga el archivo desde Examinar o arrastralo sobre la ventana.");
                ui.label("- Ajusta tolerancia y limites solo si lo necesitas.");
                ui.label("- Pulsa Resolver y revisa el resumen inferior.");

                ui.add_space(10.0);
                ui.label(RichText::new("Ejemplo incluido").color(Palette::text_secondary()));
                ui.label(SAMPLE_DESCRIPTION);

                ui.collapsing("Ver JSON del ejemplo", |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(200.0)
                        .show(ui, |ui| {
                            ui.monospace(SAMPLE_QP_JSON.trim());
                        });
                });
            });
        });
    }

    fn render_status(&mut self, ui: &mut egui::Ui, state: &TaskState) {
        section_card(ui, SectionStyle::status(), |ui| match state {
            TaskState::Idle => {
                ui.label(
                    RichText::new("Listo para resolver. Carga un problema para comenzar.")
                        .color(Palette::text_secondary()),
                );
            }
            TaskState::Running => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        RichText::new("Trabajando en la solucion...")
                            .color(Palette::text_secondary()),
                    );
                });
            }
            TaskState::Failure(message) => {
                ui.colored_label(Palette::status_error(), format!("Error: {}", message));
            }
            TaskState::Success(summary) => {
                render_solution_summary(ui, summary);
            }
        });
    }
}

impl App for CvxrsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.handle_file_drops(ctx);

        let state = self.task_state.lock().expect("task state poisoned").clone();

        let busy = state.is_running();

        egui::TopBottomPanel::top("app_header")
            .frame(egui::Frame::none().fill(Palette::top_panel()))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(10.0);
                    ui.heading(RichText::new("cvxrs Studio").color(Palette::accent_gold()));
                    ui.label(
                        RichText::new(
                            "Interfaz guiada para resolver programas convexos en Windows.",
                        )
                        .color(Palette::text_secondary()),
                    );
                    ui.add_space(6.0);
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Palette::main_panel()))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add_space(12.0);
                        if self.banner.is_some() {
                            self.render_banner(ui);
                            ui.add_space(12.0);
                        }
                        self.render_gemini_section(ctx, ui);
                        ui.add_space(12.0);
                        self.render_problem_section(ctx, ui, busy);
                        ui.add_space(12.0);
                        self.render_quick_start(ui);
                        ui.add_space(12.0);
                        self.render_status(ui, &state);
                    });
            });
    }
}

fn configure_style(ctx: &egui::Context) {
    install_fonts(ctx);

    let mut visuals = egui::Visuals::dark();
    visuals.window_rounding = egui::Rounding::same(14.0);
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 12.0),
        blur: 28.0,
        spread: 0.0,
        color: Palette::shadow_color(),
    };
    visuals.window_fill = Palette::surface_alt();
    visuals.panel_fill = Palette::main_panel();
    visuals.extreme_bg_color = Palette::background();
    visuals.faint_bg_color = Palette::outline_faint();
    visuals.hyperlink_color = Palette::hyperlink();
    visuals.widgets.noninteractive.bg_fill = Palette::top_panel();
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Palette::text_secondary());
    visuals.widgets.inactive.bg_fill = Palette::button_idle();
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.1, Palette::text_primary());
    visuals.widgets.hovered.bg_fill = Palette::button_hover();
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.1, Palette::accent_gold());
    visuals.widgets.active.bg_fill = Palette::button_active();
    visuals.widgets.active.fg_stroke = Stroke::new(1.2, Palette::accent_gold_soft());
    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(206, 176, 112, 70);
    visuals.selection.stroke = Stroke::new(1.2, Palette::accent_gold());
    ctx.set_visuals(visuals.clone());

    let mut style = (*ctx.style()).clone();
    style.visuals = visuals;
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(26.0, FontFamily::Name(FONT_SEMIBOLD.into())),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(17.0, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(16.0, FontFamily::Name(FONT_SEMIBOLD.into())),
    );
    style
        .text_styles
        .insert(TextStyle::Monospace, FontId::monospace(15.5));
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(14.0, FontFamily::Proportional),
    );
    style.spacing.item_spacing = egui::vec2(16.0, 20.0);
    style.spacing.button_padding = egui::vec2(18.0, 14.0);
    style.spacing.window_margin = Margin::symmetric(22.0, 18.0);
    style.spacing.indent = 18.0;
    style.animation_time = 0.18;
    style.visuals.override_text_color = Some(Palette::text_primary());
    ctx.set_style(style);
}

fn render_solution_summary(ui: &mut egui::Ui, summary: &SolveSummary) {
    let solution = &summary.solution;
    let card_width = ui.available_width();
    egui::Frame::group(ui.style())
        .fill(Palette::surface_highlight())
        .stroke(Stroke::new(1.0, Palette::border_strong()))
        .rounding(12.0)
        .inner_margin(Margin::symmetric(20.0, 18.0))
        .shadow(small_shadow())
        .show(ui, |ui| {
            ui.set_width(card_width);
            ui.vertical(|ui| {
                let status_color = match solution.status {
                    Status::Optimal => Palette::status_optimal(),
                    Status::MaxIterations | Status::MaxTime => Palette::status_warning(),
                    _ => Palette::status_error(),
                };
                ui.heading(
                    RichText::new(format!(
                        "{} finalizo con estado {:?}",
                        summary.method.short_name(),
                        solution.status
                    ))
                    .color(status_color),
                );
                ui.label(
                    RichText::new(format!("Archivo: {}", summary.problem_path.display()))
                        .color(Palette::text_secondary()),
                );
                if let Some(path) = &summary.output_path {
                    ui.label(
                        RichText::new(format!("Solucion guardada en: {}", path.display()))
                            .color(Palette::hyperlink()),
                    );
                }

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.label(format!("Iteraciones: {}", solution.iterations));
                    ui.separator();
                    ui.label(format!("Objetivo: {:.6}", solution.objective_value));
                    ui.separator();
                    ui.label(format!(
                        "Tiempo total: {:.2}s",
                        solution.stats.solve_time.as_secs_f64()
                    ));
                    ui.separator();
                    ui.label(format!(
                        "Factorizaciones: {}",
                        solution.stats.factorizations
                    ));
                });

                ui.add_space(12.0);
                egui::CollapsingHeader::new("Ver detalles de la solucion")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        egui::Frame::group(ui.style())
                            .fill(Palette::surface_alt())
                            .stroke(Stroke::new(1.0, Palette::border_soft()))
                            .rounding(10.0)
                            .inner_margin(Margin::symmetric(14.0, 12.0))
                            .show(ui, |ui| {
                                ui.label(format!("Dimension primal: {}", solution.primal.len()));
                                if !solution.primal.is_empty() {
                                    let preview: Vec<String> = solution
                                        .primal
                                        .iter()
                                        .take(8)
                                        .map(|value| format!("{:.6}", value))
                                        .collect();
                                    let suffix = if solution.primal.len() > preview.len() {
                                        ", ..."
                                    } else {
                                        ""
                                    };
                                    ui.label(format!(
                                        "Primal (primeros valores): [{}{}]",
                                        preview.join(", "),
                                        suffix
                                    ));
                                }

                                ui.label(format!(
                                    "Historial de iteraciones: {}",
                                    solution.stats.history.len()
                                ));
                                if let Some(last) = solution.stats.history.last() {
                                    ui.label(format!(
                                        "Ultima iteracion -> prim_inf: {:.3e}, dual_inf: {:.3e}, gap: {:.3e}",
                                        last.primal_residual, last.dual_residual, last.relative_gap
                                    ));
                                }
                            });
                    });

                if let Some(json) = &summary.solution_json {
                    ui.add_space(12.0);
                    egui::CollapsingHeader::new("JSON de la solucion")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            egui::Frame::group(ui.style())
                                .fill(Palette::surface())
                                .stroke(Stroke::new(1.0, Palette::border_soft()))
                                .rounding(10.0)
                                .inner_margin(Margin::symmetric(14.0, 12.0))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    let mut json_preview = json.clone();
                                    let editor = egui::TextEdit::multiline(&mut json_preview)
                                        .font(TextStyle::Monospace)
                                        .desired_rows(18)
                                        .desired_width(f32::INFINITY)
                                        .interactive(false)
                                        .frame(true);
                                    ui.add(editor);
                                });
                        });
                }
            });
        });
}

fn solve_problem(
    method: MethodChoice,
    problem_path: PathBuf,
    tolerance: Option<f64>,
    max_iters: Option<usize>,
    time_limit: Option<u64>,
    output_path: Option<PathBuf>,
    log_json: bool,
) -> Result<(Solution<Scalar>, Option<String>)> {
    tracing::info!(
        ?problem_path,
        ?output_path,
        ?method,
        "starting solve from GUI"
    );

    let mut options = SolveOptions::<Scalar>::default();
    if let Some(tol) = tolerance {
        options.tolerance = tol as Scalar;
    }
    if let Some(iters) = max_iters {
        options.max_iterations = iters;
    }
    if let Some(limit) = time_limit {
        options.max_time = Some(Duration::from_secs(limit));
    }

    let extension = problem_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let mut solver = Solver::<Scalar>::new()
        .method(method.to_method())
        .options(options);
    let solution = match extension.as_str() {
        "json" => match read_json_problem(&problem_path)? {
            JsonProblem::Qp { problem } => solver.solve_qp(problem)?,
            JsonProblem::Lp { problem } => solver.solve_lp(problem)?,
        },
        "mps" => {
            return Err(anyhow!(
                "El formato MPS todavía no está soportado por la interfaz gráfica."
            ));
        }
        other => {
            return Err(anyhow!(
                "Extensión de archivo desconocida: {}. Usa JSON o MPS.",
                other
            ));
        }
    };

    if let Some(path) = &output_path {
        write_solution(path, &solution)?;
    }

    let solution_json = if log_json {
        let json = serde_json::to_string_pretty(&solution)?;
        tracing::info!("solution-json={}", json);
        Some(json)
    } else {
        None
    };

    tracing::info!(
        status = ?solution.status,
        objective = %solution.objective_value,
        iterations = solution.iterations,
        "solver finished"
    );

    Ok((solution, solution_json))
}

fn convert_image_with_gemini(image_path: &Path) -> Result<String> {
    let image_bytes =
        fs::read(image_path).map_err(|err| anyhow!("No se pudo leer la imagen: {}", err))?;
    let encoded_image = BASE64_STANDARD.encode(image_bytes);
    let mime_type = guess_mime_type(image_path);

    let client = Client::builder().timeout(Duration::from_secs(45)).build()?;

    let url = format!(
        "{}/{}:generateContent?key={}",
        GEMINI_ENDPOINT, GEMINI_MODEL, GEMINI_API_KEY
    );

    let payload = serde_json::json!({
        "system_instruction": {
            "role": "system",
            "parts": [{ "text": GEMINI_SYSTEM_PROMPT }]
        },
        "contents": [{
            "role": "user",
            "parts": [
                { "text": GEMINI_USER_PROMPT },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded_image
                    }
                }
            ]
        }]
    });

    let response = client.post(url).json(&payload).send()?;
    let status = response.status();
    let body = response.text()?;

    if !status.is_success() {
        return Err(anyhow!(
            "Gemini devolvio un error {}: {}",
            status.as_u16(),
            body
        ));
    }

    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|err| anyhow!("Respuesta inesperada de Gemini: {}", err))?;

    let text = value
        .get("candidates")
        .and_then(|candidates| candidates.as_array())
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(|parts| parts.as_array())
        .and_then(|parts| {
            parts
                .iter()
                .find_map(|part| part.get("text").and_then(|text| text.as_str()))
        })
        .ok_or_else(|| anyhow!("Gemini no devolvio una respuesta de texto util."))?;

    Ok(text.trim().to_string())
}

fn guess_mime_type(path: &Path) -> &'static str {
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        let ext = ext.to_ascii_lowercase();
        match ext.as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "bmp" => "image/bmp",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "heic" => "image/heic",
            _ => "application/octet-stream",
        }
    } else {
        "application/octet-stream"
    }
}

fn parse_optional_f64(input: &str) -> Result<Option<f64>, String> {
    parse_optional(input, |value| {
        value.parse::<f64>().map_err(|err| err.to_string())
    })
}

fn parse_optional_usize(input: &str) -> Result<Option<usize>, String> {
    parse_optional(input, |value| {
        value.parse::<usize>().map_err(|err| err.to_string())
    })
}

fn parse_optional_u64(input: &str) -> Result<Option<u64>, String> {
    parse_optional(input, |value| {
        value.parse::<u64>().map_err(|err| err.to_string())
    })
}

fn parse_optional<T, F>(input: &str, parser: F) -> Result<Option<T>, String>
where
    F: Fn(&str) -> Result<T, String>,
{
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    parser(trimmed).map(Some)
}
