/// cloak-nerd — GLiNER PII inference sidecar for Cloak
///
/// Reads NDJSON requests from stdin, returns NDJSON responses on stdout.
/// Each input line: {"text": "...", "labels": ["NAME", "EMAIL", ...]}
/// Each output line: {"entities": [{"start":5,"end":15,"type":"NAME","score":0.95},...]}
use anyhow::{Context, Result};
use clap::Parser;
use ort::session::Session;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokenizers::Tokenizer;

// CLI

#[derive(Parser)]
#[command(name = "cloak-nerd", about = "GLiNER PII inference sidecar")]
struct Cli {
    /// Path to model.onnx
    #[arg(long)]
    model: PathBuf,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: PathBuf,

    /// Path to gliner_config.json (ent_token, sep_token)
    #[arg(long)]
    config: PathBuf,

    /// Entity score threshold (0.0–1.0), applied after sigmoid
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Maximum number of spans per text
    #[arg(long, default_value = "32")]
    max_spans: usize,
}

// GLiNER model config — the subset of gliner_config.json this binary needs.
// Must match the checkpoint the .onnx/tokenizer.json were exported from,
// since the prompt format is baked into how the model was trained.

#[derive(Deserialize)]
struct GlinerConfig {
    ent_token: String,
    sep_token: String,
    #[serde(default = "default_max_len")]
    max_len: usize,
}

fn default_max_len() -> usize {
    2048
}

// Wire types — the JSON protocol over stdin/stdout

#[derive(Deserialize)]
struct Request {
    text: String,
    labels: Vec<String>,
}

#[derive(Serialize)]
struct Response {
    entities: Vec<Entity>,
}

#[derive(Serialize)]
struct Entity {
    start: usize,
    end: usize,
    #[serde(rename = "type")]
    entity_type: String,
    score: f32,
}

// GLiNER label mapping

fn to_gliner_label(cloak_type: &str) -> &str {
    match cloak_type {
        "NAME" => "person",
        "USERNAME" => "username",
        "ADDRESS" => "address",
        "HOSTNAME" => "hostname",
        "ORGANIZATION" => "organization",
        "EMAIL" => "email address",
        "PHONE" => "phone number",
        "IPv4" => "ip address",
        _ => "person",
    }
}

// Word splitting — GLiNER classifies spans over words, not raw subword
// token positions. Must match the training-time WhitespaceTokenSplitter
// exactly: `\w+(?:[-_]\w+)*|\S`.

struct Word {
    text: String,
    start: usize,
    end: usize,
}

fn split_words(text: &str, word_re: &Regex) -> Vec<Word> {
    word_re
        .find_iter(text)
        .map(|m| Word {
            text: m.as_str().to_string(),
            start: m.start(),
            end: m.end(),
        })
        .collect()
}

// Inference

fn run_inference(
    text: &str,
    labels: &[String],
    session: &mut Session,
    tokenizer: &Tokenizer,
    word_re: &Regex,
    cfg: &GlinerConfig,
    threshold: f32,
    max_spans: usize,
) -> Result<Response> {
    let mut words = split_words(text, word_re);
    words.truncate(cfg.max_len);
    let num_text_words = words.len();

    if num_text_words == 0 {
        return Ok(Response { entities: vec![] });
    }

    // Prompt = [ENT, label_1, ENT, label_2, ..., ENT, label_N, SEP] followed
    // by the real text words. Each entry is one pre-tokenized "word" — a
    // multi-word label like "email address" stays a single unit, matching
    // GLiNER's `is_split_into_words=True` prompt construction.
    let gliner_labels: Vec<&str> = labels.iter().map(|l| to_gliner_label(l)).collect();
    let mut all_words: Vec<&str> = Vec::with_capacity(gliner_labels.len() * 2 + 1 + num_text_words);
    for label in &gliner_labels {
        all_words.push(cfg.ent_token.as_str());
        all_words.push(label);
    }
    all_words.push(cfg.sep_token.as_str());
    let prompt_len = all_words.len();
    for w in &words {
        all_words.push(w.text.as_str());
    }

    let encoding = tokenizer
        .encode(all_words, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let seq_len = input_ids.len();
    let attention_mask: Vec<i64> = vec![1; seq_len];

    // words_mask: 1-indexed word position (after skipping the prompt) on
    // the first subword token of each word; 0 for prompt words, specials,
    // and continuation subwords. Mirrors GLiNER's `prepare_word_mask`.
    let mut words_mask: Vec<i64> = Vec::with_capacity(seq_len);
    let mut prev_word_id: Option<u32> = None;
    let mut seen_words: usize = 0;
    for wid in encoding.get_word_ids() {
        match wid {
            None => words_mask.push(0),
            Some(w) => {
                let is_new_word = Some(*w) != prev_word_id;
                if is_new_word {
                    seen_words += 1;
                }
                if is_new_word && seen_words > prompt_len {
                    words_mask.push((seen_words - prompt_len) as i64);
                } else {
                    words_mask.push(0);
                }
                prev_word_id = Some(*w);
            }
        }
    }

    let text_lengths: Vec<i64> = vec![num_text_words as i64];

    let input_ids_arr = ndarray::Array2::from_shape_vec((1, seq_len), input_ids)
        .context("reshape input_ids")?;
    let attention_mask_arr = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask)
        .context("reshape attention_mask")?;
    let words_mask_arr = ndarray::Array2::from_shape_vec((1, seq_len), words_mask)
        .context("reshape words_mask")?;
    let text_lengths_arr = ndarray::Array2::from_shape_vec((1, 1), text_lengths)
        .context("reshape text_lengths")?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ort::value::Tensor::from_array(input_ids_arr).context("input_ids tensor")?,
            "attention_mask" => ort::value::Tensor::from_array(attention_mask_arr).context("attention_mask tensor")?,
            "words_mask" => ort::value::Tensor::from_array(words_mask_arr).context("words_mask tensor")?,
            "text_lengths" => ort::value::Tensor::from_array(text_lengths_arr).context("text_lengths tensor")?,
        ])
        .context("ONNX inference failed")?;

    // logits shape: [1, num_text_words, num_classes, 3], where the trailing
    // 3 are [start, end, inside] BIO-style tag logits per word per class
    // (verified against the real checkpoint's UniEncoderTokenGLiNER/
    // TokenDecoder — this architecture has no span_idx/span_mask/max_width
    // at all, unlike the span-enumeration GLiNER variant).
    let (logits_shape, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .context("extract logits")?;
    let num_classes = logits_shape.get(2).copied().unwrap_or(1) as usize;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let tag = |word: usize, class: usize, t: usize| -> f32 {
        sigmoid(logits_data[(word * num_classes + class) * 3 + t])
    };

    // For every (start_word, class) pair scoring above threshold on the
    // "start" tag, pair it with every (end_word, class) above threshold on
    // "end" (same class, end >= start), then require every word in between
    // to clear threshold on "inside". Span score = min over all three
    // (mirrors GLiNER's TokenDecoder._calculate_span_score exactly).
    let mut start_cands: Vec<(usize, usize, f32)> = Vec::new(); // (word, class, prob)
    let mut end_cands: Vec<(usize, usize, f32)> = Vec::new();
    for w in 0..num_text_words {
        for c in 0..num_classes {
            let start_p = tag(w, c, 0);
            if start_p > threshold {
                start_cands.push((w, c, start_p));
            }
            let end_p = tag(w, c, 1);
            if end_p > threshold {
                end_cands.push((w, c, end_p));
            }
        }
    }

    let mut candidates: Vec<(f32, usize, usize, usize)> = Vec::new();
    for &(st, cls, start_p) in &start_cands {
        for &(ed, cls2, end_p) in &end_cands {
            if cls2 != cls || ed < st {
                continue;
            }
            let mut inside_min = f32::INFINITY;
            let mut inside_ok = true;
            for pos in st..=ed {
                let p = tag(pos, cls, 2);
                if p < threshold {
                    inside_ok = false;
                    break;
                }
                inside_min = inside_min.min(p);
            }
            if !inside_ok {
                continue;
            }
            let score = start_p.min(end_p).min(inside_min);
            candidates.push((score, st, ed, cls));
        }
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy flat-NER decode: highest-scoring spans win, overlapping
    // lower-scored spans are dropped (mirrors GLiNER's `greedy_search`).
    let mut accepted: Vec<(f32, usize, usize, usize)> = Vec::new();
    for cand in candidates {
        let (_, s, e, _) = cand;
        let overlaps = accepted
            .iter()
            .any(|(_, as_, ae, _)| s <= *ae && *as_ <= e);
        if !overlaps {
            accepted.push(cand);
        }
    }
    accepted.sort_by_key(|(_, s, _, _)| *s);
    accepted.truncate(max_spans);

    let entities = accepted
        .into_iter()
        .map(|(score, s, e, c)| {
            let cloak_type = labels.get(c).map(String::as_str).unwrap_or("NAME");
            Entity {
                start: words[s].start,
                end: words[e].end,
                entity_type: cloak_type.to_string(),
                score,
            }
        })
        .collect();

    Ok(Response { entities })
}

// Main — NDJSON loop

fn main() -> Result<()> {
    let cli = Cli::parse();

    // eprintln!("⟳  Loading GLiNER config: {}", cli.config.display());
    let cfg_json = std::fs::read_to_string(&cli.config).context("read gliner config")?;
    let cfg: GlinerConfig = serde_json::from_str(&cfg_json).context("parse gliner config")?;

    // Model was trained/exported with this exact whitespace word splitter;
    // any other splitting scheme would misalign span_idx with the model's
    // word-level span grid.
    let word_re = Regex::new(r"\w+(?:[-_]\w+)*|\S").context("build word splitter regex")?;

    // eprintln!("⟳  Loading ONNX model: {}", cli.model.display());
    let mut session = Session::builder()?
        .commit_from_file(&cli.model)
        .context("failed to load ONNX model")?;

    // eprintln!("⟳  Loading tokenizer: {}", cli.tokenizer.display());
    let tokenizer = Tokenizer::from_file(&cli.tokenizer)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // eprintln!("✓  Ready — waiting for input on stdin...");

    use std::io::{BufRead, BufReader, Write};

    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();

    for line in reader.lines() {
        let line = line.context("read stdin")?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: Request = serde_json::from_str(&line).context("parse JSON request")?;

        let response = run_inference(
            &request.text,
            &request.labels,
            &mut session,
            &tokenizer,
            &word_re,
            &cfg,
            cli.threshold,
            cli.max_spans,
        )?;

        let json = serde_json::to_string(&response).context("serialize response")?;
        writeln!(writer, "{json}").context("write stdout")?;
        writer.flush().context("flush stdout")?;
    }

    Ok(())
}
